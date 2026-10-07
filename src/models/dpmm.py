"""
Dirichlet Process Mixture Model (DPMM) over frozen DINOv2 patch embeddings.

Ported from the official code of:

    Schulthess, N. and Konukoglu, E., "Anomaly Detection by Clustering DINO
    Embeddings using a Dirichlet Process Mixture", MICCAI 2025.
    https://papers.miccai.org/miccai-2025/paper/2425_paper.pdf

    @InProceedings{Schulthess2025Anomaly,
        author = {Schulthess, Nico and Konukoglu, Ender},
        title = {{Anomaly Detection by Clustering DINO Embeddings using a
                  Dirichlet Process Mixture}},
        booktitle = {MICCAI 2025},
        year = {2025},
    }

Original code: https://github.com/nschutel/AnomalyDINO-DPMM (vendored locally
under anomalydino-dpmm/), itself based on AnomalyDINO
(https://github.com/dammsi/AnomalyDINO, Apache License 2.0). The
anomalydino-dpmm repository is released under CC-BY-NC 4.0.

This module keeps only what BMAD training/inference needs: a truncated
stick-breaking DP mixture fit online (each `step` is one E/M update, with an
exponentially-weighted or exp-decaying running average of the sufficient
statistics -- see `schedule`), plus patch-level anomaly scores. Training
and section 8's original evaluation (src/eval/bmad_dpmm.py) use the
log-likelihood (`sample_score`); the paper's own anomaly score (Sec. 2.2,
Eq. 11) is the cosine distance to the nearest component mean
(`cosine_distance_to_nearest_cluster`), with the Euclidean variant
(`distance_to_nearest_cluster`) kept for its Table 3 ablation -- both ported
from anomalydino-dpmm/src/DirichletProcessMixture/dpmm.py and used by
src/eval/bmad_dpmm_paper.py. Visualization and sampling are dropped.
"""

import math

import torch
from torch.distributions import MultivariateNormal
from torch.special import digamma


def _diag_gaussian_log_prob(data: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """
    data: Nx1xD, mean/var: KxD (diagonal covariance) -> log N(x | mean, diag(var)), size NxK.

    Mathematically identical to
    `MultivariateNormal(mean, scale_tril=diag(sqrt(var))).log_prob(data)`, but
    O(N*K*D) instead of O(N*K*D^2): the dense path constructs a full DxD
    (mostly-zero) Cholesky factor per component and solves against it, which
    dominates runtime once D is in the hundreds (DINOv2 patch embeddings).
    Only valid for diagonal/spherical covariances -- "full" cov_type still
    needs the general MultivariateNormal machinery.
    """
    diff2 = (data - mean[None, :, :]) ** 2          # NxKxD
    D = mean.shape[-1]
    return -0.5 * (
        (diff2 / var[None, :, :]).sum(-1)
        + torch.log(var).sum(-1)[None, :]
        + D * math.log(2 * math.pi)
    )


class DPMM:
    def __init__(
        self,
        K: int,                          # truncation level (max number of components)
        D: int,                          # data dimension
        update_rate: float | None = None,  # 0 <= rate <= 1; None -> standard (batch) EM
        schedule: str = "ema",           # "ema": exponential moving average, "exp": exponentially decaying step size
        alpha: float = 2.0,              # DP concentration parameter (higher -> more components)
        alpha_fixed: bool = False,
        eps: float | None = None,
        cov_type: str = "diag",          # "full", "diag" or "spherical"
        reg_covar: float = 1e-6,
        device: str = "cpu",
    ):
        assert alpha >= 1, f"alpha needs to be >= 1, but is {alpha}"

        self.K = K
        self.D = D
        self.update_rate = update_rate
        self.schedule = schedule.lower()
        self.alpha = alpha
        self.alpha_fixed = alpha_fixed
        self.eps = eps if eps is not None else torch.finfo(torch.float32).eps
        self.cov_type = cov_type
        self.reg_covar = reg_covar

        self.device = torch.device(device)
        self.initialize()

    def initialize(self, data: torch.Tensor | None = None):
        self.iterations = 0

        self.resp_stat = torch.zeros((self.K,), device=self.device)
        self.mean_stat = torch.zeros((self.K, self.D), device=self.device)
        self.cov_stat = torch.zeros((self.K, self.D, self.D), device=self.device)

        beta_distribution = torch.distributions.Beta(1, self.alpha)
        breaking_ratios = beta_distribution.sample((self.K,))
        breaking_ratios[-1] = 1
        self.v = breaking_ratios.to(self.device)

        self.mean = torch.randn(self.K, self.D, device=self.device)
        if data is None:
            self.cov = torch.eye(self.D, device=self.device)
        else:
            data = data.float()
            n_samples = min(self.K, data.shape[0])
            self.mean[:n_samples, :] = data[torch.randperm(n_samples), :]
            self.cov = torch.diag(torch.var(data, dim=0) + self.reg_covar)
        self.cov = self.cov[None, ...].repeat(self.K, 1, 1)
        self.compute_covariance_cholesky()

    def state_dict(self):
        return {
            "iterations": self.iterations,
            "resp_stat": self.resp_stat,
            "mean_stat": self.mean_stat,
            "cov_stat": self.cov_stat,
            "v": self.v,
            "mean": self.mean,
            "cov": self.cov,
        }

    def load_state_dict(self, state_dict):
        self.iterations = state_dict["iterations"]
        self.resp_stat = state_dict["resp_stat"]
        self.mean_stat = state_dict["mean_stat"]
        self.cov_stat = state_dict["cov_stat"]
        self.v = state_dict["v"]
        self.mean = state_dict["mean"]
        self.cov = state_dict["cov"]
        self.compute_covariance_cholesky()

    def get_step_size(self) -> float:
        if self.update_rate is None:
            return 1
        if self.schedule == "ema":
            return self.update_rate
        if self.schedule == "exp":
            return self.iterations ** (-self.update_rate)
        raise NotImplementedError(f"unknown schedule {self.schedule}")

    def calculate_pi(self) -> torch.Tensor:
        pi = torch.cumprod(1 - self.v, 0)
        pi = torch.roll(pi, 1)
        pi[0] = 1
        pi *= self.v
        pi /= pi.sum()
        return pi

    def calculate_log_pi(self) -> torch.Tensor:
        log_pi = torch.cumsum(torch.log(1 - self.v), 0)
        log_pi = torch.roll(log_pi, 1)
        log_pi[0] = 0
        log_pi += torch.log(self.v)
        return log_pi

    def get_weighted_log_prob(self, data: torch.Tensor, weight_threshold: float = 0.0) -> torch.Tensor:
        """data: Nx1xD -> log(pi_i * N(y | mu_i, cov_i)), size NxK."""
        log_pi = self.calculate_log_pi()
        component_mask = torch.exp(log_pi) > weight_threshold

        mean = self.mean[component_mask, :]
        weighted_log_prob = torch.full((data.shape[0], self.K), -torch.inf, device=self.device)

        if self.cov_type in ("diag", "spherical"):
            idxs = torch.arange(self.D, device=self.device)
            var = self.cov[component_mask][:, idxs, idxs]  # KxD
            log_prob = _diag_gaussian_log_prob(data, mean, var)
        else:
            gaussian = MultivariateNormal(mean, scale_tril=self.cov_chol[component_mask, :, :])
            log_prob = gaussian.log_prob(data)

        weighted_log_prob[:, component_mask] = log_pi[component_mask] + log_prob
        return weighted_log_prob

    def get_log_responsibilities(self, data: torch.Tensor) -> torch.Tensor:
        weighted_log_prob = self.get_weighted_log_prob(data)
        return weighted_log_prob - weighted_log_prob.logsumexp(1, keepdim=True)

    def score(self, data: torch.Tensor) -> torch.Tensor:
        """data: NxD -> mean log-likelihood over all N points (scalar)."""
        return self.sample_score(data).mean()

    def sample_score(self, data: torch.Tensor, weight_threshold: float = 0.0) -> torch.Tensor:
        """data: NxD -> log(sum_i pi_i * N(y | mu_i, cov_i)), size N."""
        data = data[:, None, :]
        weighted_log_prob = self.get_weighted_log_prob(data, weight_threshold)
        return torch.logsumexp(weighted_log_prob, dim=1)

    def distance_to_nearest_cluster(self, data: torch.Tensor, weight_threshold: float = 0.0) -> torch.Tensor:
        """data: NxD -> Euclidean distance to the nearest component mean with pi > weight_threshold, size N.
        Official dpmm.py `distance_to_nearest_cluster(covariance_weighted_norm=False)`."""
        means = self.mean[self.calculate_pi() > weight_threshold, :]
        return torch.cdist(data, means, compute_mode="donot_use_mm_for_euclid_dist").amin(1)

    def cosine_distance_to_nearest_cluster(self, data: torch.Tensor, weight_threshold: float = 0.0) -> torch.Tensor:
        """data: NxD -> 1 - max_k cos(data, mean_k) over components with pi > weight_threshold, size N
        (higher = more anomalous). Official dpmm.py `cosine_distance_to_nearest_cluster`; paper Eq. 11."""
        means = self.mean[self.calculate_pi() > weight_threshold, :]
        similarity = torch.nn.functional.normalize(data, dim=1) @ torch.nn.functional.normalize(means, dim=1).T
        return 1 - similarity.amax(1)

    def e_step(self, data: torch.Tensor) -> torch.Tensor:
        return self.get_log_responsibilities(data)

    def estimate_covariance_full(self):
        cov = self.cov_stat / (self.resp_stat[:, None, None] + self.eps)
        mean_cov = self.mean[:, None, :] * self.mean[:, None, :].transpose(-2, -1)
        cov -= mean_cov
        cov += torch.eye(self.D, device=self.device)[None, :, :] * self.reg_covar
        return cov

    def estimate_covariance_diagonal(self):
        idxs = torch.arange(self.D)
        data_squared = self.cov_stat[:, idxs, idxs] / (self.resp_stat[:, None] + self.eps)
        mean_squared = self.mean ** 2
        mixed_term = self.mean * self.mean_stat / (self.resp_stat[:, None] + self.eps)
        variance = data_squared + mean_squared - 2 * mixed_term
        cov = torch.zeros_like(self.cov)
        cov[:, idxs, idxs] = variance + self.reg_covar
        return cov

    def estimate_covariance_spherical(self):
        diagonal_cov = self.estimate_covariance_diagonal()
        variance = diagonal_cov.diagonal(dim1=1, dim2=2).mean(1)
        variance = variance[:, None].repeat(1, self.D)
        return torch.diag_embed(variance)

    def compute_covariance_cholesky(self):
        if self.cov_type == "full":
            self.cov_chol = torch.linalg.cholesky(self.cov, upper=False)
        elif self.cov_type in ("diag", "spherical"):
            self.cov_chol = torch.sqrt(self.cov)
        else:
            raise NotImplementedError(f"unknown cov_type {self.cov_type}")

    def m_step(self, data: torch.Tensor, resp: torch.Tensor):
        n_samples = data.shape[0]

        step_size = self.get_step_size()
        self.resp_stat *= (1 - step_size)
        self.mean_stat *= (1 - step_size)
        self.cov_stat *= (1 - step_size)

        data_cov = (data * data.transpose(-2, -1))[:, None, :, :]
        self.resp_stat += step_size * resp.mean(0)
        self.mean_stat += step_size * (data * resp[:, :, None]).mean(0)
        self.cov_stat += step_size * torch.einsum("...ndf,nk->kdf", data_cov.transpose(0, 1), resp) / n_samples

        self.mean = self.mean_stat / (self.resp_stat[:, None] + self.eps)

        if self.cov_type == "full":
            self.cov = self.estimate_covariance_full()
        elif self.cov_type == "diag":
            self.cov = self.estimate_covariance_diagonal()
        elif self.cov_type == "spherical":
            self.cov = self.estimate_covariance_spherical()
        self.compute_covariance_cholesky()

        double_sum = torch.cumsum(self.resp_stat.flip(0), 0).flip(0)
        double_sum[0] = 0
        double_sum = double_sum.roll(-1)
        self.v = self.resp_stat / (self.resp_stat + (self.alpha - 1) / n_samples + double_sum + self.eps)

        if not self.alpha_fixed:
            self.alpha = (self.K - 1) / (
                digamma(n_samples * (self.resp_stat + double_sum) + self.alpha + 1) -
                digamma(n_samples * double_sum + self.alpha)
            )[:-1].sum().item()

    def step(self, data: torch.Tensor):
        """One online E/M update. data: NxD patch embeddings from one batch."""
        self.iterations += 1
        data = data[:, None, :].float()
        log_resp = self.e_step(data)
        self.m_step(data, torch.exp(log_resp))
