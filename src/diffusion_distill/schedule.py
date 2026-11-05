"""DDIM-переходы и аналитическая цель Progressive Distillation.

alpha = sqrt(alphas_cumprod), sigma = sqrt(1-alphas_cumprod).
Индекс -1 соответствует чистому изображению: alpha=1, sigma=0.
"""

from dataclasses import dataclass

import torch


def leading_grid(steps: int = 64, train_steps: int = 1000) -> tuple[int, ...]:
    if steps < 1 or steps > train_steps:
        raise ValueError("steps must be in [1, train_steps]")
    stride = train_steps // steps
    return tuple(i * stride for i in reversed(range(steps))) + (-1,)


def validate_grid(grid: tuple[int, ...]) -> None:
    if (
        len(grid) < 2
        or grid[-1] != -1
        or any(a <= b for a, b in zip(grid[:-1], grid[1:]))
    ):
        raise ValueError("grid must be strictly decreasing and end at -1")
    if any(t < 0 for t in grid[:-1]):
        raise ValueError("only the final endpoint may be negative")


def coarsen_grid(grid: tuple[int, ...]) -> tuple[int, ...]:
    validate_grid(grid)
    if (len(grid) - 1) % 2:
        raise ValueError("two-to-one distillation needs an even number of transitions")
    return grid[:-1:2] + (-1,)


def triplets(grid: tuple[int, ...]) -> tuple[tuple[int, int, int], ...]:
    coarsen_grid(grid)
    return tuple(
        (grid[i], grid[i + 1], grid[i + 2]) for i in range(0, len(grid) - 1, 2)
    )


class DDIMSchedule:
    def __init__(self, alphas_cumprod: torch.Tensor):
        self.alphas_cumprod = alphas_cumprod.detach().cpu().clone()
        if self.alphas_cumprod.ndim != 1 or not torch.all(
            (self.alphas_cumprod > 0) & (self.alphas_cumprod <= 1)
        ):
            raise ValueError(
                "alphas_cumprod must be a positive one-dimensional VP schedule"
            )

    def _times(self, t, batch: int) -> torch.Tensor:
        values = torch.as_tensor(t, dtype=torch.long, device="cpu").reshape(-1)
        if (
            values.numel() not in (1, batch)
            or torch.any(values < -1)
            or torch.any(values >= len(self.alphas_cumprod))
        ):
            raise ValueError("invalid timestep or timestep batch shape")
        return values

    def coefficients(self, t, like: torch.Tensor):
        values = self._times(t, len(like))
        a2 = self.alphas_cumprod[values.clamp_min(0)]
        a2 = torch.where(values == -1, torch.ones_like(a2), a2)
        a2 = a2.to(device=like.device, dtype=like.dtype).reshape(
            -1, *([1] * (like.ndim - 1))
        )
        return a2.sqrt(), (1 - a2).sqrt()

    def x0(self, x: torch.Tensor, epsilon: torch.Tensor, t) -> torch.Tensor:
        alpha, sigma = self.coefficients(t, x)
        return (x - sigma * epsilon) / alpha

    def step(self, x: torch.Tensor, epsilon: torch.Tensor, t, next_t) -> torch.Tensor:
        t_values, s_values = self._times(t, len(x)), self._times(next_t, len(x))
        if (
            torch.any(t_values < 0)
            or torch.any(s_values >= t_values)
            or epsilon.shape != x.shape
        ):
            raise ValueError("DDIM needs matching tensors and -1 <= next_t < t")
        alpha_s, sigma_s = self.coefficients(s_values, x)
        return alpha_s * self.x0(x, epsilon, t_values) + sigma_s * epsilon

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t) -> torch.Tensor:
        alpha, sigma = self.coefficients(t, x0)
        return alpha * x0 + sigma * noise


def predict_epsilon(model, x: torch.Tensor, t) -> torch.Tensor:
    times = torch.as_tensor(t, dtype=torch.long, device=x.device)
    result = model(x, times)
    return result if isinstance(result, torch.Tensor) else result.sample


@dataclass
class DistillationTarget:
    epsilon: torch.Tensor
    x0: torch.Tensor
    endpoint: torch.Tensor


@torch.no_grad()
def distillation_target(
    teacher, x, t, middle_t, next_t, schedule: DDIMSchedule
) -> DistillationTarget:
    middle = schedule.step(x, predict_epsilon(teacher, x, t), t, middle_t)
    endpoint = schedule.step(
        middle, predict_epsilon(teacher, middle, middle_t), middle_t, next_t
    )
    alpha_t, sigma_t = schedule.coefficients(t, x)
    alpha_s, sigma_s = schedule.coefficients(next_t, x)
    ratio = sigma_s / sigma_t
    denominator = alpha_s - ratio * alpha_t
    if torch.any(denominator.abs() < 1e-10) or torch.any(sigma_t == 0):
        raise ValueError("degenerate two-to-one target for the supplied timesteps")
    target_x0 = (endpoint - ratio * x) / denominator
    epsilon = (x - alpha_t * target_x0) / sigma_t
    return DistillationTarget(epsilon, target_x0, endpoint)


@dataclass
class SamplingResult:
    images: torch.Tensor
    forward_calls: int
    nfe_per_image: int


@torch.inference_mode()
def sample_ddim(
    model, latents, schedule: DDIMSchedule, grid: tuple[int, ...]
) -> SamplingResult:
    validate_grid(grid)
    x = latents.clone()
    calls = 0
    for t, next_t in zip(grid[:-1], grid[1:]):
        x = schedule.step(x, predict_epsilon(model, x, t), t, next_t)
        calls += 1
    return SamplingResult(x, calls, calls)
