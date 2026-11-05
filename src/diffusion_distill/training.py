"""Обучение ученика Progressive Distillation и сохранение состояния."""

import copy
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from .artifacts import (
    atomic_torch_save,
    environment,
    memory_stats,
    sha256_file,
    state_hash,
    synchronize,
    write_json,
)
from .schedule import (
    coarsen_grid,
    distillation_target,
    leading_grid,
    predict_epsilon,
    triplets,
)


@dataclass(frozen=True)
class PilotConfig:
    seed: int = 20261104
    max_steps: int = 100
    max_seconds: float = 1200
    batch_size: int = 2
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    teacher_steps: int = 64
    ema_decay: float = 0.95


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu(v) for v in value)
    return value


def _backend_rng(device):
    if device.type == "mps":
        return torch.mps.get_rng_state()
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device)
    return None


def train_pilot(
    teacher,
    schedule,
    images: torch.Tensor,
    config: PilotConfig,
    output_dir,
    device="cpu",
    resume=None,
) -> dict:
    """max_steps задаёт итоговое число обновлений, max_seconds — время текущего запуска.

    Изображения выбираются с возвращением. Для данных, времени и шума
    используются независимые генераторы случайных чисел.
    """
    if (
        config.batch_size < 1
        or config.max_steps < 1
        or config.max_seconds < 0
        or len(images) < 1
    ):
        raise ValueError("invalid pilot budget or empty image subset")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "checkpoint.pt"
    if resume is None and checkpoint.exists():
        resume = checkpoint
    device = torch.device(device)
    teacher = teacher.to(device).eval().requires_grad_(False)
    teacher_hash = state_hash(teacher.state_dict())
    student = copy.deepcopy(teacher).train().requires_grad_(True)
    initial_student_hash = state_hash(student.state_dict())
    ema = copy.deepcopy(student).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        foreach=False,
    )
    generators = {
        name: torch.Generator().manual_seed(config.seed + i)
        for i, name in enumerate(("data", "time", "noise"))
    }
    grid = leading_grid(config.teacher_steps, len(schedule.alphas_cumprod))
    triples = torch.tensor(triplets(grid), dtype=torch.long)
    image_hash = state_hash({"images": images})
    torch.manual_seed(config.seed)
    if device.type == "mps":
        torch.mps.manual_seed(config.seed)
    history, step, previous_seconds = [], 0, 0.0
    resume_source_hash = None
    if resume:
        resume_source_hash = sha256_file(resume)
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if sha256_file(resume) != resume_source_hash:
            raise ValueError("checkpoint changed while loading")
        stable_keys = (
            "seed",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "grad_clip",
            "teacher_steps",
            "ema_decay",
        )
        if (
            any(saved["config"][k] != asdict(config)[k] for k in stable_keys)
            or saved["teacher_hash"] != teacher_hash
            or saved["image_hash"] != image_hash
            or saved["teacher_grid"] != grid
        ):
            raise ValueError(
                "checkpoint does not match teacher/data/training configuration"
            )
        result_path = output_dir / "result.json"
        if (
            saved["step"] >= config.max_steps
            and Path(resume).resolve() == checkpoint.resolve()
            and result_path.exists()
        ):
            result = json.loads(result_path.read_text())
            if (
                result.get("checkpoint_sha256") == resume_source_hash
                and result.get("completed_steps") == saved["step"]
            ):
                return {
                    **result,
                    "steps_this_invocation": 0,
                    "resume_source_sha256": resume_source_hash,
                }
        student.load_state_dict(saved["student"], strict=True)
        ema.load_state_dict(saved["ema"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and key != "step":
                    state[key] = value.to(device)
        for name, generator in generators.items():
            generator.set_state(saved["generators"][name])
        torch.set_rng_state(saved["cpu_rng"])
        if saved["device"] == str(device) and saved["backend_rng"] is not None:
            if device.type == "mps":
                torch.mps.set_rng_state(saved["backend_rng"])
            elif device.type == "cuda":
                torch.cuda.set_rng_state(saved["backend_rng"], device)
        history, step = saved["history"], saved["step"]
        previous_seconds = saved["training_seconds"]
        initial_student_hash = saved["initial_student_hash"]
    started_step = step
    synchronize(device)
    started = time.perf_counter()

    def save():
        atomic_torch_save(
            checkpoint,
            _cpu(
                {
                    "format_version": 1,
                    "student": student.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": asdict(config),
                    "step": step,
                    "teacher_grid": grid,
                    "student_grid": coarsen_grid(grid),
                    "teacher_hash": teacher_hash,
                    "initial_student_hash": initial_student_hash,
                    "image_hash": image_hash,
                    "generators": {k: g.get_state() for k, g in generators.items()},
                    "cpu_rng": torch.get_rng_state(),
                    "backend_rng": _backend_rng(device),
                    "device": str(device),
                    "history": history,
                    "training_seconds": previous_seconds
                    + time.perf_counter()
                    - started,
                }
            ),
        )

    fields = ("step", "loss", "grad_norm", "seconds", "elapsed_seconds")
    with (output_dir / "training.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
        while (
            step < config.max_steps
            and time.perf_counter() - started < config.max_seconds
        ):
            step_started = time.perf_counter()
            indices = torch.randint(
                len(images), (config.batch_size,), generator=generators["data"]
            )
            clean = images[indices].to(device)
            selected = torch.randint(
                len(triples), (config.batch_size,), generator=generators["time"]
            )
            t, middle, next_t = triples[selected].unbind(dim=1)
            noise = torch.randn(
                clean.shape, generator=generators["noise"], dtype=clean.dtype
            ).to(device)
            noisy = schedule.add_noise(clean, noise, t)
            target = distillation_target(teacher, noisy, t, middle, next_t, schedule)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(predict_epsilon(student, noisy, t), target.epsilon)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite PD loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                student.parameters(), config.grad_clip, error_if_nonfinite=True
            )
            optimizer.step()
            with torch.no_grad():
                for averaged, current in zip(ema.parameters(), student.parameters()):
                    averaged.lerp_(current, 1 - config.ema_decay)
                for averaged, current in zip(ema.buffers(), student.buffers()):
                    averaged.copy_(current)
            synchronize(device)
            step += 1
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "grad_norm": float(grad_norm),
                "seconds": time.perf_counter() - step_started,
                "elapsed_seconds": previous_seconds + time.perf_counter() - started,
            }
            history.append(row)
            writer.writerow(row)
            stream.flush()
            if step == started_step + 1 or step % 10 == 0:
                print(
                    f"step={step}/{config.max_steps} loss={row['loss']:.6f} grad={row['grad_norm']:.4f} seconds={row['seconds']:.2f}",
                    flush=True,
                )
            if step % 25 == 0:
                save()
    save()
    result = {
        "completed_steps": step,
        "steps_this_invocation": step - started_step,
        "stop_reason": "max_steps" if step >= config.max_steps else "time_budget",
        "training_seconds": previous_seconds + time.perf_counter() - started,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "resume_source_sha256": resume_source_hash,
        "teacher_unchanged": state_hash(teacher.state_dict()) == teacher_hash,
        "student_changed": state_hash(student.state_dict()) != initial_student_hash,
        "teacher_hash": teacher_hash,
        "history": history,
        "config": asdict(config),
        "teacher_grid": grid,
        "student_grid": coarsen_grid(grid),
        "environment": environment(device),
        "memory": memory_stats(device),
    }
    write_json(output_dir / "result.json", result)
    return result
