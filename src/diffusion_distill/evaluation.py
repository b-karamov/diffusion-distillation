"""Генерация изображений, измерение времени и расчёт Clean-FID."""

import json
import statistics
import time
from pathlib import Path

import torch
from PIL import Image

from .artifacts import (
    environment,
    memory_stats,
    save_grid,
    save_images,
    state_hash,
    synchronize,
    validate_images,
    write_json,
)
from .schedule import sample_ddim


def generate_images(
    model,
    latents,
    schedule,
    grid,
    output_dir,
    batch_size=32,
    device="cpu",
    identity=None,
):
    if batch_size < 1 or len(latents) < 1:
        raise ValueError("positive batch and sample counts required")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    folder = output_dir / "images"
    folder.mkdir(exist_ok=True)
    path = output_dir / "generation.json"
    run_identity = {
        **(identity or {}),
        "model_state_sha256": state_hash(model.state_dict()),
        "schedule_sha256": state_hash({"alphas_cumprod": schedule.alphas_cumprod}),
        "latents_hash": state_hash({"latents": latents}),
        "count": len(latents),
        "batch_size": batch_size,
        "grid": list(grid),
        "dtype": str(latents.dtype),
        "environment": environment(device),
    }
    if path.exists():
        result = json.loads(path.read_text())
        if result["identity"] != run_identity:
            raise ValueError(
                "generation inputs or environment differ from the saved run"
            )
        for i in range(result["completed_count"]):
            with Image.open(folder / f"{i:06d}.png") as image:
                if image.size != (32, 32) or image.mode != "RGB":
                    raise ValueError("committed generation was modified or damaged")
                image.verify()
    else:
        if list(folder.glob("*.png")):
            raise ValueError("images exist without a generation manifest")
        result = {
            "identity": run_identity,
            "completed_count": 0,
            "forward_calls": 0,
            "nfe_per_image": None,
            "sampling_seconds": 0.0,
        }
        write_json(path, result)
    for start in range(result["completed_count"], len(latents), batch_size):
        noise = latents[start : start + batch_size].to(device)
        synchronize(device)
        began = time.perf_counter()
        samples = sample_ddim(model, noise, schedule, grid)
        synchronize(device)
        elapsed = time.perf_counter() - began
        save_images(samples.images, folder, offset=start)
        if start == 0:
            save_grid(samples.images[:16], output_dir / "grid.png")
        result.update(
            completed_count=start + len(noise),
            forward_calls=result["forward_calls"] + samples.forward_calls,
            nfe_per_image=samples.nfe_per_image,
            sampling_seconds=result["sampling_seconds"] + elapsed,
        )
        write_json(path, result)
        if (
            start == 0
            or result["completed_count"] % (10 * batch_size) == 0
            or result["completed_count"] == len(latents)
        ):
            print(
                f"generated={result['completed_count']}/{len(latents)} NFE={samples.nfe_per_image} sampling_seconds={result['sampling_seconds']:.1f}",
                flush=True,
            )
    validate_images(folder, len(latents))
    return result


def benchmark(
    model,
    latents,
    schedule,
    grid,
    batch_sizes=(1, 32),
    warmup=5,
    series=3,
    repeats=10,
    device="cpu",
) -> dict:
    if min(warmup, series, repeats) < 0 or series * repeats < 1:
        raise ValueError("invalid benchmark repetition counts")
    result = {
        "environment": environment(device),
        "warmup_batches": warmup,
        "series": series,
        "batches_per_series": repeats,
        "boundary": "device-resident latents -> complete DDIM output; excludes I/O, load and FID",
        "timings": [],
    }
    for batch in batch_sizes:
        if batch < 1 or len(latents) < batch:
            raise ValueError("not enough fixed latents for benchmark batch size")
        noise = latents[:batch].to(device)
        for _ in range(warmup):
            sample_ddim(model, noise, schedule, grid)
        synchronize(device)
        if torch.device(device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        timings, series_means = [], []
        for run in range(series):
            values = []
            for _ in range(repeats):
                synchronize(device)
                began = time.perf_counter()
                sample = sample_ddim(model, noise, schedule, grid)
                synchronize(device)
                values.append(time.perf_counter() - began)
            timings.extend(values)
            series_means.append(statistics.mean(values))
            print(
                f"timing batch={batch} series={run + 1}/{series} mean_batch_seconds={series_means[-1]:.4f}",
                flush=True,
            )
        mean = statistics.mean(timings)
        result["timings"].append(
            {
                "batch_size": batch,
                "batch_seconds": timings,
                "series_mean_seconds": series_means,
                "mean_batch_seconds": mean,
                "std_batch_seconds": statistics.stdev(timings)
                if len(timings) > 1
                else 0.0,
                "mean_image_seconds": mean / batch,
                "images_per_second": batch / mean,
                "nfe_per_image": sample.nfe_per_image,
                "memory": memory_stats(device),
            }
        )
    return result


def compute_fid(image_dir, count: int, device="cuda") -> dict:
    validate_images(image_dir, count)
    from importlib.metadata import version
    from cleanfid import fid

    score = fid.compute_fid(
        str(image_dir),
        dataset_name="cifar10",
        dataset_res=32,
        dataset_split="train",
        mode="clean",
        device=torch.device(device),
        batch_size=64,
        num_workers=0,
        use_dataparallel=False,
    )
    return {
        "value": float(score),
        "metric": "Clean-FID",
        "mode": "clean",
        "generated_count": count,
        "reference": "CIFAR-10 train",
        "reference_count": 50000,
        "scipy_version": version("scipy"),
        "environment": environment(device),
    }
