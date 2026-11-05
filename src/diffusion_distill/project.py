"""Подготовка данных, обучение и оценка моделей."""

import csv
import json
import shutil
import time
from pathlib import Path

import torch

from .artifacts import (
    choose_device,
    environment,
    fixed_latents,
    save_grid,
    sha256_file,
    state_hash,
    synchronize,
    write_json,
)
from .evaluation import benchmark, compute_fid, generate_images
from .schedule import (
    DDIMSchedule,
    coarsen_grid,
    leading_grid,
    sample_ddim,
)
from .training import PilotConfig, train_pilot

SEED = 20261104


def configure_cache(root):
    root = Path(root).resolve()
    torch.hub.set_dir(str(root / ".cache" / "torch"))
    return root


def prepare(root, count: int = 10000) -> dict:
    root = configure_cache(root)
    from huggingface_hub import snapshot_download, hf_hub_download
    from torchvision.datasets import CIFAR10
    from torchvision.datasets.utils import check_integrity

    model = json.loads((root / "configs/model.json").read_text())
    print(f"Model: {model['model_id']}, revision: {model['revision']}", flush=True)
    previous_path = root / "artifacts" / "prepare.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    snapshot_relative = previous.get("snapshot_relative")
    candidate = root / snapshot_relative if snapshot_relative else None
    if (
        candidate is not None
        and previous.get("model", {}).get("revision") == model["revision"]
        and all(
            (candidate / name).exists()
            for name in ("config.json", "scheduler_config.json", model["weights_file"])
        )
    ):
        snapshot = str(candidate)
    else:
        snapshot = snapshot_download(
            model["model_id"],
            revision=model["revision"],
            allow_patterns=[
                "config.json",
                "scheduler_config.json",
                "model_index.json",
                "README.md",
                model["weights_file"],
            ],
            cache_dir=root / ".cache" / "huggingface",
        )
    print("Dataset: CIFAR-10 train", flush=True)
    dataset_spec = root / "configs" / "dataset.json"
    if dataset_spec.exists():
        spec = json.loads(dataset_spec.read_text())
        archive_path = root / "data" / "cifar-10-python.tar.gz"
        if not check_integrity(archive_path, CIFAR10.tgz_md5):
            mirror = hf_hub_download(
                repo_id=spec["mirror_repo"],
                repo_type="dataset",
                revision=spec["mirror_revision"],
                filename="cifar-10-python.tar.gz",
                cache_dir=root / ".cache" / "huggingface",
            )
            if not check_integrity(mirror, CIFAR10.tgz_md5):
                raise ValueError(
                    "CIFAR-10 mirror differs from the official archive MD5"
                )
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mirror, archive_path)
    dataset = CIFAR10(root / "data", train=True, download=True)
    subset = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(SEED)
    )[:2048]
    from .artifacts import atomic_torch_save

    subset_path = root / "artifacts" / "subset_indices.pt"
    atomic_torch_save(subset_path, subset)
    noise_path = root / "artifacts" / f"noise-{count}.pt"
    _, noise = fixed_latents(noise_path, count, SEED)
    manifest = {
        "model": model,
        "snapshot_relative": str(Path(snapshot).relative_to(root)),
        "weights_sha256": sha256_file(Path(snapshot) / model["weights_file"]),
        "dataset": {
            "name": "CIFAR-10",
            "source": "https://www.cs.toronto.edu/~kriz/cifar.html",
            "download": json.loads(dataset_spec.read_text())
            if dataset_spec.exists()
            else {"url": CIFAR10.url, "archive_md5": CIFAR10.tgz_md5},
            "train_count": len(dataset),
            "test_count": 10000,
            "training_subset_count": 2048,
            "subset_indices_sha256": sha256_file(
                root / "artifacts" / "subset_indices.pt"
            ),
            "conditions": "The official page requests citation of the technical report. A separate image-dataset license is not explicitly specified.",
            "normalization": "uint8 RGB -> float32 [-1,1]; no augmentation in pilot",
        },
        "noise": noise,
        "noise_file": f"artifacts/noise-{count}.pt",
        "teacher_grid": leading_grid(64),
        "clip_sample": False,
        "eta": 0.0,
        "prediction_type": "epsilon",
        "environment": environment(),
    }
    write_json(root / "artifacts" / "prepare.json", manifest)
    return manifest


def load_teacher(root, device="cpu"):
    root = configure_cache(root)
    from diffusers import DDPMScheduler, UNet2DModel

    manifest = json.loads((root / "artifacts" / "prepare.json").read_text())
    snapshot = root / manifest["snapshot_relative"]
    if (
        sha256_file(snapshot / manifest["model"]["weights_file"])
        != manifest["weights_sha256"]
    ):
        raise ValueError("teacher checkpoint hash mismatch")
    teacher = (
        UNet2DModel.from_pretrained(
            str(snapshot),
            local_files_only=True,
            use_safetensors=manifest["model"]["weights_file"].endswith(".safetensors"),
            low_cpu_mem_usage=False,
        )
        .to(device=device, dtype=torch.float32)
        .eval()
        .requires_grad_(False)
    )
    scheduler = DDPMScheduler.from_pretrained(str(snapshot), local_files_only=True)
    manifest["architecture"] = {
        "config": dict(teacher.config),
        "parameters": sum(p.numel() for p in teacher.parameters()),
        "student_trainable": "all UNet parameters; identical independent architecture",
        "teacher_trainable": 0,
    }
    write_json(root / "artifacts" / "architecture.json", manifest["architecture"])
    return teacher, DDIMSchedule(scheduler.alphas_cumprod), scheduler, manifest


def load_noise(root, manifest):
    path = Path(root) / manifest["noise_file"]
    if sha256_file(path) != manifest["noise"]["sha256"]:
        raise ValueError("fixed noise checksum differs from preparation manifest")
    z, _ = fixed_latents(path, manifest["noise"]["count"], manifest["noise"]["seed"])
    return z


def preview(root, device="auto", checkpoint=None, count=16) -> dict:
    root, device = configure_cache(root), choose_device(device)
    teacher, schedule, _, manifest = load_teacher(root, device)
    z = load_noise(root, manifest)[:count].to(device)
    grid = leading_grid(64)
    name = "teacher"
    source_hash, source_step = None, None
    if checkpoint:
        source_hash = sha256_file(checkpoint)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        source_step = saved["step"]
        teacher.load_state_dict(saved["ema"], strict=True)
        grid = tuple(saved["student_grid"])
        name = "student_ema"
    output = root / "artifacts" / "local" / name
    synchronize(device)
    began = time.perf_counter()
    result = sample_ddim(teacher, z, schedule, grid)
    synchronize(device)
    elapsed = time.perf_counter() - began
    save_grid(result.images, output / "grid.png")
    metrics = {
        "count": len(z),
        "nfe_per_image": result.nfe_per_image,
        "forward_calls": result.forward_calls,
        "sampling_seconds": elapsed,
        "grid": grid,
        "environment": environment(device),
        "weights": "pilot EMA" if checkpoint else manifest["model"],
        "weights_sha256": manifest["weights_sha256"],
        "checkpoint_sha256": source_hash,
        "checkpoint_step": source_step,
    }
    write_json(output / "preview.json", metrics)
    return metrics


def pilot(
    root,
    device="auto",
    max_steps=100,
    max_seconds=1200,
    resume=None,
    output_name="pilot",
) -> dict:
    root, device = configure_cache(root), choose_device(device)
    from torchvision.datasets import CIFAR10

    teacher, schedule, _, manifest = load_teacher(root, device)
    dataset = CIFAR10(root / "data", train=True, download=False)
    indices = torch.load(root / "artifacts" / "subset_indices.pt", weights_only=True)
    if (
        sha256_file(root / "artifacts" / "subset_indices.pt")
        != manifest["dataset"]["subset_indices_sha256"]
    ):
        raise ValueError("training subset checksum mismatch")
    images = (
        torch.from_numpy(dataset.data[indices.numpy()])
        .permute(0, 3, 1, 2)
        .float()
        .div(127.5)
        .sub(1)
    )
    config = PilotConfig(max_steps=max_steps, max_seconds=max_seconds)
    result = train_pilot(
        teacher,
        schedule,
        images,
        config,
        root / "artifacts" / "local" / output_name,
        device,
        resume=resume,
    )
    result["model"] = manifest["model"]
    write_json(root / "artifacts" / "local" / output_name / "result.json", result)
    return result


def baseline(root, device="auto", batch_size=32, skip_timing=False) -> dict:
    root, device = configure_cache(root), choose_device(device)
    teacher, schedule, ddpm_scheduler, manifest = load_teacher(root, device)
    z = load_noise(root, manifest)
    output = root / "artifacts" / "baseline"
    grid = leading_grid(64)
    generation = generate_images(
        teacher,
        z,
        schedule,
        grid,
        output,
        batch_size=batch_size,
        device=device,
        identity={
            "weights_sha256": manifest["weights_sha256"],
            "model_revision": manifest["model"]["revision"],
            "sampler": "DDIM epsilon eta0 clipFalse leading64",
        },
    )
    timing_path = output / "timing.json"
    timings = None
    if not skip_timing:
        if timing_path.exists():
            timings = json.loads(timing_path.read_text())
            if (
                timings["environment"] != environment(device)
                or timings.get("generation_identity") != generation["identity"]
            ):
                raise ValueError(
                    "timing metadata differs from the generation or environment"
                )
        else:
            timings = benchmark(teacher, z, schedule, grid, device=device)
            timings["generation_identity"] = generation["identity"]
            write_json(timing_path, timings)
    fid_path = output / "fid.json"
    if fid_path.exists():
        fid_result = json.loads(fid_path.read_text())
        if fid_result.get("generation_identity") != generation["identity"]:
            raise ValueError("FID metadata differs from the generation")
    else:
        fid_result = compute_fid(output / "images", len(z), device=device)
        fid_result["generation_identity"] = generation["identity"]
        write_json(fid_path, fid_result)
    # Контрольная сетка исходного DDPM-1000.
    control_path = output / "ddpm_control.json"
    if not control_path.exists():
        from diffusers import DDPMPipeline

        pipe = DDPMPipeline(unet=teacher, scheduler=ddpm_scheduler).to(device)
        control = pipe(
            batch_size=4,
            num_inference_steps=1000,
            generator=torch.Generator(device="cpu").manual_seed(SEED),
            output_type="np",
        )
        control_images = torch.from_numpy(control.images).permute(0, 3, 1, 2) * 2 - 1
        save_grid(control_images, output / "ddpm_control.png", columns=2, scale=6)
        write_json(
            control_path,
            {
                "sampler": "original stochastic DDPM",
                "steps": 1000,
                "clip_sample": True,
                "count": 4,
                "seed": SEED,
                "purpose": "visual control only, not FID",
            },
        )
    result = {
        "generation": generation,
        "timing": timings,
        "fid": fid_result,
        "model": manifest["model"],
        "weights_sha256": manifest["weights_sha256"],
    }
    write_json(output / "result.json", result)
    if timings:
        with (output / "timing.csv").open("w", newline="") as stream:
            columns = (
                "batch_size",
                "mean_batch_seconds",
                "std_batch_seconds",
                "mean_image_seconds",
                "images_per_second",
                "nfe_per_image",
            )
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(
                {key: row[key] for key in columns} for row in timings["timings"]
            )
    with (output / "fid.csv").open("w", newline="") as stream:
        columns = (
            "metric",
            "mode",
            "value",
            "generated_count",
            "reference",
            "reference_count",
        )
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerow({key: fid_result[key] for key in columns})
    return result


def evaluate_student(root, device="auto", batch_size=32, checkpoint=None) -> dict:
    """Оценка EMA-ученика и сопоставимые замеры времени учителя."""
    root, device = configure_cache(root), choose_device(device)
    checkpoint = (
        Path(checkpoint) if checkpoint else root / "artifacts/local/pilot/checkpoint.pt"
    )
    checkpoint_hash = sha256_file(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if sha256_file(checkpoint) != checkpoint_hash:
        raise ValueError("checkpoint changed while loading")
    model, schedule, _, manifest = load_teacher(root, device)
    teacher_hash = state_hash(model.state_dict())
    teacher_grid = leading_grid(64)
    if (
        saved["teacher_hash"] != teacher_hash
        or tuple(saved["teacher_grid"]) != teacher_grid
    ):
        raise ValueError("student checkpoint does not match the frozen teacher/grid")
    grid = coarsen_grid(teacher_grid)
    if tuple(saved["student_grid"]) != grid or saved["step"] < 1:
        raise ValueError(
            "student checkpoint has an unrelated grid or no training steps"
        )
    source = {
        "sha256": checkpoint_hash,
        "step": saved["step"],
        "weights": "EMA",
        "teacher_state_sha256": teacher_hash,
    }
    model.load_state_dict(saved["ema"], strict=True)
    model.eval().requires_grad_(False)
    del saved
    z = load_noise(root, manifest)
    output = root / "artifacts/student_baseline"
    write_json(output / "checkpoint.json", source)
    generation = generate_images(
        model,
        z,
        schedule,
        grid,
        output,
        batch_size=batch_size,
        device=device,
        identity={
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_step": source["step"],
            "weights": "EMA",
            "teacher_weights_sha256": manifest["weights_sha256"],
            "model_revision": manifest["model"]["revision"],
            "sampler": "DDIM epsilon eta0 clipFalse nested32",
        },
    )

    def measured_timing(model, grid, path, identity):
        if path.exists():
            value = json.loads(path.read_text())
            if (
                value.get("environment") != environment(device)
                or value.get("evaluation_identity") != identity
            ):
                raise ValueError(
                    "timing inputs or environment differ from the saved run"
                )
        else:
            value = benchmark(model, z, schedule, grid, device=device)
            value["evaluation_identity"] = identity
            write_json(path, value)
        return value

    print("Student timing", flush=True)
    timings = measured_timing(
        model, grid, output / "timing.json", generation["identity"]
    )
    # При замере на устройстве находится одна U-Net.
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    model, _, _, _ = load_teacher(root, device)
    print("Teacher timing", flush=True)
    teacher_identity = {
        "model_state_sha256": teacher_hash,
        "grid": list(teacher_grid),
        "noise_sha256": manifest["noise"]["sha256"],
        "comparison_student": generation["identity"],
    }
    teacher_timing = measured_timing(
        model, teacher_grid, output / "teacher_timing.json", teacher_identity
    )
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    print("Student FID", flush=True)
    fid_path = output / "fid.json"
    if fid_path.exists():
        fid_result = json.loads(fid_path.read_text())
        if fid_result.get("generation_identity") != generation["identity"]:
            raise ValueError("FID belongs to a different student generation")
    else:
        fid_result = compute_fid(output / "images", len(z), device=device)
        fid_result["generation_identity"] = generation["identity"]
        write_json(fid_path, fid_result)
    teacher_rows = {r["batch_size"]: r for r in teacher_timing["timings"]}
    speedup = [
        {
            "batch_size": r["batch_size"],
            "teacher_batch_seconds": teacher_rows[r["batch_size"]][
                "mean_batch_seconds"
            ],
            "student_batch_seconds": r["mean_batch_seconds"],
            "speedup": teacher_rows[r["batch_size"]]["mean_batch_seconds"]
            / r["mean_batch_seconds"],
        }
        for r in timings["timings"]
    ]
    result = {
        "role": "pilot_student_ema",
        "checkpoint": source,
        "generation": generation,
        "timing": timings,
        "teacher_timing": teacher_timing,
        "fid": fid_result,
        "speedup": speedup,
        "model": manifest["model"],
    }
    write_json(output / "result.json", result)
    for name, rows, columns in (
        (
            "timing",
            timings["timings"],
            (
                "batch_size",
                "mean_batch_seconds",
                "std_batch_seconds",
                "mean_image_seconds",
                "images_per_second",
                "nfe_per_image",
            ),
        ),
        (
            "teacher_timing",
            teacher_timing["timings"],
            (
                "batch_size",
                "mean_batch_seconds",
                "std_batch_seconds",
                "mean_image_seconds",
                "images_per_second",
                "nfe_per_image",
            ),
        ),
        (
            "fid",
            [fid_result],
            (
                "metric",
                "mode",
                "value",
                "generated_count",
                "reference",
                "reference_count",
            ),
        ),
        (
            "speedup",
            speedup,
            ("batch_size", "teacher_batch_seconds", "student_batch_seconds", "speedup"),
        ),
    ):
        with (output / (name + ".csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows({key: row[key] for key in columns} for row in rows)
    return result
