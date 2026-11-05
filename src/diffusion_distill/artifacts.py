"""Данные запусков, контрольные суммы и сведения об устройстве."""

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import torch
from PIL import Image


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_hash(state) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(f"{key}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(
            value.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
    return digest.hexdigest()


def write_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temp.replace(path)


def atomic_torch_save(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def fixed_latents(path, count: int, seed: int = 20261104):
    path = Path(path)
    if count < 1:
        raise ValueError("noise count must be positive")
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["count"] != count or payload["seed"] != seed:
            raise ValueError("noise count or seed differs from the saved configuration")
        z = payload["latents"]
    else:
        z = torch.randn(
            (count, 3, 32, 32),
            generator=torch.Generator().manual_seed(seed),
            dtype=torch.float32,
        )
        atomic_torch_save(path, {"latents": z, "seed": seed, "count": count})
    if (
        z.shape != (count, 3, 32, 32)
        or z.dtype != torch.float32
        or not torch.isfinite(z).all()
    ):
        raise ValueError("invalid stored noise tensor")
    return z, {
        "count": count,
        "seed": seed,
        "dtype": "float32",
        "sha256": sha256_file(path),
    }


def uint8_images(images: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(images).all():
        raise ValueError("cannot save non-finite images")
    return (
        ((images.detach().cpu() + 1) * 127.5)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
    )


def save_images(images, directory, offset: int = 0) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for i, pixels in enumerate(uint8_images(images)):
        target = directory / f"{offset + i:06d}.png"
        temporary = target.with_suffix(".tmp")
        Image.fromarray(pixels.numpy()).save(temporary, format="PNG")
        temporary.replace(target)


def save_grid(images, path, columns: int = 4, scale: int = 4) -> None:
    array = uint8_images(images)
    h, w = array.shape[1:3]
    canvas = Image.new(
        "RGB", (columns * w, ((len(array) + columns - 1) // columns) * h), "white"
    )
    for i, pixels in enumerate(array):
        canvas.paste(
            Image.fromarray(pixels.numpy()), ((i % columns) * w, (i // columns) * h)
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.resize(
        (canvas.width * scale, canvas.height * scale), Image.Resampling.NEAREST
    ).save(path)


def validate_images(directory, expected_count: int) -> int:
    directory = Path(directory)
    files = sorted(directory.glob("*.png"))
    expected_names = [f"{i:06d}.png" for i in range(expected_count)]
    if expected_count < 1 or [p.name for p in files] != expected_names:
        raise ValueError(
            f"expected exactly {expected_count} consecutively numbered PNG images, found {len(files)}"
        )
    for path in files:
        with Image.open(path) as image:
            if image.size != (32, 32) or image.mode != "RGB":
                raise ValueError(f"invalid FID image dimensions/mode: {path.name}")
            image.verify()
    return len(files)


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this process")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this process")
    return torch.device(requested)


def synchronize(device) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def memory_stats(device) -> dict:
    device = torch.device(device)
    if device.type == "cuda":
        return {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "allocated_bytes": torch.cuda.memory_allocated(device),
        }
    if device.type == "mps":
        return {
            "current_allocated_bytes": torch.mps.current_allocated_memory(),
            "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
            "note": "MPS current values, not CUDA peak VRAM",
        }
    return {"note": "CPU: GPU memory not applicable"}


def environment(device="cpu") -> dict:
    device = torch.device(device)
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "torchvision",
            "diffusers",
            "huggingface-hub",
            "numpy",
            "clean-fid",
            "Pillow",
        )
    }
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor()
        if device.type == "mps"
        else None,
        "versions": versions,
    }
