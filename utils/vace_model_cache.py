"""Validate and stage local Wan VACE assets into a real temporary cache.

The source remains read-only. A complete cache is reused without reading the
source; missing files are copied atomically under an inter-process file lock.
No Torch import, model deserialization, network access or weight download.
"""

import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


def resolve_model_files(directory):
    directory = Path(directory).expanduser().resolve()
    shards = sorted(directory.glob("diffusion_pytorch_model*.safetensors"))
    index = directory / "diffusion_pytorch_model.safetensors.index.json"
    if index.is_file():
        shards = sorted({directory / name for name in json.loads(index.read_text())["weight_map"].values()})
    if not shards:
        raise ValueError(f"No diffusion_pytorch_model*.safetensors in {directory}")
    text_encoder = directory / "models_t5_umt5-xxl-enc-bf16.pth"
    vae = directory / "Wan2.1_VAE.pth"
    tokenizer = directory / "google/umt5-xxl"
    for path in [*shards, text_encoder, vae, tokenizer / "tokenizer_config.json"]:
        if not path.is_file():
            raise ValueError(f"Missing local model file: {path}")
    if not any((tokenizer / name).is_file() for name in ("tokenizer.json", "spiece.model")):
        raise ValueError(f"Missing tokenizer.json or spiece.model in {tokenizer}")
    return dict(diffusion=[str(path) for path in shards], text_encoder=str(text_encoder),
                vae=str(vae), tokenizer=str(tokenizer))


def _files(directory):
    directory = Path(directory)
    model = resolve_model_files(directory)
    if len(model["diffusion"]) != 7:
        raise ValueError("Wan2.1-VACE-14B requires seven diffusion shards")
    paths = [Path(path) for path in model["diffusion"]]
    paths += [Path(model["text_encoder"]), Path(model["vae"])]
    paths += [path for path in Path(model["tokenizer"]).rglob("*") if path.is_file()]
    index = directory / "diffusion_pytorch_model.safetensors.index.json"
    if index.is_file():
        paths.append(index)
    for path in paths:
        if path.stat().st_size <= 0:
            raise ValueError(f"Empty local VACE asset: {path}")
        try:
            relative = path.relative_to(directory)
            if ".." in relative.parts:
                raise ValueError("Parent traversal in model index")
        except ValueError as error:
            raise ValueError(f"Model index references a path outside its directory: {path}") from error
    return paths


def _is_local(path, cache):
    return not path.is_symlink() and path.resolve().is_relative_to(cache)


def complete_cache(cache):
    cache = Path(cache).expanduser().absolute()
    if cache.is_symlink():
        return False
    try:
        return all(_is_local(path, cache) for path in _files(cache))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def ensure_model_cache(source=None, cache=None, dry_run=False):
    """Prefer complete cache; otherwise copy from the selected local source.

    source may be a callable, so configured storage is not inspected when the
    cache is already complete. dry_run validates source and reports the future
    copy without staging large files; actual inference always stages first.
    """
    cache = Path(cache or os.environ.get("WAN_VACE_CACHE_DIR", "/tmp/vace-model-cache")).expanduser().absolute()
    if complete_cache(cache):
        return cache
    selected = source() if callable(source) else source
    if selected is None:
        raise ValueError("No complete VACE cache and no configured local weight source")
    selected = Path(selected).expanduser().resolve()
    paths = _files(selected)
    if selected == cache.resolve():
        raise ValueError(f"VACE cache is incomplete and cannot repair itself: {cache}")
    if dry_run:
        print(f"VACE cache needs staging: {selected} -> {cache} (copy on actual run)", file=sys.stderr)
        return selected
    if cache.is_symlink():
        raise ValueError(f"VACE cache must be a real directory, not a symlink: {cache}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    with (cache.parent / f".{cache.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if complete_cache(cache):
            return cache
        cache.mkdir(parents=True, exist_ok=True)
        targets = [(path, cache / path.relative_to(selected)) for path in paths]
        pending = [(src, dst) for src, dst in targets
                   if not (_is_local(dst, cache) and dst.is_file() and dst.stat().st_size == src.stat().st_size)]
        required = sum(src.stat().st_size for src, _ in pending)
        if shutil.disk_usage(cache).free < required:
            raise ValueError(f"Insufficient cache disk space; need {required} bytes in {cache}")
        for src, dst in pending:
            # Detach only links inside cache, never write through into source.
            relative = dst.relative_to(cache)
            parent = cache
            for name in relative.parts[:-1]:
                parent = parent / name
                if parent.is_symlink():
                    parent.unlink()
                parent.mkdir(exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{dst.name}.", dir=dst.parent)
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                print(f"Caching VACE asset: {relative}", file=sys.stderr, flush=True)
                shutil.copy2(src, temporary)
                if temporary.stat().st_size != src.stat().st_size:
                    raise ValueError(f"Incomplete cache copy: {relative}")
                temporary.replace(dst)
            finally:
                temporary.unlink(missing_ok=True)
        source_index = selected / "diffusion_pytorch_model.safetensors.index.json"
        cache_index = cache / source_index.name
        if not source_index.is_file():
            cache_index.unlink(missing_ok=True)
        if not complete_cache(cache):
            raise ValueError(f"VACE cache validation failed after staging: {cache}")
    return cache
