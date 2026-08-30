#!/usr/bin/env python3
"""Apply the temporary hybrid-NVFP4 PLE resolver workaround."""

from pathlib import Path


ROOTS = (
    Path("/usr/local/lib"),
    Path("/usr/lib"),
)

NEEDLE = '''    """Select global-scale FP8 only for quantized PLE checkpoint shards."""

    if not isinstance(quant_config, Fp8Config):
'''

REPLACEMENT = '''    """Select global-scale FP8 only for quantized PLE checkpoint shards."""

    import os as _os
    if _os.environ.get("VLLM_PLE_FORCE_FP8") == "1":
        return Qwen3_8FlashNextPLEFp8EmbeddingMethod()

    if not isinstance(quant_config, Fp8Config):
'''

PATCH_MARKER = 'if _os.environ.get("VLLM_PLE_FORCE_FP8") == "1":'


def candidates() -> list[Path]:
    paths: list[Path] = []
    relative = Path("vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py")
    for root in ROOTS:
        paths.extend(root.glob(f"python*/dist-packages/{relative}"))
        paths.extend(root.glob(f"python*/site-packages/{relative}"))
    return sorted(set(paths))


def main() -> None:
    matches = candidates()
    if len(matches) != 1:
        raise SystemExit(f"expected one ple_layer.py, found {len(matches)}: {matches}")

    target = matches[0]
    source = target.read_text()
    if PATCH_MARKER in source:
        print(f"PLE workaround already present in {target}")
        return
    if source.count(NEEDLE) != 1:
        raise SystemExit("vLLM PLE resolver changed; refusing an unverified patch")

    target.write_text(source.replace(NEEDLE, REPLACEMENT, 1))
    compile(target.read_text(), str(target), "exec")
    print(f"applied PLE workaround to {target}")


if __name__ == "__main__":
    main()
