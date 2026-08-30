#!/usr/bin/env python3
"""Normalize OrcaRouter's QSA layer label for the day-zero vLLM image."""

from pathlib import Path


ROOTS = (
    Path("/usr/local/lib"),
    Path("/usr/lib"),
)

NEEDLE = '''        rope_scaling = kwargs.get("rope_scaling")
        rope_theta = kwargs.get("rope_theta", 10_000.0)
        super().__init__(layer_types=layer_types, **kwargs)
'''

REPLACEMENT = '''        rope_scaling = kwargs.get("rope_scaling")
        rope_theta = kwargs.get("rope_theta", 10_000.0)

        # OrcaRouter uses the newer Transformers label for Qwen Sparse
        # Attention. This vLLM image calls the same QSA path full_attention.
        if layer_types is not None:
            layer_types = [
                "full_attention"
                if layer_type == "qwen_sparse_attention"
                else layer_type
                for layer_type in layer_types
            ]

        super().__init__(layer_types=layer_types, **kwargs)
'''

PATCH_MARKER = 'if layer_type == "qwen_sparse_attention"'


def candidates() -> list[Path]:
    paths: list[Path] = []
    relative = Path("vllm/models/qwen3_8_flash_next/config.py")
    for root in ROOTS:
        paths.extend(root.glob(f"python*/dist-packages/{relative}"))
        paths.extend(root.glob(f"python*/site-packages/{relative}"))
    return sorted(set(paths))


def patch(target: Path) -> None:
    source = target.read_text()
    if PATCH_MARKER in source:
        print(f"QSA layer normalization already present in {target}")
        return
    if source.count(NEEDLE) != 1:
        raise SystemExit("vLLM Qwen config changed; refusing an unverified patch")

    target.write_text(source.replace(NEEDLE, REPLACEMENT, 1))
    compile(target.read_text(), str(target), "exec")
    print(f"applied QSA layer normalization to {target}")


def main() -> None:
    matches = candidates()
    if len(matches) != 1:
        raise SystemExit(f"expected one Qwen config.py, found {len(matches)}: {matches}")
    patch(matches[0])


if __name__ == "__main__":
    main()
