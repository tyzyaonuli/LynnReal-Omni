"""Compile, verify, and time the release INT8 GEMM on one CUDA GPU."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import warnings


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def elapsed_ms(torch, function, warmups=3, repeats=10):
    for _ in range(warmups):
        function()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return {
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "measurements_ms": values,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import triton
    from model import int8_gemm

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("validation requires exactly one visible CUDA GPU")

    device = torch.cuda.get_device_properties(0)
    rng = torch.Generator(device="cuda").manual_seed(219)
    report = {
        "status": "running",
        "gpu": device.name,
        "capability": [device.major, device.minor],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "integer_accumulator": "int32",
        "output": "bfloat16",
        "checks": [],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)

    # These are the four projection families used by the release transformer.
    # A non-tile-aligned row count exercises masks and more than one M tile while
    # keeping this preflight small compared with the full 10-second benchmark.
    shapes = [
        (257, 21504, 5376),
        (257, 5376, 7168),
        (257, 28672, 5376),
        (257, 5376, 14336),
    ]
    tiles = [tile for tile in int8_gemm._TILES if tile[-1] == 0]
    if device.major >= 10 and len(tiles) != 4:
        raise AssertionError(f"expected four Blackwell tiles, got {tiles}")

    started = time.perf_counter()
    for shape_index, (m, n, k) in enumerate(shapes):
        a = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8, generator=rng)
        # Packed model weights expose [K,N] with stride (1,K).
        b = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8, generator=rng).t()
        row_scale = torch.rand((m, 1), device="cuda", generator=rng) / 127
        column_scale = torch.rand((1, n), device="cuda", generator=rng) / 127
        bias = (torch.randn(n, device="cuda", dtype=torch.bfloat16, generator=rng)
                if shape_index % 2 else None)
        reference = int8_gemm._torch_matmul(a, b, row_scale, column_scale, bias)

        tile_checks = []
        for tile in tiles:
            actual = int8_gemm.int8_matmul(
                a, b, row_scale, column_scale, bias, tile=tile
            )
            exact = torch.equal(actual, reference)
            tile_checks.append({"tile": list(tile), "exact": exact})
            if not exact:
                difference = (actual.float() - reference.float()).abs()
                raise AssertionError(
                    f"tile {tile} differs for {(m, n, k)}: max_abs={difference.max().item()}"
                )

        disabled_before = set(int8_gemm._disabled)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            selected_output = int8_gemm.int8_matmul(a, b, row_scale, column_scale, bias)
        if int8_gemm._disabled != disabled_before:
            raise AssertionError(f"autotuned Triton path fell back for {(m, n, k)}")
        if not torch.equal(selected_output, reference):
            raise AssertionError(f"autotuned result differs for {(m, n, k)}")

        selected = [
            value for key, value in int8_gemm._selected.items()
            if key[-5:] == (m, n, k, b.stride(), bias is not None)
        ]
        if len(selected) != 1:
            raise AssertionError(f"missing unique selected tile for {(m, n, k)}: {selected}")

        deterministic = True
        for _ in range(5):
            deterministic &= torch.equal(
                int8_gemm.int8_matmul(a, b, row_scale, column_scale, bias),
                selected_output,
            )
        if not deterministic:
            raise AssertionError(f"nondeterministic result for {(m, n, k)}")

        triton_timing = elapsed_ms(
            torch, lambda: int8_gemm.int8_matmul(a, b, row_scale, column_scale, bias)
        )
        torch_timing = elapsed_ms(
            torch, lambda: int8_gemm._torch_matmul(a, b, row_scale, column_scale, bias)
        )
        check = {
            "shape_mnk": [m, n, k],
            "weight_stride": list(b.stride()),
            "bias": bias is not None,
            "all_explicit_tiles_exact": True,
            "autotuned_exact": True,
            "deterministic_repeats": 5,
            "selected_tile": list(selected[0]),
            "triton": triton_timing,
            "torch_int_mm": torch_timing,
            "triton_speedup_over_torch": (
                torch_timing["median_ms"] / triton_timing["median_ms"]
            ),
            "tile_checks": tile_checks,
        }
        report["checks"].append(check)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(check), flush=True)
        del a, b, row_scale, column_scale, bias, reference, selected_output, actual
        torch.cuda.empty_cache()

    report["status"] = "passed"
    report["elapsed_seconds"] = time.perf_counter() - started
    report["fallback_shapes"] = len(int8_gemm._disabled)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
