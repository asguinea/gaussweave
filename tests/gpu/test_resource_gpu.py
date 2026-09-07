from __future__ import annotations

import pytest

from gaussweave.accounting.resources import (
    gpu_memory_snapshot,
    load_profile,
    measure_operation,
    reset_gpu_peaks,
    time_operation,
)

pytestmark = pytest.mark.gpu


def test_gpu_snapshot_peak_reset_and_cuda_timing(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    baseline = gpu_memory_snapshot()
    assert baseline.device_name and baseline.total_vram_bytes > 0
    assert baseline.free_device_bytes is not None
    reset_gpu_peaks()
    tensor = torch.ones((1024, 1024), device="cuda:0")
    peak = gpu_memory_snapshot()
    assert peak.peak_allocated_bytes >= tensor.numel() * tensor.element_size()
    del tensor
    reset_gpu_peaks()
    after_reset = gpu_memory_snapshot()
    assert after_reset.peak_allocated_bytes == after_reset.allocated_bytes
    timing = time_operation(
        lambda: torch.ones((64, 64), device="cuda:0").sum(),
        warmup_count=2,
        repetition_count=4,
        cuda_device="cuda:0",
    )
    assert timing.synchronized and len(timing.samples_seconds) == 4
    profile = load_profile(
        "smoke",
        {"smoke": {"disk_safety_reserve_bytes": 0, "expected_workspace_bytes": 0}},
    )
    record = measure_operation(
        lambda: torch.ones((256, 256), device="cuda:0").square(),
        operation_name="gpu-test",
        profile=profile,
        workspace_root=tmp_path,
        device="cuda:0",
        warmup_count=1,
        repetition_count=2,
    )
    assert record.gpu_final and record.gpu_final.peak_allocated_bytes > 0
