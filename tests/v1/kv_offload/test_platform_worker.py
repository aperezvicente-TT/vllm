# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU offloading on a platform that supplies its own offloading worker and
whose KV caches have no host-addressable storage."""

import uuid
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import vllm.v1.kv_offload.cpu.spec as cpu_spec_module
from tests.v1.kv_offload.conftest import make_offloading_config
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    OffloadingSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.kv_offload.pointer_utils import resolve_device_pointers
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec


class FakePlatformWorker(OffloadingWorker):
    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        blocks_per_chunk: int,
        num_cpu_chunks: int,
        mmap_region: SharedOffloadRegion | None = None,
        canonical_layout: bool = False,
    ):
        self.kv_caches = kv_caches
        self.blocks_per_chunk = blocks_per_chunk
        self.num_cpu_chunks = num_cpu_chunks
        self.mmap_region = mmap_region
        self.canonical_layout = canonical_layout

    def submit_store(self, job_id, device_ptrs, dst_spec) -> bool:
        return True

    def submit_load(self, job_id, src_spec, device_ptrs, expedited=False) -> bool:
        return True

    def get_finished(self) -> list[TransferResult]:
        return []

    def wait(self, job_ids: set[int]) -> None:
        return


FAKE_WORKER = f"{__name__}.{FakePlatformWorker.__qualname__}"


def _fake_platform(monkeypatch, worker_cls: str | None) -> None:
    """A platform that is neither CUDA-alike nor XPU."""
    platform = cpu_spec_module.current_platform
    monkeypatch.setattr(platform, "is_cuda_alike", lambda: False)
    monkeypatch.setattr(platform, "is_rocm", lambda: False)
    monkeypatch.setattr(platform, "is_xpu", lambda: False)
    monkeypatch.setattr(platform, "get_offloading_worker_cls", lambda: worker_cls)


def _create_spec(**kwargs: Any) -> OffloadingSpec:
    return OffloadingSpecFactory.create_spec(make_offloading_config(**kwargs))


def test_cpu_spec_uses_platform_worker(monkeypatch):
    _fake_platform(monkeypatch, FAKE_WORKER)
    spec = _create_spec(
        worker_kv_bytes_per_block=SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT,
        blocks_per_chunk=2,
    )
    assert isinstance(spec, CPUOffloadingSpec)

    kv_caches = MagicMock()
    worker = spec.get_worker(kv_caches)

    assert isinstance(worker, FakePlatformWorker)
    assert worker.kv_caches is kv_caches
    assert worker.blocks_per_chunk == 2
    assert worker.num_cpu_chunks == spec.num_chunks
    # Off the shared-region platforms the CPU spec keeps a private buffer.
    assert worker.mmap_region is None
    assert spec.get_worker(kv_caches) is worker


def test_cpu_spec_refuses_platform_without_worker(monkeypatch):
    _fake_platform(monkeypatch, None)
    spec = _create_spec()
    with pytest.raises(Exception, match="get_offloading_worker_cls"):
        spec.get_worker(MagicMock())


def test_tiering_spec_gives_platform_worker_the_shared_region(monkeypatch):
    """Secondary tiers read the CPU tier through the scheduler's mapping of the
    shared region, so the platform worker gets the worker side of it."""
    _fake_platform(monkeypatch, FAKE_WORKER)
    page = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    spec = _create_spec(
        spec_name="TieringOffloadingSpec",
        engine_id=f"test-{uuid.uuid4().hex}",
        cpu_bytes_to_use=page * 4,
        worker_kv_bytes_per_block=page,
    )
    assert isinstance(spec, TieringOffloadingSpec)
    manager = spec.get_manager()
    worker = spec.get_worker(MagicMock())
    assert isinstance(worker, FakePlatformWorker)
    assert isinstance(worker.mmap_region, SharedOffloadRegion)
    assert spec._scheduler_mmap is not None
    try:
        # What the platform worker stores in its slot is what the secondary
        # tiers see.
        view = worker.mmap_region.create_next_worker_view(page)
        view[1].fill_(7)
        kv_view = spec._scheduler_mmap.create_kv_memoryview()
        flat = kv_view.cast("B")
        chunk = spec.kv_bytes_per_chunk
        assert bytes(flat[chunk : chunk + page]) == b"\x07" * page
        assert bytes(flat[:page]) == b"\x00" * page
        flat.release()
        kv_view.release()
        del view
    finally:
        worker.mmap_region.cleanup()
        manager.shutdown()


def test_register_storage_free_kv_caches(monkeypatch):
    """Caches without storage (all at address 0) are separate layers, not
    aliases, and each reference names its layer; block ids are recoverable
    from the resolved device pointers."""
    num_blocks, block_size, num_heads, head_size = 10, 16, 4, 64
    attn = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_heads,
        head_size=head_size,
        dtype=torch.float16,
    )
    names = [f"model.layers.{i}.self_attn.attn" for i in range(3)]
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=num_blocks * attn.page_size_bytes,
                layers=[name],
                layer_stride=0,
                block_stride=attn.page_size_bytes,
            )
            for name in names
        ],
        kv_cache_groups=[KVCacheGroupSpec(names, attn)],
    )
    kv_caches = {
        name: torch.empty(
            num_blocks,
            block_size,
            num_heads,
            2 * head_size,
            dtype=torch.float16,
            device="meta",
        )
        for name in names
    }

    vllm_config = MagicMock()
    parallel_config = vllm_config.parallel_config
    parallel_config.tensor_parallel_size = 1
    parallel_config.decode_context_parallel_size = 1
    parallel_config.prefill_context_parallel_size = 1
    parallel_config.cp_kv_cache_interleave_size = 1
    parallel_config.world_size = 1
    parallel_config.rank = 0
    vllm_config.model_config.get_total_num_kv_heads.return_value = num_heads
    spec = MagicMock(spec=OffloadingSpec)
    spec.replicated_layout = False
    spec.config = MagicMock()
    spec.config.canonical_layout = False
    spec.config.parallel.rank = 0

    worker = OffloadingConnectorWorker(
        spec=spec, vllm_config=vllm_config, kv_cache_config=kv_cache_config
    )
    worker.register_kv_caches(kv_caches)
    canonical = spec.get_worker.call_args[0][0]

    assert len(canonical.tensors) == len(names)
    (refs,) = canonical.group_data_refs
    assert [ref.layer_name for ref in refs] == names
    assert sorted(ref.tensor_idx for ref in refs) == list(range(len(names)))

    block_ids = np.array([3, 7], dtype=np.int64)
    ptrs = resolve_device_pointers(
        GPULoadStoreSpec(block_ids, group_sizes=(2,), block_indices=(0,)),
        canonical,
    )
    for i, ref in enumerate(refs):
        tensor = canonical.tensors[ref.tensor_idx].tensor
        layer_ptrs = ptrs.ptrs[2 * i : 2 * i + 2]
        decoded = (layer_ptrs - tensor.data_ptr()) // tensor.stride(0)
        assert decoded.tolist() == block_ids.tolist()
