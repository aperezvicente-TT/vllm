# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MooncakeConnector host-buffer mode (lab backport of the upstream PR)."""

import asyncio
import concurrent.futures
import contextlib
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.config import set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.utils import allocate_host_mirrors
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import mooncake_connector
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    KVConnectorRole,
    MooncakeConnector,
    MooncakeConnectorMetadata,
    MooncakeXferResponse,
    MooncakeXferResponseStatus,
    PullReqMeta,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec

from .test_mooncake_connector import patch_worker_dependencies
from .utils import create_vllm_config


def test_allocate_host_mirrors_keeps_layout_and_storage_sharing():
    raw = torch.zeros(4, 2, 8, dtype=torch.float16)
    caches = {
        "k": raw[:, 0],
        "v": raw[:, 1],
        "meta0": torch.empty(4, 8, dtype=torch.bfloat16, device="meta"),
        "meta1": torch.empty(4, 8, dtype=torch.bfloat16, device="meta"),
    }
    mirrors = allocate_host_mirrors(caches)
    for name, cache in caches.items():
        m = mirrors[name]
        assert m.device.type == "cpu"
        assert (m.shape, m.stride(), m.dtype, m.storage_offset()) == (
            cache.shape, cache.stride(), cache.dtype, cache.storage_offset())

    def storage(n):
        return mirrors[n].untyped_storage().data_ptr()

    assert storage("k") == storage("v")
    assert len({storage("k"), storage("meta0"), storage("meta1")}) == 3


@contextlib.contextmanager
def _host_buffer_worker(monkeypatch, kv_role):
    def no_accelerator():
        raise AssertionError("host-buffer mode must not bind an accelerator")

    monkeypatch.setattr(mooncake_connector.current_platform, "device_type", "fake")
    monkeypatch.setattr(torch.accelerator, "current_device_index", no_accelerator)
    spec = FullAttentionSpec(block_size=16, num_kv_heads=4, head_size=64,
                             dtype=torch.float16)
    names = ["model.layers.0.self_attn", "model.layers.1.self_attn"]
    cfg = KVCacheConfig(num_blocks=8, kv_cache_tensors=[],
                        kv_cache_groups=[KVCacheGroupSpec(names, spec)])
    vllm_config = create_vllm_config(kv_connector="MooncakeConnector", kv_role=kv_role,
                                     block_size=16)
    vllm_config.kv_transfer_config.kv_buffer_device = "cpu"
    mod = "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector"
    with (set_current_vllm_config(vllm_config), patch_worker_dependencies(),
          patch(f"{mod}.threading.Event"), patch(f"{mod}.threading.Thread") as thr):
        worker = MooncakeConnector(vllm_config, KVConnectorRole.WORKER, cfg).connector_worker
        thr.return_value.is_alive.return_value = False
        with patch.object(worker.engine, "batch_register_memory", return_value=0) as reg:
            events = []
            worker.set_host_xfer_buffer_ops(
                lambda src, dst, s, d, direction: events.append((direction, src, dst, list(s))))
            caches = {n: torch.empty(8, 16, 4, 128, dtype=torch.float16, device="meta")
                      for n in names}
            worker.register_kv_caches(caches)
            yield worker, reg, caches, events
        worker.shutdown()
        worker.shutdown = MagicMock()


def _run_now(coro, loop):
    fut = concurrent.futures.Future()
    fut.set_result(asyncio.run(coro))
    return fut


def test_registers_host_mirrors(monkeypatch):
    with _host_buffer_worker(monkeypatch, "kv_consumer") as (worker, reg, caches, _):
        mirrors = worker.host_xfer_buffers
        ptrs, lens = reg.call_args[0]
        assert ptrs == [m.untyped_storage().data_ptr() for m in mirrors.values()]
        assert lens == [m.untyped_storage().nbytes() for m in mirrors.values()]
        assert worker.device_kv_caches is caches


def test_producer_copies_to_host_before_send(monkeypatch):
    with _host_buffer_worker(monkeypatch, "kv_producer") as (worker, _, caches, events):
        monkeypatch.setattr(mooncake_connector.asyncio, "run_coroutine_threadsafe",
                            MagicMock(side_effect=lambda coro, loop: coro.close()))
        monkeypatch.setattr(worker, "record_send_reqs",
                            lambda m: events.append(("ready",)) or asyncio.sleep(0))
        meta = MooncakeConnectorMetadata()
        meta.add_new_req("p-req", [[0, 3, 5]], {"transfer_id": "x"},
                         load_remote_cache=False)
        worker.start_load_kv(meta)
        (direction, src, dst, ids), ready = events
        assert (direction, ids, ready) == ("d2h", [3, 5], ("ready",))
        assert all(src[n] is caches[n] for n in caches)


@pytest.mark.parametrize("ok", [True, False])
def test_consumer_copies_to_device_before_reporting(monkeypatch, ok):
    with _host_buffer_worker(monkeypatch, "kv_consumer") as (worker, _, caches, events):
        monkeypatch.setattr(mooncake_connector.asyncio, "run_coroutine_threadsafe", _run_now)
        pm = PullReqMeta(d_req_id="d", transfer_id="x", local_block_ids=[[2, 6]],
                         remote_engine_id="p", remote_bootstrap_addr="http://b:1",
                         pull_tasks_count=1)
        resp = (MooncakeXferResponse(status=MooncakeXferResponseStatus.FINISH, ok_reqs=["d"])
                if ok else MooncakeXferResponse(status=MooncakeXferResponseStatus.ERROR,
                                                err_reqs=["d"], err_msg="x"))
        worker.process_pulling_result(resp, {"d": pm})
        assert events == []
        _, recv = worker.get_finished()
        if ok:
            assert recv == {"d"}
            ((direction, src, dst, ids),) = events
            assert (direction, ids) == ("h2d", [2, 6])
            assert all(dst[n] is caches[n] for n in caches)
        else:
            assert events == []
