# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_worker as bw


def _worker():
    worker = object.__new__(bw.NixlBaseConnectorWorker)
    worker.use_mla = False
    worker.kv_cache_layout = "HND"
    worker.kv_transfer_config = SimpleNamespace(enable_permute_local_kv=False)
    return worker


def test_host_buffers_come_from_the_platform_when_it_allocates_them():
    caches = {
        "a": torch.empty(4, 2, 8, 16, dtype=torch.bfloat16),
        "b": torch.empty(4, 2, 8, 16, dtype=torch.bfloat16),
    }
    mine = torch.zeros(4, 2, 8, 16, dtype=torch.bfloat16)
    calls = []

    def alloc(shapes):
        calls.append(shapes)
        return {"a": mine}  # the platform may leave layers to regular memory

    worker = _worker()
    with patch.object(bw.current_platform, "alloc_kv_host_buffers", alloc):
        worker.initialize_host_xfer_buffer(caches)

    assert calls == [{n: ((4, 2, 8, 16), torch.bfloat16) for n in caches}]
    assert worker.host_xfer_buffers["a"] is mine
    assert worker.host_xfer_buffers["b"].shape == (4, 2, 8, 16)


def test_host_buffers_are_regular_memory_by_default():
    caches = {"a": torch.empty(4, 2, 8, 16, dtype=torch.bfloat16)}
    worker = _worker()
    with patch.object(
        bw.current_platform, "alloc_kv_host_buffers", lambda shapes: None
    ):
        worker.initialize_host_xfer_buffer(caches)
    assert worker.host_xfer_buffers["a"].shape == (4, 2, 8, 16)
    assert worker.host_xfer_buffers["a"].dtype == torch.bfloat16
