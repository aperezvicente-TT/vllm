# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared constants, lazy imports and helpers for the NIXL connector."""

import contextlib
import math
import mmap
from collections.abc import Iterator
from typing import Any

import regex as re
import torch
import zmq

from vllm.platforms import current_platform
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.kv_cache_interface import KVCacheSpec, UniformTypeKVCacheSpecs

# Supported platforms and types of kv transfer buffer.
# {device: tuple of supported kv buffer types}
_NIXL_SUPPORTED_DEVICE = {
    "cuda": (
        "cuda",
        "cpu",
    ),
    "tpu": ("cpu",),
    "xpu": (
        "cpu",
        "xpu",
    ),
    "cpu": ("cpu",),
}
# support for oot platform by providing mapping in current_platform
_NIXL_SUPPORTED_DEVICE.update(current_platform.get_nixl_supported_devices())


# TODO: merge with vllm.utils.network_utils.zmq_socket_ctx
@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:
    """Context manager for a ZMQ socket."""
    if socket_type not in (zmq.ROUTER, zmq.REQ):
        raise ValueError(f"Unexpected socket type: {socket_type}")

    ctx: zmq.Context | None = None
    try:
        ctx = zmq.Context()  # type: ignore[attr-defined]
        yield make_zmq_socket(
            ctx=ctx, path=addr, socket_type=socket_type, bind=socket_type == zmq.ROUTER
        )
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)


def get_representative_spec_type(spec: KVCacheSpec) -> type[KVCacheSpec]:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        # All inner specs are the same type; pick any.
        inner = next(iter(spec.kv_cache_specs.values()))
        return type(inner)
    return type(spec)


# Trailing 8-hex randomization suffix appended by
# ``input_processor.assign_request_id`` as ``-{random_uuid():.8}``.
_RANDOM_SUFFIX_RE = re.compile(r"-[0-9a-f]{8}$", re.IGNORECASE)


def get_base_request_id(request_id: str) -> str:
    """Strip the per-request `-<8 hex>` randomization suffix, if present."""
    return _RANDOM_SUFFIX_RE.sub("", request_id)


_HUGEPAGE_SIZES = {"2M": 2 << 20, "1G": 1 << 30}
_MAP_HUGETLB = 0x40000  # Linux; not exported by the mmap module before Python 3.13
_MAP_HUGE_SHIFT = 26


def alloc_hugepage_host_buffers(
    shapes: dict[str, tuple[tuple[int, ...], torch.dtype]], page_size: str
) -> tuple[mmap.mmap, dict[str, torch.Tensor]]:
    """Allocate host transfer buffers from one anonymous hugetlb mapping.

    Buffers are laid out in order. One whose rows (first-dimension entries, the KV
    blocks) evenly divide the huge page follows the previous one, aligned so that no
    row spans two pages; any other buffer that fits in a huge page never spans two.
    Either way each row is physically contiguous: devices that DMA into host memory
    without an IOMMU translating for them (passthrough) can only map physically
    contiguous memory. A NIC registers the mapping with one entry per huge page.

    Returns the mapping, which must outlive the tensors, and the tensors by name.
    Raises OSError when the hugetlb pool cannot back the mapping.
    """
    if page_size not in _HUGEPAGE_SIZES:
        raise ValueError(
            f"host_buffer_hugepage_size must be one of {list(_HUGEPAGE_SIZES)}, "
            f"got {page_size!r}"
        )
    page = _HUGEPAGE_SIZES[page_size]
    offsets: dict[str, tuple[int, int]] = {}
    end = 0
    for name, (shape, dtype) in shapes.items():
        nbytes = math.prod(shape) * dtype.itemsize
        row = nbytes // shape[0] if shape and shape[0] else nbytes
        if 0 < row <= page and page % row == 0:
            end = -(-end // max(row, 4096)) * max(row, 4096)
        else:
            end = -(-end // 4096) * 4096
            if nbytes <= page and end // page != (end + nbytes - 1) // page:
                end = -(-end // page) * page
        offsets[name] = (end, nbytes)
        end += nbytes
    total = max(page, -(-end // page) * page)
    flags = mmap.MAP_PRIVATE | _MAP_HUGETLB | (page.bit_length() - 1) << _MAP_HUGE_SHIFT
    try:
        buf = mmap.mmap(-1, total, flags=flags, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    except OSError as e:
        raise OSError(
            e.errno, f"{e.strerror}: {total // page} pages of {page_size} requested"
        ) from e
    tensors = {
        name: torch.frombuffer(buf, dtype=torch.uint8, count=nbytes, offset=off)
        .view(shapes[name][1])
        .view(shapes[name][0])
        for name, (off, nbytes) in offsets.items()
    }
    return buf, tensors
