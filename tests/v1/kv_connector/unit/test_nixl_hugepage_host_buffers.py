# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import mmap

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.nixl import utils


@pytest.fixture
def regular_mmap(monkeypatch):
    """Back the mapping with regular memory, recording the flags asked for."""
    calls = []
    real = mmap.mmap

    def fake(fileno, length, flags, prot):
        calls.append((length, flags))
        return real(fileno, length, flags=mmap.MAP_PRIVATE, prot=prot)

    monkeypatch.setattr(utils.mmap, "mmap", fake)
    return calls


def _page_spans(buf, t, page):
    base = torch.frombuffer(buf, dtype=torch.uint8).data_ptr()
    start = t.data_ptr() - base
    return start, start + t.numel() * t.element_size() - 1


def test_rows_dividing_the_page_are_packed_without_spanning_pages(regular_mmap):
    page = 2 << 20
    # 768 KiB buffers of three 256 KiB rows: packed back to back, so five fill two
    # pages, and a buffer may span pages but none of its rows does.
    shapes = {f"l{i}": ((3, 128, 1024), torch.bfloat16) for i in range(5)}
    buf, tensors = utils.alloc_hugepage_host_buffers(shapes, "2M")
    row = 256 << 10
    for t in tensors.values():
        start, _ = _page_spans(buf, t, page)
        assert start % row == 0
        for r in range(3):
            a = start + r * row
            assert a // page == (a + row - 1) // page
    assert regular_mmap == [
        (2 * page, mmap.MAP_PRIVATE | utils._MAP_HUGETLB | 21 << utils._MAP_HUGE_SHIFT)
    ]


def test_buffers_with_other_rows_that_fit_a_page_never_span_two(regular_mmap):
    page = 2 << 20
    shapes = {f"l{i}": ((3, 250_000), torch.uint8) for i in range(5)}  # 750 kB
    buf, tensors = utils.alloc_hugepage_host_buffers(shapes, "2M")
    for t in tensors.values():
        start, end = _page_spans(buf, t, page)
        assert start // page == end // page
        assert start % 4096 == 0
    assert regular_mmap[0][0] == 3 * page  # two per page


def test_buffers_have_requested_shape_and_dtype_and_are_disjoint(regular_mmap):
    shapes = {
        "a": ((4, 8, 16, 64), torch.bfloat16),
        "b": ((2, 3), torch.float32),
        "c": ((5000,), torch.uint8),
    }
    buf, tensors = utils.alloc_hugepage_host_buffers(shapes, "1G")
    for name, (shape, dtype) in shapes.items():
        assert tensors[name].shape == shape
        assert tensors[name].dtype == dtype
        assert tensors[name].is_contiguous()
    for i, t in enumerate(tensors.values()):
        t.view(torch.uint8).fill_(i + 1)
    for i, t in enumerate(tensors.values()):
        assert bool((t.view(torch.uint8) == i + 1).all())


def test_buffer_larger_than_a_page_is_placed(regular_mmap):
    page = 2 << 20
    shapes = {"big": ((3 * page,), torch.uint8), "small": ((page,), torch.uint8)}
    _, tensors = utils.alloc_hugepage_host_buffers(shapes, "2M")
    assert tensors["big"].numel() == 3 * page
    start = tensors["small"].data_ptr() - tensors["big"].data_ptr()
    assert start == 3 * page


def test_unknown_page_size_is_rejected():
    with pytest.raises(ValueError, match="host_buffer_hugepage_size"):
        utils.alloc_hugepage_host_buffers({"a": ((1,), torch.uint8)}, "4K")
