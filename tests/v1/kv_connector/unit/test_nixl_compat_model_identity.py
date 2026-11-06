# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    _model_identity,
)


@pytest.mark.parametrize(
    "model, identity",
    [
        # Offline: a repo id replaced by its snapshot path, in any cache directory.
        (
            "/root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct"
            "/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
            "meta-llama/Llama-3.1-8B-Instruct",
        ),
        (
            "/shared/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/"
            "0e9e39f249a16976918f6564b8830bc894c89659/",
            "meta-llama/Llama-3.1-8B-Instruct",
        ),
        ("/hub/models--Qwen--Qwen3-0.6B/snapshots/abc", "Qwen/Qwen3-0.6B"),
        # Online repo ids and other local paths are unchanged.
        ("meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"),
        ("/models/llama-3.1-8b", "/models/llama-3.1-8b"),
        ("/data/snapshots/abc", "/data/snapshots/abc"),
    ],
)
def test_model_identity(model, identity):
    assert _model_identity(model) == identity
