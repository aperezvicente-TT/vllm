# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Push-specific (WRITE) worker-side logic for the NIXL connector.

A dedicated `nixl-push-writer` thread owns all push-related NIXL ops:
calls `get_new_notifs` (routing PUSH_REG internally; HB / completion
notifs are forwarded to the engine main thread), sends PUSH_REG via
`send_notif`, matches D registrations with P finished blocks, and
issues WRITE transfers via `make_prepped_xfer` / `transfer`.

The engine main thread feeds the writer through queues:
`_reg_send_inbox` (D-side regs to send), `_finished_blocks_inbox`
(P-side blocks from metadata) and `_pending_completion_notifs`
(non-PUSH_REG notifs forwarded back for HB / completion accounting).
The handshake-completion callback feeds `_deferred_push_inbox` with
matched pushes whose P→D handshake has finished so the writer can
(re-)issue the WRITE without ever blocking on the network.

Wake model: the writer self-polls every
`_PUSH_WRITER_POLL_INTERVAL_MS` only while it has unmatched
`_push_finished_blocks` (i.e. P-side blocks waiting for a D PUSH_REG
notif that has no other wake source). All other progress is
event-driven: the engine main thread sets `_push_writer_wake` from
`start_load_kv` (when handing it new work) and from `get_finished`
(so each engine step gives the writer a chance to drain NIXL notifs);
the handshake-completion callback sets the same event after a deferred
PUSH_REG send or a deferred push WRITE has been queued. When a request's
lease expires (the base worker reports it via `done_sending`) or the WRITE completes,
`get_finished` enqueues an eviction onto `_evict_finished_inbox` so
the writer drops any leftover `_push_finished_blocks` /
`_pending_d_registrations` and stops self-polling.

Early push (host-buffer saves only). While a request still prefills, the
scheduler reports after each chunk how many of its leading blocks hold final
KV. The main thread keeps the request's saved blocks in position order and,
once an asynchronous copy op reports a chunk's save done (a synchronous op's
are done on return), hands the final blocks to the writer
(`_early_blocks_inbox`). Once D's PUSH_REG has arrived, the writer WRITEs
them to D's blocks with the same suffix alignment as the final push (P block
`i` -> D block `i - (N - len(D blocks))`), never the prompt's last block.
When the request finishes, the final WRITE (which carries the completion
notif) covers only the blocks not yet written, and is posted only after every
early WRITE is done, so D never sees completion before all its blocks landed.
Early WRITEs are tracked apart from `_sending_transfers`; one that fails
makes the final WRITE cover every block. A D that set `accepts_parts` in its
PUSH_REG gets a PUSH_PART notif with each early WRITE and copies those blocks
to the device as they arrive. If P aborts the request, the writer waits for
its early WRITEs and sends PUSH_ABORT so D fails the receive.
"""

import dataclasses
import queue
import threading
import time
from collections import defaultdict, deque
from collections.abc import Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import msgspec

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorTransferResults,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    PUSH_ABORT_NOTIF_PREFIX,
    PUSH_PART_NOTIF_PREFIX,
    PUSH_REG_NOTIF_PREFIX,
    NixlAgentMetadata,
    NixlConnectorMetadata,
    RemoteMeta,
    ReqId,
    ReqMeta,
    TransferHandle,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    ReadSpec,
    _is_attention_spec,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import get_base_request_id
from vllm.logger import init_logger
from vllm.utils.gpu_sync_debug import gpu_sync_allowed

if TYPE_CHECKING:
    import torch

    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

# Writer-thread poll cadence while there is in-flight push state. When
# fully idle, the writer blocks on a wake event signalled by the engine
# main thread (start_load_kv / get_finished). Smaller -> lower latency
# while active, slightly more CPU.
_PUSH_WRITER_POLL_INTERVAL_MS = 1.0


@dataclass
class _EarlySaves:
    """Main thread: a prefilling request's host-buffer saves (early push)."""

    total: int
    # P logical block ids, in position order.
    blocks: list[int] = field(default_factory=list)
    # (end position, copy-op handles) per save not yet known to be done.
    pending: deque[tuple[int, list[Any]]] = field(default_factory=deque)
    done: int = 0
    final: int = 0
    handed: int = 0


@dataclass
class _EarlyPush:
    """Writer thread: a request's early WRITEs to D."""

    total: int
    # Saved final blocks not yet written, by position.
    ready: dict[int, int] = field(default_factory=dict)
    # Blocks written (or being written), by position.
    written: dict[int, int] = field(default_factory=dict)
    inflight: list[TransferHandle] = field(default_factory=list)
    reg: dict[str, Any] | None = None
    # Early WRITEs stopped; the final WRITE covers every block.
    failed: bool = False
    # The request finished while early WRITEs were in flight.
    finish: tuple[BlockIds, dict[str, Any]] | None = None
    # Aborted on P (PUSH_ABORT once the early WRITEs are done) or evicted.
    abort: bool = False
    evicted: bool = False
    # A handshake with D is in progress.
    handshaking: bool = False


class NixlPushConnectorWorker(NixlBaseConnectorWorker):
    """Push-specific (WRITE) worker logic. See module docstring."""

    # Distinguishes push from pull in the NIXL compatibility hash.
    _TRANSFER_MODE: str = "push"

    _supports_pp_hma = True

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)

        # Heartbeat handshakes to a PP-sharded producer must be notif-only,
        # like the PUSH_REG path.
        self._hb_handshake_notif_only = True

        # Push-specific state.
        # P-side: outgoing WRITE handles awaiting completion, keyed by
        # request_id. Mutated by writer (submit) and main thread
        # (``_pop_done_transfers``); guarded by
        # ``_sending_transfers_lock``.
        self._sending_transfers = defaultdict[ReqId, list[TransferHandle]](list)
        self._send_failures: set[ReqId] = set()
        self._sending_transfers_lock = threading.Lock()

        # Writer-thread owned matching state.
        # P-side: finished request blocks received from scheduler metadata
        # that have not yet been matched with an incoming D registration.
        self._push_finished_blocks: dict[ReqId, BlockIds] = {}
        # P-side: D registrations received via NIXL notification that have
        # not yet been matched with a finished P request.
        self._pending_d_registrations: dict[ReqId, dict[str, Any]] = {}

        # Cross-thread channels.
        self._reg_send_inbox: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
        self._finished_blocks_inbox: queue.Queue[tuple[str, BlockIds]] = queue.Queue()
        self._pending_completion_notifs: queue.Queue[bytes] = queue.Queue()
        # Main thread → writer: req_ids whose lease has expired or whose
        # WRITE has completed. Writer drops them from
        # ``_push_finished_blocks`` so an unmatched entry doesn't keep the
        # writer busy-polling forever.
        self._evict_finished_inbox: queue.Queue[str] = queue.Queue()
        # Handshakes that have just completed and are ready for the WRITE on wthread
        self._deferred_push_inbox = queue.Queue[tuple[str, BlockIds, dict[str, Any]]]()

        # Early push (see module docstring).
        # Main thread: saves of prefilling requests, by P request id.
        self._early_saves: dict[ReqId, _EarlySaves] = {}
        # Main thread: prefilling requests whose saves give no completion to
        # rely on (a failed save, or a copy op that returned no handle and is
        # not declared synchronous); their blocks all go with the final WRITE.
        self._early_off: set[ReqId] = set()
        # Main thread -> writer: (req_id, prompt blocks, [(position, block)]).
        self._early_blocks_inbox: queue.Queue[
            tuple[str, int, list[tuple[int, int]]]
        ] = queue.Queue()
        # Main thread -> writer: P requests aborted (or preempted) mid-prefill.
        self._early_abort_inbox: queue.Queue[str] = queue.Queue()
        # Writer: early WRITE state by P request id.
        self._early_pushes: dict[ReqId, _EarlyPush] = {}
        # Writer: aborted P requests, with the time until which a PUSH_REG
        # for them (late, or one per D rank) is answered with PUSH_ABORT.
        self._aborted_reqs: dict[ReqId, float] = {}
        assert vllm_config.kv_transfer_config is not None
        self._aborted_req_ttl = float(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "kv_lease_duration", 30
            )
        )
        # D side: physical blocks already copied to the device from PUSH_PARTs.
        self._push_parts_loaded: dict[ReqId, set[int]] = {}

        # Wake signal from engine main thread (start_load_kv / get_finished).
        # Writer self-polls at _PUSH_WRITER_POLL_INTERVAL_MS while it has
        # active in-flight state; otherwise it blocks until signalled.
        self._push_writer_wake = threading.Event()

        self._push_writer_stop = threading.Event()
        self._push_writer_thread: threading.Thread | None = None

    def _validate_remote_parallel_config(
        self, agent_metadata: NixlAgentMetadata
    ) -> None:
        super()._validate_remote_parallel_config(agent_metadata)
        local = (self.block_size, self._physical_blocks_per_logical_kv_block)
        remote = (
            agent_metadata.block_size,
            agent_metadata.physical_blocks_per_logical_kv_block,
        )
        if local != remote:
            raise NotImplementedError(
                "NixlPushConnector requires identical P/D block sizes. Kernel "
                "block size and kernel blocks per logical block: local "
                f"{local}, remote engine {agent_metadata.engine_id} {remote}."
            )

    # --- Lifecycle ----------------------------------------------------- #

    def register_kv_caches(self, kv_caches: dict[str, "torch.Tensor"]):
        super().register_kv_caches(kv_caches)
        if self._mixed_mem_types:
            raise NotImplementedError(
                "NixlPushConnector does not support mixed-memory KV caches"
            )
        if self._push_writer_thread is None:
            self._push_writer_thread = threading.Thread(
                target=self._push_writer_loop,
                daemon=True,
                name="nixl-push-writer",
            )
            self._push_writer_thread.start()
            logger.info("nixl-push-writer thread started (rank=%d)", self.tp_rank)

    def _validate_remote_agent_handshake(
        self,
        nixl_agent_meta: NixlAgentMetadata,
        remote_tp_size: int,
        remote_dcp_size: int = 1,
        local_region_ids: Sequence[int] | None = None,
    ) -> None:
        super()._validate_remote_agent_handshake(
            nixl_agent_meta, remote_tp_size, remote_dcp_size, local_region_ids
        )
        assert self.transfer_topo is not None
        if (
            self._head_sharded_draft_regions
            and self.transfer_topo.tp_size != remote_tp_size
        ):
            raise NotImplementedError(
                "NIXL push mode does not support a head-sharded draft's KV "
                "under an MLA target with heterogeneous TP. Use pull mode."
            )

    def shutdown(self) -> None:
        self._push_writer_stop.set()
        # Unblock the writer if it's waiting in the no-active-state branch.
        self._push_writer_wake.set()
        if self._push_writer_thread is not None:
            self._push_writer_thread.join(timeout=2)
            self._push_writer_thread = None
        with self._sending_transfers_lock:
            for handles in self._sending_transfers.values():
                for handle in handles:
                    self.nixl_wrapper.release_xfer_handle(handle)
            self._sending_transfers.clear()
            self._send_failures.clear()
        for st in self._early_pushes.values():
            for handle in st.inflight:
                self.nixl_wrapper.release_xfer_handle(handle)
        self._early_pushes.clear()
        super().shutdown()

    # --- Engine-main-thread entry point -------------------------------- #

    def start_load_kv(self, metadata: NixlConnectorMetadata):
        """Pre-process metadata; defer NIXL ops to the writer thread."""
        if self.pcp_rank > 0 and not self.pcp_dcp_sharded:
            return

        # D-side: track reqs waiting for P to push.
        for req_id, meta in metadata.reqs_to_recv.items():
            meta.local_physical_block_ids = self._logical_to_kernel_block_ids(
                meta.local_block_ids, self._physical_blocks_per_logical_kv_block
            )
            assert meta.remote is not None
            remote_engine_id = meta.remote.engine_id
            # Update last activity from this remote (same as pull), but only for
            # an already-connected engine so we never leave an _engine_last_active
            # entry without a _remote_agents entry (breaks _cleanup_remote_engine).
            if remote_engine_id in self._remote_agents:
                self._engine_last_active[remote_engine_id] = time.perf_counter()

            logger.debug(
                "start_load_kv (push) for request %s from remote engine %s. "
                "Num local_block_ids: %s. Num remote_block_ids: %s. ",
                req_id,
                remote_engine_id,
                len(meta.local_physical_block_ids),
                len(meta.remote.block_ids),
            )
            self._recving_metadata[req_id] = meta

        # --- D-side: registrations to send to P via NIXL ---
        if metadata.push_registrations:
            for req_id, reg_data in metadata.push_registrations.items():
                self._reg_send_inbox.put((req_id, reg_data))
            self._push_writer_wake.set()

        # --- D-side: receives whose registration watchdog expired ---
        for req_id in metadata.push_expired_recvs:
            if req_id in self._recving_metadata:
                self._failed_recv_reqs.put(req_id)

        # --- P-side: early push of saves that landed since the last step ---
        self._poll_early_saves()

        # --- P-side: newly finished blocks awaiting a D registration match ---
        if metadata.push_finished_blocks:
            for req_id, block_ids in metadata.push_finished_blocks.items():
                # Every save of a finished request has landed: hand the rest
                # to the final WRITE.
                self._early_saves.pop(req_id, None)
                self._early_off.discard(req_id)
                self._finished_blocks_inbox.put((req_id, block_ids))
            self._push_writer_wake.set()

        # Batch + lease tracking (same as pull).
        for req_id in metadata.reqs_in_batch:
            self._reqs_to_process.add(req_id)
        for req_id in metadata.reqs_not_processed:
            self._reqs_to_process.discard(req_id)
            assert req_id not in self._reqs_to_send
            # Aborted on P: D must not wait for a push that never completes.
            self._early_saves.pop(req_id, None)
            self._early_off.discard(req_id)
            self._early_abort_inbox.put(req_id)
            self._push_writer_wake.set()
        # Rebase scheduler-clock deadlines onto this worker's clock — see the
        # equivalent block in pull_worker.start_load_kv for the rationale.
        now_local = time.perf_counter()
        for req_id, expiration_time in metadata.reqs_to_send.items():
            if req_id in self._reqs_to_process:
                if metadata.scheduler_clock:
                    expiration_time = now_local + (
                        expiration_time - metadata.scheduler_clock
                    )
                self._reqs_to_send[req_id] = expiration_time

        # Heartbeats still leave from the main thread (base worker behaviour).
        self._send_heartbeats(metadata)

    # --- Writer thread ------------------------------------------------- #

    def _push_writer_loop(self) -> None:
        sleep_s = _PUSH_WRITER_POLL_INTERVAL_MS / 1000.0

        while not self._push_writer_stop.is_set():
            try:
                # 1. D registrations to send.
                while True:
                    try:
                        rid, rd = self._reg_send_inbox.get_nowait()
                    except queue.Empty:
                        break
                    self._send_registration_to_p(rid, rd)

                # 2. Deferred P→D pushes whose handshake just completed; do xfer now
                while True:
                    try:
                        rid, blocks, rd = self._deferred_push_inbox.get_nowait()
                    except queue.Empty:
                        break
                    self._do_start_push_kv(rid, blocks, rd)

                # 2b. Early push: saved blocks, aborts, WRITE completions.
                self._drain_early_inboxes()
                self._poll_early_writes()

                # 3. P-side finished blocks; match against pending regs.
                while True:
                    try:
                        rid, blocks = self._finished_blocks_inbox.get_nowait()
                    except queue.Empty:
                        break
                    matched = self._pop_matching_registration(rid)
                    if matched is not None:
                        self._do_start_push_kv(rid, blocks, matched)
                    else:
                        self._push_finished_blocks[rid] = blocks

                # 3b. Evict finished blocks for requests that have either
                # completed (WRITE acknowledged) or whose lease expired
                # without a D registration.  Drop pending registrations
                # for the same reason so we don't leak state.
                while True:
                    try:
                        rid = self._evict_finished_inbox.get_nowait()
                    except queue.Empty:
                        break
                    self._push_finished_blocks.pop(rid, None)
                    self._pending_d_registrations.pop(rid, None)
                    self._evict_early_push(rid)

                # 4. NIXL notifs: route PUSH_REG; forward the rest.
                for notifs in self.nixl_wrapper.get_new_notifs().values():
                    for notif in notifs:
                        if notif.startswith(PUSH_REG_NOTIF_PREFIX):
                            self._handle_push_reg_notif(notif)
                        else:
                            self._pending_completion_notifs.put(notif)
            except Exception:
                logger.exception("nixl-push-writer error; continuing")

            # Self-poll only while there is no other wake source: P-side
            # finished blocks or early pushes waiting for a D PUSH_REG match,
            # and early WRITEs in flight. All other progress is event-driven
            # (see module docstring).
            if self._push_finished_blocks or self._early_pushes:
                self._push_writer_stop.wait(timeout=sleep_s)
            else:
                self._push_writer_wake.wait()
                self._push_writer_wake.clear()

    def _handle_push_reg_notif(self, notif: bytes) -> None:
        try:
            reg_data = msgspec.msgpack.decode(notif[len(PUSH_REG_NOTIF_PREFIX) :])
        except Exception:
            logger.exception("Failed to decode PUSH_REG notification payload")
            return
        rid = reg_data.get("request_id") if isinstance(reg_data, dict) else None
        if not isinstance(rid, str):
            logger.warning("PUSH_REG notif missing request_id; dropping")
            return

        if self._is_aborted(rid):
            self._send_push_abort(reg_data)
            return

        match = self._pop_matching_finished_blocks(rid)
        if match is not None:
            fin_id, blocks = match
            self._do_start_push_kv(fin_id, blocks, reg_data)
        else:
            self._pending_d_registrations[rid] = reg_data
            early = self._match_early_push(rid)
            if early is not None and early[1].reg is None:
                early[1].reg = reg_data
                self._post_early_writes(*early)

    # --- D-side registration send (writer thread) ---------------------- #

    def _send_registration_to_p(
        self,
        req_id: str,
        reg_data: dict[str, Any],
    ) -> None:
        """Handshake (if needed) then send PUSH_REG. `send_notif` always
        executes on the writer; the handshake runs on the background executor
        and the request is re-queued onto `_reg_send_inbox` once it
        completes (at which point `_ensure_handshake` returns `None` and we
        send directly)."""
        remote_pp_size = reg_data.get("remote_pp_size", 1)
        fut = self._ensure_handshake(
            reg_data["remote_engine_id"],
            reg_data["remote_host"],
            reg_data["remote_port"],
            reg_data["remote_tp_size"],
            pp_size=remote_pp_size,
            # D only ever sends PUSH_REG notifs to P and never reads or writes
            # P's memory in push mode, so it never needs the transfer
            # descriptors set up by the full add_remote_agent path.
            notif_agents_only=True,
        )
        if fut is None:
            self._do_send_reg_notif(req_id, reg_data)
            return

        def _on_handshake(
            f: Future[tuple[dict[tuple[int, int], str], float]],
            rid: str = req_id,
            rd: dict[str, Any] = reg_data,
        ) -> None:
            try:
                f.result()
            except Exception as e:
                self._log_failure(
                    failure_type="push_reg_handshake_failed", req_id=rid, error=e
                )
                self._failed_recv_reqs.put(rid)
                return
            # Re-queue for the writer to send now that the handshake is done.
            self._reg_send_inbox.put((rid, rd))
            # Wake the writer so it sends the PUSH_REG promptly even if
            # otherwise parked.
            self._push_writer_wake.set()

        fut.add_done_callback(_on_handshake)

    def _do_send_reg_notif(self, req_id: str, reg_data: dict[str, Any]) -> None:
        engine_id = reg_data["remote_engine_id"]
        if self._accepts_push_parts():
            reg_data = {**reg_data, "accepts_parts": True}
        notif_msg = PUSH_REG_NOTIF_PREFIX + msgspec.msgpack.encode(reg_data)
        # _remote_agents is mutated on other threads; snapshot under the lock.
        with self._handshake_lock:
            agents = dict(self._remote_agents.get(engine_id) or {})
        if not agents:
            logger.error(
                "No remote agents for engine %s; cannot send registration for %s",
                engine_id,
                req_id,
            )
            self.xfer_stats.record_failed_notification()
            self._failed_recv_reqs.put(req_id)
            return
        for rank, agent_name in agents.items():
            try:
                self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_msg)
            except Exception as e:
                self._log_failure(
                    failure_type="push_reg_notif_failed",
                    req_id=req_id,
                    error=e,
                    remote_rank=rank,
                )
                self.xfer_stats.record_failed_notification()
                # Earlier registrations may still trigger WRITEs into D's blocks.
                # Keep the receive pending until those writes are finished.
        logger.debug(
            "Sent PUSH_REG for %s to engine %s (%dB)", req_id, engine_id, len(notif_msg)
        )

    # --- Matching helpers --------------------------------------------- #

    def _pop_matching_registration(self, request_id: str) -> dict[str, Any] | None:
        """Pop the D-side registration matching *request_id*.

        Exact key first, then a match after stripping the random suffix from
        both sides. No match leaves the request unmatched (push not started).
        """
        data = self._pending_d_registrations.pop(request_id, None)
        if data is not None:
            return data
        base_id = get_base_request_id(request_id)
        for reg_id in list(self._pending_d_registrations):
            if get_base_request_id(reg_id) == base_id:
                return self._pending_d_registrations.pop(reg_id)
        return None

    def _pop_matching_finished_blocks(
        self, request_id: str
    ) -> tuple[str, BlockIds] | None:
        """Pop the P-side finished blocks matching *request_id*.

        Same lookup as `_pop_matching_registration`: exact key, then a
        match after stripping the random suffix from both sides.
        """
        blocks = self._push_finished_blocks.pop(request_id, None)
        if blocks is not None:
            return request_id, blocks
        base_id = get_base_request_id(request_id)
        for fin_id in list(self._push_finished_blocks):
            if get_base_request_id(fin_id) == base_id:
                return fin_id, self._push_finished_blocks.pop(fin_id)
        return None

    # --- WRITE transfer logic (writer thread) ------------------------- #

    def _do_start_push_kv(
        self,
        request_id: str,
        local_block_ids: BlockIds,
        registration_data: dict[str, Any],
    ) -> None:
        """Start push-based KV transfer from P worker to D node.

        The P→D handshake runs on the base worker's background executor.
        If it isn't ready yet we register a completion callback, defer the
        WRITE, and re-drive this request via `_deferred_push_inbox` once
        the handshake resolves -- so the writer thread never blocks on the
        network (mirrors `_send_registration_to_p`).
        """
        if not local_block_ids:
            logger.warning("No local blocks to push for request %s", request_id)
            self._evict_early_push(request_id)
            return

        early = self._early_pushes.get(request_id)
        if early is not None:
            if early.inflight:
                # Completion must not reach D before every early WRITE landed;
                # ``_poll_early_writes`` re-drives this once they are done.
                early.finish = (local_block_ids, registration_data)
                return
            del self._early_pushes[request_id]

        # ``local_block_ids`` are P's logical block IDs; ``remote_block_ids``
        # (D's, from the PUSH_REG notif) are also logical.
        decode_engine_id = registration_data["decode_engine_id"]
        remote_block_ids = registration_data["local_block_ids"]
        decode_request_id = registration_data["request_id"]

        # Runs on the background executor; defer the WRITE until it's ready.
        fut = self._ensure_handshake(
            decode_engine_id,
            registration_data["decode_host"],
            registration_data["decode_port"],
            registration_data["decode_tp_size"],
        )
        if fut is not None:

            def _on_handshake(
                f: Future[tuple[dict[tuple[int, int], str], float]],
                rid: str = request_id,
                blocks: BlockIds = local_block_ids,
                rd: dict[str, Any] = registration_data,
            ) -> None:
                if (e := f.exception()) is not None:
                    # The engine reclaims the blocks via the TTL so we dont free here
                    self._log_failure(
                        failure_type="push_handshake_failed", req_id=rid, error=e
                    )
                    return
                self._deferred_push_inbox.put((rid, blocks, rd))
                self._push_writer_wake.set()

            fut.add_done_callback(_on_handshake)
            return
        # Keep the engine alive while it is actively receiving pushes, mirroring
        # how pull-mode transfers touch _engine_last_active in start_load_kv.
        self._engine_last_active[decode_engine_id] = time.perf_counter()

        logical_local = self._as_grouped_block_ids(local_block_ids)
        logical_remote = self._as_grouped_block_ids(remote_block_ids)
        if early is not None:
            logical_local, logical_remote = self._skip_early_written(
                request_id, early, logical_local, logical_remote
            )
        physical_local = self._logical_to_kernel_block_ids(
            logical_local, self._physical_blocks_per_logical_kv_block
        )

        push_meta = ReqMeta(
            local_block_ids=logical_local,
            local_physical_block_ids=physical_local,
            tp_size=self.world_size,
            remote=RemoteMeta(
                block_ids=logical_remote,
                host="",
                port=0,
                engine_id=decode_engine_id,
                request_id=decode_request_id,
            ),
        )

        t0 = time.perf_counter()
        self._xfer_blocks_for_req(req_id=request_id, meta=push_meta)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if elapsed_ms > 200.0:
            logger.warning(
                "_do_start_push_kv for %s took %.1fms (slow NIXL submission)",
                request_id,
                elapsed_ms,
            )

    @staticmethod
    def _as_grouped_block_ids(block_ids: BlockIds) -> BlockIds:
        """Normalise a sequence of block IDs to a tuple-of-groups shape.

        `BlockIds` is canonically a tuple of per-group lists, but some
        registration payloads collapse a single-group case to a flat
        list. Re-wrap that case so downstream group-aware helpers see a
        consistent shape."""
        if block_ids and not isinstance(block_ids[0], (list, tuple)):
            return (list(block_ids),)
        return block_ids

    def _xfer_blocks_for_req(self, req_id: str, meta: ReqMeta):
        """Issue WRITE transfers to one or more remote TP ranks."""
        assert meta.remote is not None and self.transfer_topo is not None
        engine_id = meta.remote.engine_id
        plan = self.tp_mappings[engine_id]
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        tp_ratio = self.transfer_topo.tp_ratio(remote_info.remote_tp_size)

        # Expand D's logical IDs using the ratio learned during the
        # NIXL handshake. ``meta`` is freshly built by
        # ``_do_start_push_kv`` so mutating it here is safe.
        meta.remote.block_ids = self._logical_to_kernel_block_ids(
            meta.remote.block_ids,
            remote_info.remote_physical_blocks_per_logical,
        )
        remote_block_ids = meta.remote.block_ids
        local_block_ids = meta.local_physical_block_ids
        if not self._dst_pairs[engine_id].groups_match:
            raise NotImplementedError(
                "NixlPushConnector does not support different producer and "
                "consumer cache-group layouts"
            )

        # MLA latent is replicated across D's TP ranks: the tp-mapping
        # collapses it to one rank (fine for reads), but push must WRITE every
        # D rank or the rest decode stale KV. For hybrid MLA+SSM the sharded
        # SSM state already targets every covered D rank, so only the
        # attention groups need widening; pure MLA writes to all handshaked
        # ranks (only the dst differs per rank).
        replicate_attn = self.use_mla and tp_ratio < 0
        if replicate_attn and not self._has_mamba:
            assert len(plan.all_source_ranks) == 1
            write_ranks = sorted(self.dst_xfer_side_handles[engine_id])
        else:
            write_ranks = list(plan.all_source_ranks)

        num_groups = len(local_block_ids)

        def group_ids(block_ids: BlockIds, rank: int) -> BlockIds:
            return [
                list(block_ids[g])
                if (self._is_csa_linear and tp_ratio < 0)
                or (replicate_attn and _is_attention_spec(self._group_spec_types[g]))
                or rank in plan.source_ranks_per_group[g]
                else []
                for g in range(num_groups)
            ]

        read_specs = [
            ReadSpec(
                remote_rank=rank,
                local_block_ids=group_ids(local_block_ids, rank),
                remote_block_ids=group_ids(remote_block_ids, rank),
            )
            for rank in write_ranks
        ]

        handles: list[int] = []
        for i, spec in enumerate(read_specs):
            remote_block_size = remote_info.remote_block_size
            logger.debug(
                "Remote agent %s available, calling _xfer_blocks"
                " on remote rank %s with remote block size %s for req %s",
                meta.remote.engine_id,
                spec.remote_rank,
                remote_block_size,
                req_id,
            )
            if tp_ratio < 0 and (not self.use_mla or len(plan.all_source_ranks) > 1):
                # Multiple targets: write each rank its chunk of local memory.
                # Hybrid MLA+SSM also lands here: its split handles replicate
                # the attention descriptors and chunk only the SSM state.
                split_key = (tp_ratio, remote_block_size)
                local_xfer_side_handle = self.src_xfer_handles_by_tp_ratio[split_key][i]
            else:
                local_xfer_side_handle = self.src_xfer_handles_by_block_size[
                    remote_block_size
                ]

            remote_xfer_side_handle = self.dst_xfer_side_handles[meta.remote.engine_id][
                spec.remote_rank
            ]

            handle = self._xfer_blocks(
                read_spec=spec,
                request_id=req_id,
                dst_engine_id=meta.remote.engine_id,
                remote_request_id=meta.remote.request_id,
                local_xfer_side_handle=local_xfer_side_handle,
                remote_xfer_side_handle=remote_xfer_side_handle,
            )
            if handle is not None:
                handles.append(handle)

        # Publish all the request's WRITE handles in one locked update: a
        # partial set would let ``_pop_done_transfers`` finish the request
        # early, then double-report it as the remaining writes land.
        if handles:
            with self._sending_transfers_lock:
                self._sending_transfers[req_id].extend(handles)

    def _xfer_blocks(
        self,
        read_spec: ReadSpec,
        dst_engine_id: str,
        request_id: str,
        remote_request_id: str,
        local_xfer_side_handle: int,
        remote_xfer_side_handle: int,
    ) -> int | None:
        """Post a WRITE point-to-point xfer request.

        Returns the in-flight transfer handle (so the caller can track all of
        a request's handles atomically), or `None` if nothing was submitted.
        """
        assert self.transfer_topo is not None
        remote_rank = read_spec.remote_rank
        local_block_ids = read_spec.local_block_ids
        remote_block_ids = read_spec.remote_block_ids

        remote_info = self.transfer_topo.get_engine_info(dst_engine_id)

        notif_id = f"{remote_request_id}:{self.world_size}".encode()

        if len(local_block_ids) == 0:
            logger.warning("No blocks to push for request %s", request_id)
            return None

        # Prefix caching: D allocated only uncached blocks, so on a partial hit it
        # sends fewer than P's. End-trim P's blocks to that same suffix so we WRITE only
        # the uncomputed tail into D's slots. Runs on kernel ids, post-expansion.
        remote_block_ids, local_block_ids = self._apply_prefix_caching(
            decode_block_ids=remote_block_ids,
            prefill_block_ids=local_block_ids,
            decode_physical_per_logical=remote_info.remote_physical_blocks_per_logical,
            prefill_physical_per_logical=self._physical_blocks_per_logical_kv_block,
        )

        local_block_ids = list(local_block_ids)
        remote_block_ids = list(remote_block_ids)
        assert len(local_block_ids) == len(remote_block_ids), (
            f"push group-count mismatch for {request_id}: {len(local_block_ids)} "
            f"local vs {len(remote_block_ids)} remote groups"
        )
        for i in range(len(local_block_ids)):
            assert len(local_block_ids[i]) == len(remote_block_ids[i]), (
                f"push block-count mismatch for {request_id} group {i}: "
                f"{len(local_block_ids[i])} local vs "
                f"{len(remote_block_ids[i])} remote blocks"
            )

        local_block_descs_ids, remote_block_descs_ids = self._write_desc_ids(
            local_block_ids, remote_block_ids, dst_engine_id
        )

        handle = None
        try:
            handle = self.nixl_wrapper.make_prepped_xfer(
                "WRITE",
                local_xfer_side_handle,
                local_block_descs_ids,
                remote_xfer_side_handle,
                remote_block_descs_ids,
                notif_msg=notif_id,
            )
            self.nixl_wrapper.transfer(handle)
            # Caller tracks the handle (atomically with the request's other
            # writes) so P can free blocks once all of them are done.
            return handle
        except Exception as e:
            self._log_failure(
                failure_type="transfer_setup_failed",
                req_id=request_id,
                msg="Push WRITE submission failed; releasing handle",
                error=e,
                dst_engine_id=dst_engine_id,
                remote_rank=remote_rank,
            )
            # On the P side this WRITE failure is purely outbound; we
            # don't have a ``_recving_metadata`` entry to invalidate, so
            # we just release the handle and let the engine reschedule
            # via the lease / watchdog.
            with self._sending_transfers_lock:
                self._send_failures.add(request_id)
            if not self._handle_failed_transfer(request_id, handle):
                return handle
            return None

    def _write_desc_ids(
        self,
        local_block_ids: BlockIds,
        remote_block_ids: BlockIds,
        dst_engine_id: str,
    ):
        """Descriptor ids of a WRITE of kernel blocks to *dst_engine_id*.

        Pair k writes local region pairs.local[k] into remote region
        pairs.remote[k] at its cache group's blocks."""
        assert self.transfer_topo is not None
        remote_info = self.transfer_topo.get_engine_info(dst_engine_id)
        pairs = self._dst_pairs[dst_engine_id]
        remote_block_descs_ids = self._compute_desc_ids(
            block_ids=remote_block_ids,
            dst_num_blocks=self.dst_num_blocks[dst_engine_id],
            block_size_ratio=None,
            physical_blocks_per_logical=remote_info.remote_physical_blocks_per_logical,
            region_num_blocks=self.dst_region_num_blocks[dst_engine_id],
            pairs=(pairs.remote, pairs.group),
            ssm_regions=pairs.remote_ssm_regions,
        )
        local_block_descs_ids = self._compute_desc_ids(
            block_ids=local_block_ids,
            dst_num_blocks=self.dst_num_blocks[self.engine_id],
            block_size_ratio=None,
            physical_blocks_per_logical=self._physical_blocks_per_logical_kv_block,
            region_num_blocks=self.dst_region_num_blocks[self.engine_id],
            pairs=(pairs.local, pairs.group),
        )
        assert len(local_block_descs_ids) == len(remote_block_descs_ids)
        return local_block_descs_ids, remote_block_descs_ids

    # --- Early push: main thread --------------------------------------- #

    def save_kv_to_host(self, metadata: NixlConnectorMetadata) -> dict[ReqId, list]:
        handles = super().save_kv_to_host(metadata)
        synchronous = getattr(self.copy_blocks, "synchronous", False)
        for req_id, meta in metadata.reqs_to_save.items():
            saves = self._early_saves.get(req_id)
            if req_id in self._early_off:
                if meta.push_last_save:
                    self._early_off.discard(req_id)
                continue
            if meta.push_total_blocks > 0 and not handles.get(req_id) and not synchronous:
                # Nothing says when this save lands: WRITEing early could send
                # stale host bytes. Blocks already handed over were saved.
                self._early_saves.pop(req_id, None)
                if not meta.push_last_save:
                    self._early_off.add(req_id)
                continue
            if meta.push_total_blocks <= 0:
                if saves is not None:
                    # The scheduler stopped early push for this request (its
                    # blocks were rebuilt): written positions no longer hold.
                    del self._early_saves[req_id]
                    self._early_abort_inbox.put(req_id)
                    self._push_writer_wake.set()
                continue
            if saves is None:
                saves = self._early_saves[req_id] = _EarlySaves(
                    total=meta.push_total_blocks
                )
            saves.blocks.extend(meta.local_block_ids[0])
            saves.pending.append((len(saves.blocks), handles.get(req_id, [])))
            saves.final = meta.push_final_blocks
        self._poll_early_saves()
        for req_id, meta in metadata.reqs_to_save.items():
            if meta.push_last_save:
                # The rest goes with the final WRITE.
                self._early_saves.pop(req_id, None)
        return handles

    def _poll_early_saves(self) -> None:
        """Hand saved final blocks to the writer. Runs on the model thread,
        where the copy op's handles are polled."""
        handed = False
        for req_id, saves in list(self._early_saves.items()):
            try:
                while saves.pending and all(h.done() for h in saves.pending[0][1]):
                    saves.done = saves.pending.popleft()[0]
            except Exception as e:
                # The step that completes the prompt waits for its saves and
                # raises; here, just stop pushing this request early.
                logger.warning("Early push of %s stopped: save failed (%s)", req_id, e)
                del self._early_saves[req_id]
                self._early_off.add(req_id)
                continue
            # The prompt's last block always goes with the final WRITE, which
            # therefore never has an empty block list.
            ready = min(saves.done, saves.final, saves.total - 1)
            if ready > saves.handed:
                blocks = [
                    (pos, saves.blocks[pos]) for pos in range(saves.handed, ready)
                ]
                self._early_blocks_inbox.put((req_id, saves.total, blocks))
                saves.handed = ready
                handed = True
        if handed:
            self._push_writer_wake.set()

    # --- Early push: writer thread ------------------------------------- #

    def _drain_early_inboxes(self) -> None:
        while True:
            try:
                rid, total, blocks = self._early_blocks_inbox.get_nowait()
            except queue.Empty:
                break
            st = self._early_pushes.get(rid)
            if st is None:
                st = self._early_pushes[rid] = _EarlyPush(total=total)
                st.reg = self._peek_matching_registration(rid)
            st.ready.update(blocks)
            self._post_early_writes(rid, st)
        while True:
            try:
                rid = self._early_abort_inbox.get_nowait()
            except queue.Empty:
                break
            self._abort_early_push(rid)

    def _peek_matching_registration(self, request_id: str) -> dict[str, Any] | None:
        data = self._pending_d_registrations.get(request_id)
        if data is not None:
            return data
        base_id = get_base_request_id(request_id)
        for reg_id, data in self._pending_d_registrations.items():
            if get_base_request_id(reg_id) == base_id:
                return data
        return None

    def _match_early_push(self, request_id: str) -> tuple[str, _EarlyPush] | None:
        st = self._early_pushes.get(request_id)
        if st is not None:
            return request_id, st
        base_id = get_base_request_id(request_id)
        for rid, st in self._early_pushes.items():
            if get_base_request_id(rid) == base_id:
                return rid, st
        return None

    def _early_write_target(self, rid: str, st: _EarlyPush) -> tuple[str, int] | None:
        """D engine and rank for early WRITEs, or None to leave every block to
        the final WRITE. Only a layout where P block `i` is exactly D block
        `i - offset` qualifies: same TP, block size and kernel blocks per
        logical block, one destination rank, same cache groups."""
        assert st.reg is not None and self.transfer_topo is not None
        engine_id = st.reg["decode_engine_id"]
        if engine_id not in self.tp_mappings:
            return None
        info = self.transfer_topo.get_engine_info(engine_id)
        plan = self.tp_mappings[engine_id]
        if (
            info.remote_tp_size != self.world_size
            or info.remote_block_size != self.block_size
            or info.remote_physical_blocks_per_logical
            != self._physical_blocks_per_logical_kv_block
            or len(plan.all_source_ranks) != 1
            or not self._dst_pairs[engine_id].groups_match
        ):
            return None
        return engine_id, plan.all_source_ranks[0]

    def _post_early_writes(self, rid: str, st: _EarlyPush) -> None:
        if (
            st.failed
            or st.abort
            or st.handshaking
            or st.finish is not None
            or st.reg is None
            or not st.ready
        ):
            return
        reg = st.reg
        fut = self._ensure_handshake(
            reg["decode_engine_id"],
            reg["decode_host"],
            reg["decode_port"],
            reg["decode_tp_size"],
        )
        if fut is not None:
            # `_poll_early_writes` retries once it is done.
            st.handshaking = True

            def _on_handshake(f: Future, st: _EarlyPush = st) -> None:
                if f.exception() is not None:
                    st.failed = True
                st.handshaking = False
                self._push_writer_wake.set()

            fut.add_done_callback(_on_handshake)
            return
        target = self._early_write_target(rid, st)
        d_blocks = self._as_grouped_block_ids(reg["local_block_ids"])
        offset = st.total - len(d_blocks[0]) if len(d_blocks) == 1 else -1
        if target is None or offset < 0:
            st.failed = True
            st.ready.clear()
            return
        engine_id, rank = target
        positions = sorted(p for p in st.ready if p >= offset)
        for p in [p for p in st.ready if p < offset]:
            # D holds this block from its prefix cache.
            st.written[p] = st.ready.pop(p)
        if not positions:
            return
        local = [st.ready[p] for p in positions]
        remote = [d_blocks[0][p - offset] for p in positions]
        notif = b""
        if reg.get("accepts_parts"):
            notif = PUSH_PART_NOTIF_PREFIX + msgspec.msgpack.encode(
                {"request_id": reg["request_id"], "block_ids": remote}
            )
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        local_phys = self._logical_to_kernel_block_ids(
            (local,), self._physical_blocks_per_logical_kv_block
        )
        remote_phys = self._logical_to_kernel_block_ids(
            (remote,), remote_info.remote_physical_blocks_per_logical
        )
        handle = None
        try:
            local_descs, remote_descs = self._write_desc_ids(
                local_phys, remote_phys, engine_id
            )
            handle = self.nixl_wrapper.make_prepped_xfer(
                "WRITE",
                self.src_xfer_handles_by_block_size[remote_info.remote_block_size],
                local_descs,
                self.dst_xfer_side_handles[engine_id][rank],
                remote_descs,
                notif_msg=notif,
            )
            self.nixl_wrapper.transfer(handle)
        except Exception as e:
            self._log_failure(
                failure_type="early_push_setup_failed",
                req_id=rid,
                msg="Early push WRITE submission failed; the final WRITE covers "
                "every block",
                error=e,
                dst_engine_id=engine_id,
            )
            st.failed = True
            if handle is not None and not self._try_release_xfer_handle(rid, handle):
                st.inflight.append(handle)
            return
        self._engine_last_active[engine_id] = time.perf_counter()
        st.inflight.append(handle)
        for p in positions:
            st.written[p] = st.ready.pop(p)

    def _poll_early_writes(self) -> None:
        for rid, st in list(self._early_pushes.items()):
            in_progress = []
            for handle in st.inflight:
                try:
                    state = self.nixl_wrapper.check_xfer_state(handle)
                except Exception as e:
                    state = f"exception: {e}"
                if state == "PROC":
                    in_progress.append(handle)
                    continue
                if state == "DONE":
                    self.xfer_stats.record_transfer(
                        self.nixl_wrapper.get_xfer_telemetry(handle)
                    )
                else:
                    self._log_failure(
                        failure_type="early_push_failed",
                        req_id=rid,
                        xfer_state=state,
                    )
                    st.failed = True
                if not self._try_release_xfer_handle(rid, handle):
                    in_progress.append(handle)
            st.inflight = in_progress
            if in_progress:
                continue
            if st.ready:
                self._post_early_writes(rid, st)
                if st.inflight:
                    continue
            if st.evicted:
                del self._early_pushes[rid]
            elif st.abort:
                del self._early_pushes[rid]
                self._finish_abort(rid, st.reg)
            elif st.finish is not None:
                blocks, reg = st.finish
                st.finish = None
                self._do_start_push_kv(rid, blocks, reg)

    def _skip_early_written(
        self,
        rid: str,
        st: _EarlyPush,
        local: BlockIds,
        remote: BlockIds,
    ) -> tuple[BlockIds, BlockIds]:
        """Drop the blocks early WRITEs already delivered from the final push.

        Falls back to every block when the early WRITEs failed, went to
        another D request, or the request's blocks no longer sit where they
        were written."""
        if st.failed or not st.written or len(local) != 1 or len(remote) != 1:
            return local, remote
        p_blocks, d_blocks = list(local[0]), list(remote[0])
        offset = len(p_blocks) - len(d_blocks)
        if (
            st.total != len(p_blocks)
            or offset < 0
            or st.reg is None
            or list(self._as_grouped_block_ids(st.reg["local_block_ids"])[0])
            != d_blocks
            or any(p_blocks[p] != b for p, b in st.written.items())
        ):
            logger.warning(
                "Early push of %s does not match its final blocks; pushing every block",
                rid,
            )
            return local, remote
        keep = [p for p in range(offset, len(p_blocks)) if p not in st.written]
        return ([p_blocks[p] for p in keep],), ([d_blocks[p - offset] for p in keep],)

    def _evict_early_push(self, rid: str) -> None:
        st = self._early_pushes.get(rid)
        if st is None:
            return
        if st.inflight:
            st.evicted = True
        else:
            del self._early_pushes[rid]

    def _abort_early_push(self, rid: str) -> None:
        """P aborted *rid*: once its early WRITEs are done, tell D."""
        self._push_finished_blocks.pop(rid, None)
        st = self._early_pushes.get(rid)
        reg = self._pop_matching_registration(rid)
        if st is not None:
            st.reg = st.reg or reg
            st.abort = True
            st.ready.clear()
            if st.inflight:
                return
            del self._early_pushes[rid]
            reg = st.reg
        self._finish_abort(rid, reg)

    def _finish_abort(self, rid: str, reg: dict[str, Any] | None) -> None:
        if reg is not None:
            self._send_push_abort(reg)
        now = time.perf_counter()
        for old, until in list(self._aborted_reqs.items()):
            if until < now:
                del self._aborted_reqs[old]
        self._aborted_reqs[rid] = now + self._aborted_req_ttl

    def _is_aborted(self, request_id: str) -> bool:
        if not self._aborted_reqs:
            return False
        if request_id in self._aborted_reqs:
            return True
        base_id = get_base_request_id(request_id)
        return any(get_base_request_id(rid) == base_id for rid in self._aborted_reqs)

    def _send_push_abort(self, reg: dict[str, Any]) -> None:
        """Best effort: without it D's registration watchdog fails the receive."""
        engine_id = reg["decode_engine_id"]
        with self._handshake_lock:
            agents = dict(self._remote_agents.get(engine_id) or {})
        if not agents:
            logger.warning(
                "Cannot send PUSH_ABORT for %s: no connection to engine %s",
                reg["request_id"],
                engine_id,
            )
            return
        msg = PUSH_ABORT_NOTIF_PREFIX + str(reg["request_id"]).encode()
        for agent_name in agents.values():
            try:
                self.nixl_wrapper.send_notif(agent_name, notif_msg=msg)
            except Exception as e:
                self._log_failure(
                    failure_type="push_abort_notif_failed",
                    req_id=reg["request_id"],
                    error=e,
                )

    # --- D side: early parts ------------------------------------------- #

    def _accepts_push_parts(self) -> bool:
        """D copies early-written blocks to the device as they land (host
        buffer only; device-buffer WRITEs need no copy)."""
        return bool(self.use_host_buffer and self.copy_blocks is not None)

    def _handle_push_part(self, payload: bytes) -> None:
        try:
            part = msgspec.msgpack.decode(payload)
            req_id, block_ids = part["request_id"], list(part["block_ids"])
        except Exception:
            logger.exception("Failed to decode PUSH_PART notification payload")
            return
        if req_id not in self._recving_metadata or self.copy_blocks is None:
            return
        physical = self._logical_to_kernel_block_ids(
            (block_ids,), self._physical_blocks_per_logical_kv_block
        )[0]
        # Early push is full-attention only: one transfer group.
        layer_names = self.kv_cache_config.transfer_groups[0].layer_names
        with gpu_sync_allowed():
            self.copy_blocks(
                {n: self.host_xfer_buffers[n] for n in layer_names},
                {n: self.device_kv_caches[n] for n in layer_names},
                physical,
                physical,
                "h2d",
            )
        self._push_parts_loaded.setdefault(req_id, set()).update(physical)

    def sync_recved_kv_to_device(self, req_id: str, meta: ReqMeta):
        loaded = self._push_parts_loaded.pop(req_id, None)
        if loaded:
            meta = dataclasses.replace(
                meta,
                local_physical_block_ids=tuple(
                    [b for b in group if b not in loaded]
                    for group in meta.local_physical_block_ids
                ),
            )
        super().sync_recved_kv_to_device(req_id, meta)

    # --- Notification handling on engine main thread ------------------ #

    def _get_new_notifs(self) -> set[str]:
        """Drain HB / completion notifs forwarded by the writer thread.

        The writer owns `nixl_wrapper.get_new_notifs` for push; PUSH_REG
        notifs are handled there. P learns of finished pushes from its WRITE
        handles, so this never reports finished sends.
        """
        while True:
            try:
                notif = self._pending_completion_notifs.get_nowait()
            except queue.Empty:
                break

            if notif.startswith(PUSH_PART_NOTIF_PREFIX):
                self._handle_push_part(notif[len(PUSH_PART_NOTIF_PREFIX) :])
                continue
            if notif.startswith(PUSH_ABORT_NOTIF_PREFIX):
                req_id = notif[len(PUSH_ABORT_NOTIF_PREFIX) :].decode()
                if req_id in self._recving_metadata:
                    logger.warning("Prefill aborted request %s; failing it", req_id)
                    self._failed_recv_reqs.put(req_id)
                continue

            msg = notif.decode("utf-8")
            if msg.startswith("HB:"):
                self._handle_heartbeat(msg[3:])
                continue

            req_id, tp_size = msg.rsplit(":", 1)

            if (meta := self._recving_metadata.get(req_id)) is None:
                # The lease may have expired before the notif arrived.
                self.xfer_stats.record_notification_after_expiry()
                logger.error(
                    "Unrecognized request %s notif (may have expired).", req_id
                )
                continue
            # Consumer waits for one notif per producer rank writing
            # here: pp_size stages * producers-per-consumer (>1 when
            # producer TP > consumer TP; tp_size is the producer TP).
            producers_per_consumer = max(1, int(tp_size) // self.world_size)
            expected_notifs = meta.pp_size * producers_per_consumer
            self.consumer_notification_counts_by_req[req_id] += 1
            if self.consumer_notification_counts_by_req[req_id] < expected_notifs:
                continue
            del self.consumer_notification_counts_by_req[req_id]
            # P drove the transfer (we own no NIXL handle), so
            # materialise an empty ``_recving_transfers`` entry for
            # ``_pop_done_transfers`` to report done.
            self._recving_transfers.setdefault(req_id, [])
        return set()

    def get_transfer_results(self) -> KVConnectorTransferResults:
        # Engine main thread asking for completions: also wake the writer
        # so it gets a chance to drain NIXL notifs (heartbeats, completion
        # notifs, late PUSH_REGs) even if it had been parked.
        self._push_writer_wake.set()

        self._poll_early_saves()
        results = super().get_transfer_results()
        done_sending = results.finished_sending
        for req_id in results.finished_recving:
            self._push_parts_loaded.pop(req_id, None)

        # ``_pop_done_transfers`` mutates ``_sending_transfers``; the
        # writer thread also appends to it, so guard the pop.
        with self._sending_transfers_lock:
            done_pushing, failed_pushing = self._pop_done_transfers(
                self._sending_transfers
            )
            # Remember failures until the final sibling WRITE completes.
            self._send_failures.update(failed_pushing)
            successful = {
                req_id
                for req_id in done_pushing - self._send_failures
                if req_id in self._reqs_to_send or req_id in self._reqs_to_process
            }
            self._send_failures.difference_update(done_pushing | done_sending)
        # Expired requests were already reported, even if their WRITEs finish later.
        done_pushing = successful
        for req_id in done_pushing:
            self._reqs_to_send.pop(req_id, None)
            self._reqs_to_process.discard(req_id)
            done_sending.add(req_id)

        # Tell the writer to drop any state it still holds for any
        # request that just finished (push completed) or expired
        # (lease ran out without a D registration ever arriving).
        for req_id in done_sending:
            self._evict_finished_inbox.put(req_id)
        if done_sending:
            self._push_writer_wake.set()

        return results
