from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import json
from pathlib import Path


class RunPhase(IntEnum):
    DATA_AUDITED = 1
    SPLITS_FROZEN = 2
    SOURCE_FITTING_COMPLETE = 3
    PARETO_FROZEN = 4
    TARGET_UNLOCKED = 5
    TARGET_EVALUATED = 6
    RESULTS_FINALIZED = 7


@dataclass
class RunState:
    path: Path
    phase: RunPhase = RunPhase.DATA_AUDITED
    target_reads: int = 0
    target_unlocks: int = 0
    target_recording_ids: tuple[str, ...] = ()
    target_read_status: str = "none"
    target_completed_recording_ids: tuple[str, ...] = ()
    target_cache_manifest_sha256: str | None = None
    source_artifact_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.path.exists():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        try:
            self.phase = RunPhase[payload["phase"]]
            self.target_reads = int(payload["target_reads"])
            self.target_unlocks = int(payload["target_unlocks"])
            self.target_recording_ids = tuple(str(v) for v in payload["target_recording_ids"])
            self.target_read_status = str(payload.get("target_read_status", "complete" if self.target_reads else "none"))
            self.target_completed_recording_ids = tuple(
                str(v) for v in payload.get("target_completed_recording_ids", [])
            )
            self.target_cache_manifest_sha256 = payload.get("target_cache_manifest_sha256")
            self.source_artifact_hash = payload.get("source_artifact_hash")
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"Invalid persisted run state at {self.path}") from error
        if self.target_reads not in {0, 1} or self.target_unlocks not in {0, 1}:
            raise RuntimeError("Persisted target ledger violates the one-read contract")

    def transition(self, next_phase: RunPhase) -> None:
        if next_phase != self.phase + 1:
            raise RuntimeError(f"Illegal run-state transition {self.phase.name} -> {next_phase.name}")
        if next_phase == RunPhase.TARGET_UNLOCKED and not self.source_artifact_hash:
            raise RuntimeError("Target unlock requires a frozen source artifact hash")
        self.phase = next_phase
        if next_phase == RunPhase.TARGET_UNLOCKED:
            self.target_unlocks += 1
            if self.target_unlocks > 1:
                raise RuntimeError("Target may be unlocked only once per fold")
        self.save()

    def record_target_load(self, recording_ids: list[str]) -> None:
        if self.phase != RunPhase.TARGET_UNLOCKED:
            raise PermissionError("Target data cannot be materialized before TARGET_UNLOCKED")
        if self.target_reads >= 1:
            raise PermissionError("Held target can be materialized only once per fold")
        self.target_reads += 1
        self.target_recording_ids = tuple(recording_ids)
        self.target_read_status = "complete"
        self.target_completed_recording_ids = tuple(recording_ids)
        self.save()

    def begin_target_stream(self, recording_ids: list[str]) -> None:
        if self.phase != RunPhase.TARGET_UNLOCKED:
            raise PermissionError("Target stream requires TARGET_UNLOCKED")
        requested = tuple(recording_ids)
        if self.target_read_status == "in_progress":
            if requested != self.target_recording_ids:
                raise PermissionError("Cannot change the target set while resuming")
            return
        if self.target_reads >= 1:
            raise PermissionError("Held target can have only one logical read session")
        self.target_reads = 1
        self.target_recording_ids = requested
        self.target_read_status = "in_progress"
        self.save()

    def authorize_target_recording(self, recording_id: str) -> None:
        if self.phase != RunPhase.TARGET_UNLOCKED or self.target_read_status != "in_progress":
            raise PermissionError("No resumable target read session is active")
        if recording_id not in self.target_recording_ids:
            raise PermissionError("Recording is outside the frozen target set")
        if recording_id in self.target_completed_recording_ids:
            raise PermissionError("Target recording was already checkpointed")

    def complete_target_recording(self, recording_id: str) -> None:
        self.authorize_target_recording(recording_id)
        self.target_completed_recording_ids = tuple(
            [*self.target_completed_recording_ids, recording_id]
        )
        self.save()

    def commit_target_stream(self, cache_manifest_sha256: str) -> None:
        if self.target_read_status != "in_progress":
            raise RuntimeError("No target stream is in progress")
        if set(self.target_completed_recording_ids) != set(self.target_recording_ids):
            raise RuntimeError("Cannot commit an incomplete target cache")
        self.target_read_status = "complete"
        self.target_cache_manifest_sha256 = cache_manifest_sha256
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "phase": self.phase.name,
                "target_reads": self.target_reads,
                "target_unlocks": self.target_unlocks,
                "target_recording_ids": list(self.target_recording_ids),
                "target_read_status": self.target_read_status,
                "target_completed_recording_ids": list(self.target_completed_recording_ids),
                "target_cache_manifest_sha256": self.target_cache_manifest_sha256,
                "source_artifact_hash": self.source_artifact_hash,
            },
            indent=2,
        )
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(self.path)
