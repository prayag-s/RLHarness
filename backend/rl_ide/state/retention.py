"""Time-based retention: data nobody has touched must eventually disappear.

Deletion was always explicit — a user had to ask. Under a DPA that is not enough: the policy has to be
"rows older than N days are erased", applied by the product, not by an operator remembering to run it.
This module sweeps expired conversations, datasets and exported bundles on a schedule and reports
what it removed.

Rules that matter:

* **Disabled by default.** A desktop studio must never delete a user's work; a hosted deployment opts
  in with `RL_IDE_RETENTION_DAYS` (or an explicit policy file).
* **Dry-run first.** ``sweep(dry_run=True)`` reports exactly what would go, so an operator can check
  the policy against real data before arming it.
* **One failure never stops the sweep.** A locked file or a half-deleted workspace is logged and the
  rest continues.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


@dataclass
class RetentionPolicy:
    """How long each kind of artefact is kept. ``None`` means "keep forever"."""

    conversation_days: int | None = None
    dataset_days: int | None = None
    bundle_days: int | None = None
    sweep_interval_seconds: int = 3600

    @property
    def enabled(self) -> bool:
        return any(
            value is not None
            for value in (self.conversation_days, self.dataset_days, self.bundle_days)
        )

    @classmethod
    def from_env(cls) -> "RetentionPolicy":
        def _days(name: str, default: int | None) -> int | None:
            raw = (os.environ.get(name) or "").strip()
            if not raw:
                return default
            try:
                value = int(raw)
            except ValueError:
                return default
            return value if value > 0 else None

        # One knob for the common case, plus per-artefact overrides.
        base = _days("RL_IDE_RETENTION_DAYS", None)
        policy = cls(
            conversation_days=_days("RL_IDE_RETENTION_CONVERSATION_DAYS", base),
            dataset_days=_days("RL_IDE_RETENTION_DATASET_DAYS", base),
            bundle_days=_days("RL_IDE_RETENTION_BUNDLE_DAYS", base),
            sweep_interval_seconds=int(os.environ.get("RL_IDE_RETENTION_INTERVAL_SECONDS", "3600") or 3600),
        )
        return policy

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "conversation_days": self.conversation_days,
            "dataset_days": self.dataset_days,
            "bundle_days": self.bundle_days,
            "sweep_interval_seconds": self.sweep_interval_seconds,
        }


@dataclass
class SweepReport:
    deleted: dict[str, list[str]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    dry_run: bool = False

    def add(self, kind: str, ident: str) -> None:
        self.deleted.setdefault(kind, []).append(ident)

    @property
    def total(self) -> int:
        return sum(len(v) for v in self.deleted.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "total": self.total,
            "deleted": self.deleted,
            "errors": self.errors,
        }


class RetentionSweeper:
    """Applies a policy to the conversation store, the dataset store and the bundle registry."""

    def __init__(
        self,
        policy: RetentionPolicy,
        *,
        conversation_store_factory: Callable[[], Any] | None = None,
        dataset_store_factory: Callable[[], Any] | None = None,
        registry_factory: Callable[[], Any] | None = None,
        now: Callable[[], datetime] | None = None,
    ):
        self.policy = policy
        self._conversations = conversation_store_factory
        self._datasets = dataset_store_factory
        self._registry = registry_factory
        self._now = now or (lambda: datetime.now(timezone.utc))

    def _cutoff(self, days: int | None) -> datetime | None:
        return None if days is None else self._now() - timedelta(days=days)

    def sweep(self, *, dry_run: bool = False) -> SweepReport:
        report = SweepReport(dry_run=dry_run)
        if not self.policy.enabled:
            return report

        self._sweep_conversations(report, dry_run)
        self._sweep_datasets(report, dry_run)
        self._sweep_bundles(report, dry_run)
        if report.total and not dry_run:
            logger.info("retention sweep removed %s artefacts", report.total)
        return report

    # ------------------------------------------------------------- helpers
    def _sweep_conversations(self, report: SweepReport, dry_run: bool) -> None:
        cutoff = self._cutoff(self.policy.conversation_days)
        if cutoff is None or self._conversations is None:
            return
        try:
            store = self._conversations()
            summaries = store.list_conversations()
        except Exception as exc:
            report.errors.append(f"conversations: {exc}")
            return
        for summary in summaries:
            updated = _parse_iso(summary.get("updated_at"))
            if updated is None or updated >= cutoff:
                continue
            conv_id = summary["conversation_id"]
            if dry_run:
                report.add("conversations", conv_id)
                continue
            try:
                store.delete(conv_id)
                report.add("conversations", conv_id)
            except Exception as exc:
                report.errors.append(f"conversation {conv_id}: {exc}")

    def _sweep_datasets(self, report: SweepReport, dry_run: bool) -> None:
        cutoff = self._cutoff(self.policy.dataset_days)
        if cutoff is None or self._datasets is None:
            return
        try:
            store = self._datasets()
            datasets = store.list()
        except Exception as exc:
            report.errors.append(f"datasets: {exc}")
            return
        for dataset in datasets:
            created = _parse_iso(dataset.get("created_at"))
            if created is None or created >= cutoff:
                continue
            dataset_id = dataset["dataset_id"]
            if dry_run:
                report.add("datasets", dataset_id)
                continue
            try:
                if store.delete(dataset_id):
                    report.add("datasets", dataset_id)
            except Exception as exc:
                report.errors.append(f"dataset {dataset_id}: {exc}")

    def _sweep_bundles(self, report: SweepReport, dry_run: bool) -> None:
        cutoff = self._cutoff(self.policy.bundle_days)
        if cutoff is None or self._registry is None:
            return
        try:
            registry = self._registry()
            bundles = registry.list_bundles()
        except Exception as exc:
            report.errors.append(f"bundles: {exc}")
            return
        for bundle in bundles:
            created = _parse_iso(getattr(bundle, "created_at", None))
            if created is None or created >= cutoff:
                continue
            bundle_id = getattr(bundle, "bundle_name", None) or getattr(bundle, "bundle_id", None)
            if not bundle_id:
                continue
            if dry_run:
                report.add("bundles", str(bundle_id))
                continue
            try:
                if registry.delete_bundle(str(bundle_id)):
                    report.add("bundles", str(bundle_id))
            except Exception as exc:
                report.errors.append(f"bundle {bundle_id}: {exc}")


def sweep_stale_workspaces(base_dir: Path, max_age_days: int) -> int:
    """Removes workspace directories whose newest file is older than the cutoff.

    Guards against orphans whose database row is already gone (a crash between the two deletes).
    """
    root = Path(base_dir)
    if not root.is_dir():
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    removed = 0
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        try:
            newest = max((p.stat().st_mtime for p in entry.rglob("*")), default=entry.stat().st_mtime)
        except OSError:
            continue
        if datetime.fromtimestamp(newest, tz=timezone.utc) >= cutoff:
            continue
        shutil.rmtree(entry, ignore_errors=True)
        removed += 1
    return removed
