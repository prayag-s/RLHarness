"""RunBundle Registry & Sharing Engine for RL-Native Agent IDE v4.0.

Manages, catalogs, inspects, and imports self-contained .runbundle packages
and zip archives for policy sharing across teams and environments.
"""

from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Literal
import zipfile
import yaml
from pydantic import BaseModel, Field

from rl_ide.architect.models import AgentBlueprintSpec


class BundleMetadata(BaseModel):
    """Summarized metadata for a cataloged policy runbundle."""
    bundle_id: str
    name: str
    task_id: str
    adapter_name: str
    archive_type: Literal["dir", "zip"]
    bundle_path: str
    created_at: str
    onnx_size_bytes: int = 0
    dsl_terms_count: int = 0
    feature_count: int = 0
    baseline_quantiles_count: int = 0
    status: Literal["ready", "corrupt"] = "ready"
    error_message: str | None = None


class BundleDetails(BaseModel):
    """Deep inspection of a runbundle including configuration and baseline stats."""
    metadata: BundleMetadata
    blueprint_spec: dict[str, Any] = Field(default_factory=dict)
    task_config: dict[str, Any] = Field(default_factory=dict)
    reward_dsl: dict[str, Any] = Field(default_factory=dict)
    drift_baseline: dict[str, Any] = Field(default_factory=dict)
    onnx_summary: dict[str, Any] = Field(default_factory=dict)
    has_run_local_daemon: bool = False


class RunBundleRegistry:
    """Registry engine scanning, inspecting, and importing .runbundle archives."""

    def __init__(self, search_dirs: list[str | Path] | None = None):
        if search_dirs is None:
            self.search_dirs = [Path("exported_bundles").resolve()]
        else:
            self.search_dirs = [Path(d).resolve() for d in search_dirs]
        for d in self.search_dirs:
            d.mkdir(parents=True, exist_ok=True)

    def list_bundles(self) -> list[BundleMetadata]:
        """Scans all search directories for .runbundle folders and .runbundle.zip archives."""
        bundles: list[BundleMetadata] = []
        seen_ids: set[str] = set()

        for base_dir in self.search_dirs:
            if not base_dir.exists():
                continue

            # 1. Check directories
            for item in sorted(base_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if item.is_dir() and item.name.endswith(".runbundle"):
                    bundle_id = item.name.removesuffix(".runbundle")
                    if bundle_id in seen_ids:
                        continue
                    meta = self._parse_dir_metadata(item, bundle_id)
                    bundles.append(meta)
                    seen_ids.add(bundle_id)

                # 2. Check ZIP archives
                elif item.is_file() and (item.name.endswith(".runbundle.zip") or item.name.endswith(".zip")):
                    raw_id = item.name.replace(".runbundle.zip", "").replace(".zip", "")
                    bundle_id = f"{raw_id}_zip"
                    if bundle_id in seen_ids:
                        continue
                    meta = self._parse_zip_metadata(item, bundle_id)
                    bundles.append(meta)
                    seen_ids.add(bundle_id)

        return bundles

    def get_bundle(self, bundle_id: str) -> BundleDetails:
        """Retrieves detailed inspection for a specific bundle ID."""
        for meta in self.list_bundles():
            if meta.bundle_id == bundle_id:
                path = Path(meta.bundle_path)
                if meta.archive_type == "dir":
                    return self._inspect_dir(path, meta)
                else:
                    return self._inspect_zip(path, meta)

        raise FileNotFoundError(f"RunBundle with ID '{bundle_id}' not found in registry.")

    def import_bundle(
        self,
        archive_or_dir_path: str | Path,
        target_dir: str | Path | None = None,
        overwrite: bool = True,
    ) -> Path:
        """Imports an external bundle directory or zip file into the registry."""
        src = Path(archive_or_dir_path).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Bundle source not found: {src}")

        dest_base = Path(target_dir).resolve() if target_dir else self.search_dirs[0]
        dest_base.mkdir(parents=True, exist_ok=True)

        if src.is_file() and zipfile.is_zipfile(src):
            raw_name = src.name.replace(".runbundle.zip", "").replace(".zip", "")
            target_bundle = dest_base / f"{raw_name}.runbundle"
            if target_bundle.exists() and overwrite:
                shutil.rmtree(target_bundle)
            target_bundle.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(src, "r") as zf:
                for member in zf.infolist():
                    if member.is_dir():
                        continue
                    dest_file = target_bundle / Path(member.filename).name
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as source_f, open(dest_file, "wb") as target_f:
                        shutil.copyfileobj(source_f, target_f)

            installed_dir = target_bundle
        elif src.is_dir():
            target_bundle = dest_base / src.name
            if target_bundle != src:
                if target_bundle.exists() and overwrite:
                    shutil.rmtree(target_bundle)
                shutil.copytree(src, target_bundle)
            installed_dir = target_bundle
        else:
            raise ValueError(f"Unsupported bundle format at {src}")

        return installed_dir

    def delete_bundle(self, bundle_id: str) -> bool:
        """Deletes a bundle from the registry."""
        for meta in self.list_bundles():
            if meta.bundle_id == bundle_id:
                p = Path(meta.bundle_path)
                if p.is_dir():
                    shutil.rmtree(p)
                elif p.is_file():
                    p.unlink()
                return True
        return False

    def _parse_dir_metadata(self, bundle_dir: Path, bundle_id: str) -> BundleMetadata:
        try:
            mtime = datetime.fromtimestamp(bundle_dir.stat().st_mtime, tz=timezone.utc).isoformat()
            spec_file = bundle_dir / "blueprint_spec.json"
            config_file = bundle_dir / "task_config.yaml"
            onnx_file = bundle_dir / "model.onnx"
            baseline_file = bundle_dir / "drift_baseline.json"

            name = bundle_id
            task_id = bundle_id
            adapter_name = "SynthesizedGymEnv"
            feat_count = 0
            dsl_terms = 0

            if spec_file.exists():
                spec_data = json.loads(spec_file.read_text(encoding="utf-8"))
                name = spec_data.get("task_title", bundle_id)
                task_id = name
                feat_count = len(spec_data.get("observation_features", []))
                dsl_terms = len(spec_data.get("initial_reward_components", []))
            elif config_file.exists():
                config_data = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
                task_id = config_data.get("task_id", bundle_id)
                name = config_data.get("name", task_id)
                adapter_name = config_data.get("adapter_name", "Unknown")

            onnx_size = onnx_file.stat().st_size if onnx_file.exists() else 0
            quantiles_count = 0
            if baseline_file.exists():
                base_data = json.loads(baseline_file.read_text(encoding="utf-8"))
                if not feat_count:
                    feat_count = len(base_data.get("feature_names", []))
                quantiles_count = len(base_data.get("quantiles_per_feature", {}))

            return BundleMetadata(
                bundle_id=bundle_id,
                name=name,
                task_id=task_id,
                adapter_name=adapter_name,
                archive_type="dir",
                bundle_path=str(bundle_dir),
                created_at=mtime,
                onnx_size_bytes=onnx_size,
                dsl_terms_count=dsl_terms,
                feature_count=feat_count,
                baseline_quantiles_count=quantiles_count,
                status="ready",
            )
        except Exception as e:
            return BundleMetadata(
                bundle_id=bundle_id,
                name=bundle_dir.name,
                task_id="unknown",
                adapter_name="unknown",
                archive_type="dir",
                bundle_path=str(bundle_dir),
                created_at=datetime.now(timezone.utc).isoformat(),
                status="corrupt",
                error_message=str(e),
            )

    def _parse_zip_metadata(self, zip_path: Path, bundle_id: str) -> BundleMetadata:
        try:
            mtime = datetime.fromtimestamp(zip_path.stat().st_mtime, tz=timezone.utc).isoformat()
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = zf.namelist()
                spec_name = next((n for n in names if n.endswith("blueprint_spec.json")), None)
                onnx_name = next((n for n in names if n.endswith("model.onnx")), None)
                baseline_name = next((n for n in names if n.endswith("drift_baseline.json")), None)

                name = bundle_id
                feat_count = 0
                dsl_terms = 0

                if spec_name:
                    spec_data = json.loads(zf.read(spec_name).decode("utf-8"))
                    name = spec_data.get("task_title", bundle_id)
                    feat_count = len(spec_data.get("observation_features", []))
                    dsl_terms = len(spec_data.get("initial_reward_components", []))

                onnx_size = zf.getinfo(onnx_name).file_size if onnx_name else 0

                quantiles_count = 0
                if baseline_name:
                    base_data = json.loads(zf.read(baseline_name).decode("utf-8"))
                    quantiles_count = len(base_data.get("quantiles_per_feature", {}))

                return BundleMetadata(
                    bundle_id=bundle_id,
                    name=name,
                    task_id=name,
                    adapter_name="SynthesizedGymEnv",
                    archive_type="zip",
                    bundle_path=str(zip_path),
                    created_at=mtime,
                    onnx_size_bytes=onnx_size,
                    dsl_terms_count=dsl_terms,
                    feature_count=feat_count,
                    baseline_quantiles_count=quantiles_count,
                    status="ready",
                )
        except Exception as e:
            return BundleMetadata(
                bundle_id=bundle_id,
                name=zip_path.name,
                task_id="unknown",
                adapter_name="unknown",
                archive_type="zip",
                bundle_path=str(zip_path),
                created_at=datetime.now(timezone.utc).isoformat(),
                status="corrupt",
                error_message=str(e),
            )

    def _inspect_dir(self, bundle_dir: Path, meta: BundleMetadata) -> BundleDetails:
        spec_data = {}
        spec_f = bundle_dir / "blueprint_spec.json"
        if spec_f.exists():
            spec_data = json.loads(spec_f.read_text(encoding="utf-8"))

        base_data = {}
        base_f = bundle_dir / "drift_baseline.json"
        if base_f.exists():
            base_data = json.loads(base_f.read_text(encoding="utf-8"))

        has_daemon = (bundle_dir / "run_local.py").exists()

        return BundleDetails(
            metadata=meta,
            blueprint_spec=spec_data,
            drift_baseline=base_data,
            onnx_summary={"size_bytes": meta.onnx_size_bytes, "status": "ready"},
            has_run_local_daemon=has_daemon,
        )

    def _inspect_zip(self, zip_path: Path, meta: BundleMetadata) -> BundleDetails:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            spec_name = next((n for n in names if n.endswith("blueprint_spec.json")), None)
            spec_data = json.loads(zf.read(spec_name).decode("utf-8")) if spec_name else {}

            base_name = next((n for n in names if n.endswith("drift_baseline.json")), None)
            base_data = json.loads(zf.read(base_name).decode("utf-8")) if base_name else {}

            has_daemon = any(n.endswith("run_local.py") for n in names)

            return BundleDetails(
                metadata=meta,
                blueprint_spec=spec_data,
                drift_baseline=base_data,
                onnx_summary={"size_bytes": meta.onnx_size_bytes, "status": "ready"},
                has_run_local_daemon=has_daemon,
            )
