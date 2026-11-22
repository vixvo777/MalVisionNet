"""
Build a comprehensive manifest for dataset audit and leakage prevention.
The manifest is the single source of truth for sample metadata, PE validation,
and duplicate detection before any training.
"""

import yaml
import hashlib
import csv
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import pefile
from tqdm import tqdm

from ..logger import logger


class ManifestBuilder:
    """
    Builds dataset manifests with provenance tracking and duplicate auditing.
    """

    def __init__(self, config_path: str = "configs/data.yaml"):
        self.config = self._load_config(config_path)
        self.raw_dir = Path(self.config["storage"]["raw_dir"])
        self.manifest_dir = Path(self.config["storage"]["manifest_dir"])
        self.interim_dir = Path(self.config["storage"]["interim_dir"])
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

        self.parser_version = "pefile_v2023.2.7"
        self.duplicate_groups = {
            "exact_hash": {},  # SHA-256 -> [sample_ids]
            "rendered_hash": {},  # Tensor hash -> [sample_ids]
            "near_duplicates": {}  # Perceptual hash -> [sample_ids]
        }

    def _load_config(self, config_path: str) -> Dict:
        """Load YAML configuration."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _validate_pe(self, file_path: Path) -> Tuple[bool, Dict, Optional[str]]:
        """
        Validate PE file structure.
        Returns (is_valid, pe_metadata, error_message)
        """
        try:
            pe = pefile.PE(file_path)
            metadata = {
                "machine": pe.FILE_HEADER.Machine,
                "subsystem": pe.OPTIONAL_HEADER.Subsystem,
                "section_count": pe.FILE_HEADER.NumberOfSections,
                "import_count": len(pe.imports) if pe.imports else 0,
                "export_count": len(pe.exports) if pe.exports else 0,
                "size_of_image": pe.OPTIONAL_HEADER.SizeOfImage,
                "size_of_headers": pe.OPTIONAL_HEADER.SizeOfHeaders,
                "entry_point": pe.OPTIONAL_HEADER.AddressOfEntryPoint,
            }
            return True, metadata, None
        except pefile.PEFormatError as e:
            return False, {}, f"PEFormatError: {e}"
        except Exception as e:
            return False, {}, f"Validation error: {e}"

    def _compute_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of raw bytes."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _group_by_source(self, samples: List[Dict]) -> Dict:
        """
        Group samples by source_dataset and source_project for audit.
        """
        groups = {}
        for sample in samples:
            dataset = sample["source_dataset"]
            project = sample.get("source_project", "unknown")
            key = f"{dataset}/{project}"
            groups.setdefault(key, []).append(sample)
        return groups

    def build_manifest(
        self,
        source_dirs: Optional[List[str]] = None,
        force_rebuild: bool = False,
    ) -> Path:
        """
        Build manifest from raw files.
        """
        if source_dirs is None:
            source_dirs = [
                self.raw_dir / "bodmas",
                self.raw_dir / "assemblage",
            ]

        # Check if manifest already exists
        manifest_path = self.manifest_dir / "malvision_manifest.csv"
        if manifest_path.exists() and not force_rebuild:
            logger.info(f"Manifest already exists at {manifest_path}")
            return manifest_path

        samples = []
        rejection_log = []

        for source_dir in source_dirs:
            if not source_dir.exists():
                logger.warning(f"Source directory not found: {source_dir}")
                continue

            # Iterate through source files
            for file_path in tqdm(list(source_dir.iterdir()), desc="Processing"):
                if not file_path.is_file():
                    continue

                # Determine source type
                if "bodmas" in str(source_dir):
                    label = "malicious"
                    family = file_path.stem  # Use filename or extract from metadata if available
                    source_dataset = "BODMAS"
                    source_project = None
                else:  # Assemblage
                    label = "benign"
                    source_dataset = "Assemblage"
                    # Try to extract project info from filename or metadata
                    source_project = file_path.stem.split("_")[0] if "_" in file_path.stem else "unknown"

                # Compute hash and validate PE
                file_hash = self._compute_hash(file_path)
                is_valid, pe_metadata, error = self._validate_pe(file_path)

                # Skip invalid PEs
                if not is_valid:
                    rejection_log.append({
                        "file": str(file_path),
                        "hash": file_hash,
                        "reason": error,
                        "timestamp": datetime.now().isoformat(),
                    })
                    continue

                sample = {
                    "sample_id": f"{label}_{file_hash[:8]}",
                    "sha256": file_hash,
                    "label": label,
                    "family": family if label == "malicious" else None,
                    "file_size": file_path.stat().st_size,
                    "source_dataset": source_dataset,
                    "source_project": source_project,
                    "pe_valid": is_valid,
                    "pe_machine": pe_metadata.get("machine"),
                    "pe_subsystem": pe_metadata.get("subsystem"),
                    "parser_version": self.parser_version,
                    "timestamp": datetime.now().isoformat(),
                    "representation_config_id": "rep_v1_grayscale_256_224area",  # Default placeholder
                }
                samples.append(sample)

                # Track duplicates (to be used later with rendered hashes)
                self.duplicate_groups["exact_hash"].setdefault(file_hash, []).append(
                    sample["sample_id"]
                )

        # Write samples to manifest
        with open(manifest_path, "w", newline="") as csvfile:
            fieldnames = list(samples[0].keys()) if samples else []
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(samples)

        # Write rejection log if any
        if rejection_log:
            rejection_path = self.interim_dir / "rejected_files.csv"
            rejection_path.parent.mkdir(parents=True, exist_ok=True)
            with open(rejection_path, "w", newline="") as csvfile:
                fieldnames = list(rejection_log[0].keys()) if rejection_log else []
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rejection_log)
            logger.warning(f"Rejected {len(rejection_log)} files, logged to {rejection_path}")

        # Create summary report
        summary = {
            "total_samples": len(samples),
            "malicious": len([s for s in samples if s["label"] == "malicious"]),
            "benign": len([s for s in samples if s["label"] == "benign"]),
            "valid_pe": len([s for s in samples if s["pe_valid"]]),
            "invalid_pe": len(rejection_log),
            "source_groups": self._group_by_source(samples),
        }

        summary_path = self.interim_dir / "manifest_summary.json"
        with open(summary_path, "w") as f:
            yaml.dump(summary, f, default_flow_style=False)

        logger.info(
            f"Built manifest with {summary['total_samples']} samples. "
            f"Summary saved to {summary_path}"
        )

        return manifest_path

    def deduplicate_samples(
        self, manifest_path: Path
    ) -> Tuple[Path, Dict[str, List[str]]]:
        """
        Identify and group exact duplicates and near duplicates.
        Returns updated manifest and duplicate groups.
        """
        # Read manifest
        samples = []
        with open(manifest_path, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                samples.append(row)

        # Mark exact duplicates (same hash)
        exact_duplicate_groups = {}
        for sample in samples:
            hash_val = sample["sha256"]
            if hash_val in self.duplicate_groups["exact_hash"]:
                group_id = hash_val[:16]  # Short ID for readability
                sample["duplicate_group"] = group_id
                if group_id not in exact_duplicate_groups:
                    exact_duplicate_groups[group_id] = []
                exact_duplicate_groups[group_id].append(sample["sample_id"])

        # For near duplicates, we would need image hashing (e.g., imagehash)
        # This is a placeholder for the actual implementation

        # Write updated manifest with duplicate group info
        with open(manifest_path, "w", newline="") as csvfile:
            fieldnames = list(samples[0].keys()) if samples else []
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(samples)

        # Update duplicate groups file for reference
        groups_path = self.interim_dir / "duplicate_groups.json"
        with open(groups_path, "w") as f:
            json.dump(
                {
                    "exact_duplicate_groups": exact_duplicate_groups,
                    "near_duplicate_groups": self.duplicate_groups["near_duplicates"],
                },
                f,
                indent=2,
            )

        logger.info(f"Deduplication complete. Groups saved to {groups_path}")

        return manifest_path, exact_duplicate_groups


if __name__ == "__main__":
    # Example usage
    builder = ManifestBuilder()
    manifest_path = builder.build_manifest(force_rebuild=True)
    print(f"Manifest built at: {manifest_path}")