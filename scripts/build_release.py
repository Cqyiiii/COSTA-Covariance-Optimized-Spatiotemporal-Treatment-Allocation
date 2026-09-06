#!/usr/bin/env python3
"""Build reviewed, data-free public and anonymous source archives (Python 3.10+)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import zipfile


# Review additions explicitly: discovering files recursively could publish a
# newly created credential file, dataset, notebook output, or local artifact.
SOURCE_FILES = (
    ".gitignore", "README.md", "pyproject.toml", "requirements.txt", "docs/RELEASING.md",
    "docs/assets/costa-overview.svg",
    "configs/ablation_grid.yaml", "configs/base.yaml",
    "configs/clt_ic_component_ablation.yaml", "configs/clt_validation.yaml",
    "configs/inference_revised26.yaml", "configs/movielens10m_10k.yaml",
    "configs/movielens_small_9k.yaml", "configs/retailrocket_10k.yaml",
    "configs/sample.yaml", "configs/sample_debug.yaml",
    "scripts/build_release.py", "scripts/download_movielens10m.sh",
    "scripts/download_movielens_small.sh", "scripts/download_retailrocket.sh",
    "scripts/make_sample_data.py", "scripts/plot_clt_designaware_variance.py",
    "scripts/plot_clt_ic_component_ablation.py", "scripts/plot_clt_validation.py",
    "scripts/plot_inference_validation_revised26.py",
    "scripts/prepare_ablation_grid.py", "scripts/prepare_data.py",
    "scripts/run_ablation_grid.py", "scripts/run_certified_boundary_experiment.py",
    "scripts/run_clt_designaware_variance.py",
    "scripts/run_clt_ic_component_ablation.py", "scripts/run_clt_validation.py",
    "scripts/run_extra_cluster_costa_grid.py", "scripts/run_extra_costa_ic_grid.py",
    "scripts/run_inference_validation_revised26.py",
    "scripts/run_required_inference_revised26.sh", "scripts/run_simulation.py",
    "src/costa/__init__.py", "src/costa/ablation.py", "src/costa/clustering.py",
    "src/costa/config.py", "src/costa/data.py", "src/costa/designs.py",
    "src/costa/embeddings.py", "src/costa/estimators.py",
    "src/costa/gaussian_copula.py", "src/costa/graph.py", "src/costa/inference.py",
    "src/costa/inference_revised26.py", "src/costa/naming.py",
    "src/costa/outcomes.py", "src/costa/paths.py", "src/costa/result_index.py",
    "tests/test_build_release.py", "tests/test_inference_revised26.py",
    "tests/test_plot_inference_revised26.py", "tests/test_smoke.py",
)
ARTIFACT_NAMES = {"public": "costa-public", "anonymous": "anonymous-submission"}
MANIFEST_NAME = "RELEASE_MANIFEST.json"
SCHEMA = "data-free-source-release-v1"
ZIP_TIME = (1980, 1, 1, 0, 0, 0)


class ReleaseError(ValueError):
    """A release cannot be built safely from the selected source."""


def no_symlinks(path: Path) -> Path:
    """Check lexical components before resolve() can hide a symlink."""
    absolute = Path(os.path.abspath(path))
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise ReleaseError("Symlinks are forbidden in release input/output paths")
    return absolute


def _metadata(text: str) -> tuple[str, list[str]]:
    """Remove PEP 621 personal metadata without requiring Python 3.11 tomllib."""
    sections = re.split(r"(?m)(^\[[^\n]+\][ \t]*\n)", text)
    result = [sections[0]]
    identities: list[str] = []
    for index in range(1, len(sections), 2):
        header, body = sections[index:index + 2]
        if header.strip() == "[project.urls]":
            continue
        if header.strip() == "[project]":
            pattern = r"(?ms)^(?:authors|maintainers)\s*=.*?(?=^[A-Za-z_][\w.-]*\s*=|\Z)"
            for match in re.finditer(pattern, body):
                identities.extend(re.findall(r"\b(?:name|email)\s*=\s*[\"']([^\"']+)", match[0]))
            body = re.sub(pattern, "", body)
        result.extend((header, body))
    return "".join(result), identities


def anonymize(name: str, text: str) -> str:
    if name.endswith(".md"):
        text = re.sub(
            r"(?s)<!-- public-release-only:start -->.*?<!-- public-release-only:end -->",
            "", text,
        )
    if name == "README.md":
        # The maintained README introduces the scientific overview with this
        # paragraph; everything before it is title/byline/affiliation metadata.
        overview = re.search(r"(?m)^This repository\b", text)
        if overview is None:
            raise ReleaseError("README overview marker missing; review anonymous introduction")
        text = "# Anonymous supplementary code\n\n" + text[overview.start():]
        text = text.replace("prepared as a public release", "prepared as anonymous supplementary code")
        text = text.replace("This public code package", "This supplementary code package")
    elif name == "pyproject.toml":
        text, _ = _metadata(text)
    # A checkout directory is provenance, while scientific method labels remain.
    text = re.sub(r"\b[A-Za-z0-9_-]+_public_code\b", "anonymous-submission", text)
    return text


def audit_text(name: str, text: str, anonymous: bool, identities: list[str]) -> None:
    checks = {
        "private home path": r"/(?:Users|home)/[A-Za-z0-9_.-]+",
        "private key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        "credential assignment": (
            r"(?im)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|"
            r"password|passwd|secret[_-]?key)\b[\"']?\s*[:=]\s*[\"'][^\"'\r\n]{4,}[\"']"
        ),
    }
    if anonymous:
        checks.update({
            "email address": r"\b[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            "GitHub repository URL": r"(?:https?://|git@)(?:www\.)?github\.com[/:][\w.-]+/[\w.-]+",
        })
    for label, pattern in checks.items():
        if re.search(pattern, text):
            raise ReleaseError(f"Release audit rejected {name}: {label} (value withheld)")
    if anonymous and any(value.casefold() in text.casefold() for value in identities if value):
        raise ReleaseError(f"Release audit rejected {name}: source author metadata (value withheld)")


def collect_source(source: Path) -> dict[str, str]:
    selected: dict[str, str] = {}
    for name in SOURCE_FILES:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ReleaseError("Unsafe allowlist path")
        if any(part.startswith(".") for part in relative.parts) and name != ".gitignore":
            raise ReleaseError("Hidden paths cannot be added to the release allowlist")
        if relative.parts[0] in {"data", "outputs", "release_artifacts"}:
            raise ReleaseError("Data and generated artifacts cannot be added to the allowlist")
        if any(part == "__pycache__" or part.endswith(".egg-info") for part in relative.parts):
            raise ReleaseError("Cache and installation metadata cannot be added to the allowlist")
        if relative.suffix not in {".py", ".sh", ".md", ".yaml", ".toml", ".txt", ".svg"} and name != ".gitignore":
            raise ReleaseError("Only reviewed source/documentation file types are allowed")
        path = no_symlinks(source / name)
        if not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
            raise ReleaseError(f"Required regular source file missing: {name}")
        content = path.read_bytes()
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ReleaseError(f"Binary or non-UTF-8 file rejected: {name}") from exc
        if any(ord(character) < 32 and character not in "\n\r\t" for character in text):
            raise ReleaseError(f"Binary control characters rejected: {name}")
        selected[name] = text.replace("\r\n", "\n")
    return selected


def _manifest(kind: str, names: list[str]) -> bytes:
    return (json.dumps({
        "schema": SCHEMA,
        "release_kind": kind,
        "data_included": False,
        "git_history_included": False,
        "file_count": len(names) + 1,
        "files": sorted([*names, MANIFEST_NAME]),
    }, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _check_existing(destination: Path, kind: str, overwrite: bool, source: Path) -> None:
    no_symlinks(destination)
    if not destination.exists():
        return
    if not overwrite:
        raise ReleaseError(f"Destination already exists: {destination.name}; use --overwrite to replace a prior release")
    if source == destination or destination in source.parents:
        raise ReleaseError("Refusing to overwrite the source directory or an ancestor")
    try:
        if destination.is_dir():
            manifest_path = no_symlinks(destination / MANIFEST_NAME)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            current_files = set()
            for path in destination.rglob("*"):
                no_symlinks(path)
                if path.is_file():
                    current_files.add(path.relative_to(destination).as_posix())
            if current_files != set(manifest.get("files", ())):
                raise ValueError("destination contains added or missing files")
        else:
            with zipfile.ZipFile(destination) as archive:
                manifest = json.loads(archive.read(f"{ARTIFACT_NAMES[kind]}/{MANIFEST_NAME}"))
        if manifest.get("schema") != SCHEMA or manifest.get("release_kind") != kind:
            raise ValueError("unrecognized manifest")
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise ReleaseError("Refusing to overwrite an unrecognized destination") from exc


def build_release(source: Path, output: Path, variant: str = "both", overwrite: bool = False) -> dict:
    source, output = no_symlinks(source), no_symlinks(output)
    if variant not in {"both", *ARTIFACT_NAMES}:
        raise ReleaseError("Unknown release variant")
    kinds = list(ARTIFACT_NAMES) if variant == "both" else [variant]
    originals = collect_source(source)
    _, identities = _metadata(originals["pyproject.toml"])
    introduction = re.split(r"(?m)^This repository\b", originals["README.md"], maxsplit=1)[0]
    identities.extend(
        line.strip().replace("**", "")
        for line in introduction.splitlines() if line.strip() and not line.startswith("#")
    )
    prepared: dict[str, dict[str, bytes]] = {}
    for kind in kinds:
        files = {}
        for name, original in originals.items():
            content = anonymize(name, original) if kind == "anonymous" else original
            audit_text(name, content, kind == "anonymous", identities)
            files[name] = content.encode("utf-8")
        files[MANIFEST_NAME] = _manifest(kind, list(files))
        prepared[kind] = files
        for destination in (output / ARTIFACT_NAMES[kind], output / f"{ARTIFACT_NAMES[kind]}.zip"):
            _check_existing(destination, kind, overwrite, source)
    audit_path = no_symlinks(output / "build_audit.json")
    if audit_path.exists():
        try:
            prior_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ReleaseError("Refusing to replace an unrecognized audit file") from exc
        if prior_audit.get("schema") != SCHEMA:
            raise ReleaseError("Refusing to replace an unrecognized audit file")
    output.mkdir(parents=True, exist_ok=True)
    audit = {"schema": SCHEMA, "status": "passed", "releases": {}}
    with tempfile.TemporaryDirectory(prefix=".release-build-", dir=output) as temporary:
        staging = Path(temporary)
        for kind, files in prepared.items():
            label = ARTIFACT_NAMES[kind]
            directory = staging / label
            archive_path = staging / f"{label}.zip"
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
                for name, content in sorted(files.items()):
                    target = directory / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
                    mode = 0o755 if name.endswith(".sh") else 0o644
                    target.chmod(mode)
                    info = zipfile.ZipInfo(f"{label}/{name}", date_time=ZIP_TIME)
                    info.create_system = 3
                    info.external_attr = (stat.S_IFREG | mode) << 16
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.extra = b""
                    info.comment = b""
                    archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
            # Validate archive bytes against the exact emitted directory content.
            with zipfile.ZipFile(archive_path) as archive:
                if archive.testzip() is not None:
                    raise ReleaseError("Archive integrity verification failed")
                for name, content in files.items():
                    if archive.read(f"{label}/{name}") != content or (directory / name).read_bytes() != content:
                        raise ReleaseError("Archive/directory content mismatch")
            audit["releases"][kind] = {"directory": label, "archive": f"{label}.zip", "file_count": len(files),
                                         "data_files": 0, "identity_audit": "passed" if kind == "anonymous" else "public attribution retained"}
        for kind in kinds:
            label = ARTIFACT_NAMES[kind]
            for name in (label, f"{label}.zip"):
                destination = output / name
                if destination.is_dir():
                    shutil.rmtree(destination)
                elif destination.exists():
                    destination.unlink()
                (staging / name).replace(destination)
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("both", "public", "anonymous"), default="both")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Replace recognized previously built releases")
    args = parser.parse_args()
    source = Path(__file__).absolute().parents[1]
    output = args.output_dir if args.output_dir is not None else source / "release_artifacts"
    try:
        audit = build_release(source, output, args.variant, args.overwrite)
    except (ReleaseError, OSError) as exc:
        parser.exit(1, f"Release build failed: {exc}\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
