"""Release checks use synthetic source trees; no dataset download is required."""

import importlib.util
import json
from pathlib import Path
import stat
import zipfile

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_release.py"
_SPEC = importlib.util.spec_from_file_location("build_release", _SCRIPT)
release = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(release)


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    names = ("README.md", "pyproject.toml", "src/costa/__init__.py", "scripts/example.sh")
    monkeypatch.setattr(release, "SOURCE_FILES", names)
    author = "Example" + " Researcher"
    email = "researcher" + "@" + "example.invalid"
    (root / "README.md").write_text(
        "# Study\n\n**Full paper title**\n\nAuthors: " + author + "\nAffiliation line\n\n"
        "This repository implements COSTA and COSTA-IC.\n\n"
        "The repository is prepared as a public release.\n\n"
        "study" + "_public_code/\n\n"
        "<!-- public-release-only:start -->\nRelease maintenance instructions\n"
        "<!-- public-release-only:end -->\n", encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        '[project]\nname = "study"\nversion = "0.1"\nauthors = [\n'
        '  {name = "' + author + '", email = "' + email + '"}\n]\n'
        'maintainers = [{name = "' + author + '"}]\ndependencies = ["numpy"]\n\n'
        '[project.urls]\nHomepage = "https://' + 'github.com/' + 'example/study"\n\n'
        '[tool.setuptools.packages.find]\nwhere = ["src"]\n', encoding="utf-8",
    )
    (root / "src/costa").mkdir(parents=True)
    (root / "src/costa/__init__.py").write_text(
        'METHODS = ["COSTA", "COSTA_IC"]\n'
        'MARKER = "<!-- public-release-only:start --> keep <!-- public-release-only:end -->"\n',
        encoding="utf-8",
    )
    (root / "scripts").mkdir()
    (root / "scripts/example.sh").write_text('#!/bin/sh\ntrue\n', encoding="utf-8")
    return root


def test_data_caches_secrets_and_unreviewed_source_are_excluded(source, tmp_path):
    excluded = (
        "data/raw/ratings.dat", "data/processed/interactions.csv", "data/sample/toy.csv",
        "outputs/results.csv", ".git/config", ".env", ".DS_Store",
        "src/costa/__pycache__/cache.pyc", "src/costa_design.egg-info/PKG-INFO",
        "scripts/unreviewed.py", "local.key",
    )
    for name in excluded:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"must not be distributed\x00")
    output = tmp_path / "release"
    audit = release.build_release(source, output)
    assert audit["status"] == "passed"
    for kind, label in release.ARTIFACT_NAMES.items():
        with zipfile.ZipFile(output / f"{label}.zip") as archive:
            names = {name.removeprefix(label + "/") for name in archive.namelist()}
            assert names == set(release.SOURCE_FILES) | {release.MANIFEST_NAME}
        assert all(not (output / label / name).exists() for name in excluded)
        manifest = json.loads((output / label / release.MANIFEST_NAME).read_text())
        assert manifest["files"] == sorted(names)
        assert manifest["file_count"] == len(names)


def test_anonymous_metadata_removed_and_public_authorship_preserved(source, tmp_path):
    output = tmp_path / "release"
    release.build_release(source, output)
    public = output / release.ARTIFACT_NAMES["public"]
    anonymous = output / release.ARTIFACT_NAMES["anonymous"]
    assert (public / "README.md").read_text() == (source / "README.md").read_text()
    assert (public / "pyproject.toml").read_text() == (source / "pyproject.toml").read_text()
    readme = (anonymous / "README.md").read_text()
    metadata = (anonymous / "pyproject.toml").read_text()
    for omitted in ("Authors:", "Affiliation line", "Full paper title", "Release maintenance instructions"):
        assert omitted not in readme
    assert readme.startswith("# Anonymous supplementary code")
    assert "COSTA and COSTA-IC" in readme
    assert "anonymous-submission/" in readme
    assert "authors" not in metadata and "maintainers" not in metadata
    assert "project.urls" not in metadata
    assert 'dependencies = ["numpy"]' in metadata
    assert '[tool.setuptools.packages.find]' in metadata
    assert (anonymous / "src/costa/__init__.py").read_bytes() == (source / "src/costa/__init__.py").read_bytes()


def test_zip_is_deterministic_and_has_normalized_metadata(source, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    release.build_release(source, first)
    (source / "scripts/example.sh").chmod(0o600)
    release.build_release(source, second)
    for label in release.ARTIFACT_NAMES.values():
        assert (first / f"{label}.zip").read_bytes() == (second / f"{label}.zip").read_bytes()
        with zipfile.ZipFile(first / f"{label}.zip") as archive:
            assert archive.comment == b""
            assert archive.namelist() == sorted(archive.namelist())
            for info in archive.infolist():
                assert info.date_time == release.ZIP_TIME
                assert info.extra == info.comment == b""
                assert info.create_system == 3
                assert stat.S_ISREG(info.external_attr >> 16)
                assert stat.S_IMODE(info.external_attr >> 16) == (0o755 if info.filename.endswith(".sh") else 0o644)
                assert archive.read(info) == (first / info.filename).read_bytes()


@pytest.mark.parametrize("component", ["file", "parent", "output"])
def test_symlinks_refused(source, tmp_path, component):
    output = tmp_path / "release"
    if component == "file":
        target = source / "src/costa/__init__.py"
        external = tmp_path / "external.py"
        external.write_text("external = True\n")
        target.unlink()
        target.symlink_to(external)
    elif component == "parent":
        original = source / "src"
        moved = tmp_path / "moved-src"
        original.rename(moved)
        original.symlink_to(moved, target_is_directory=True)
    else:
        external = tmp_path / "external-output"
        external.mkdir()
        output.symlink_to(external, target_is_directory=True)
    with pytest.raises(release.ReleaseError, match="Symlinks"):
        release.build_release(source, output)


@pytest.mark.parametrize("violation", ["secret", "home", "author", "affiliation", "title", "email", "github", "binary"])
def test_sensitive_selected_content_fails_closed_without_echoing_values(source, tmp_path, violation):
    sensitive = {
        "secret": "API" + "_KEY = '" + "fake-credential-value" + "'\n",
        "home": "# " + "/" + "Users" + "/private-person/work\n",
        "author": "# " + "Example" + " Researcher\n",
        "affiliation": "# Affiliation line\n",
        "title": "# Full paper title\n",
        "email": "# " + "someone" + "@" + "example.invalid\n",
        "github": "# https://" + "github.com/" + "private-owner/repository\n",
        "binary": "\x00binary",
    }[violation]
    (source / "src/costa/__init__.py").write_text(sensitive)
    with pytest.raises(release.ReleaseError) as caught:
        release.build_release(source, tmp_path / "release")
    assert sensitive.strip() not in str(caught.value)
    assert not (tmp_path / "release").exists()


def test_existing_outputs_need_explicit_overwrite_and_recognized_manifest(source, tmp_path):
    output = tmp_path / "release"
    release.build_release(source, output)
    archive = output / (release.ARTIFACT_NAMES["anonymous"] + ".zip")
    original = archive.read_bytes()
    with pytest.raises(release.ReleaseError, match="already exists"):
        release.build_release(source, output)
    assert archive.read_bytes() == original
    release.build_release(source, output, overwrite=True)
    assert archive.read_bytes() == original
    manifest = output / release.ARTIFACT_NAMES["public"] / release.MANIFEST_NAME
    manifest.unlink()
    with pytest.raises(release.ReleaseError, match="unrecognized"):
        release.build_release(source, output, overwrite=True)
    assert archive.read_bytes() == original


def test_allowlist_cannot_be_expanded_into_data(source, tmp_path, monkeypatch):
    monkeypatch.setattr(release, "SOURCE_FILES", (*release.SOURCE_FILES, "data/raw/ratings.dat"))
    with pytest.raises(release.ReleaseError, match="Data and generated"):
        release.build_release(source, tmp_path / "release")


def test_overwrite_protects_files_added_after_build(source, tmp_path):
    output = tmp_path / "release"
    release.build_release(source, output)
    extra = output / release.ARTIFACT_NAMES["public"] / "local-notes.txt"
    extra.write_text("user content")
    with pytest.raises(release.ReleaseError, match="unrecognized"):
        release.build_release(source, output, overwrite=True)
    assert extra.read_text() == "user content"
