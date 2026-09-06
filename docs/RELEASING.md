# Release and anonymity notes

Run `python scripts/build_release.py` from the repository root. The builder uses an explicit file allowlist and the current working files, including the supplementary inference modules. Check changes before building: an uncommitted file can still be part of the release when it is explicitly listed. New intended source files must be added to the allowlist.

The default outputs are `release_artifacts/costa-public.zip` and `release_artifacts/anonymous-submission.zip`, corresponding extracted directories, and a local audit record outside the archives. Both packages exclude every dataset directory and generated result, including synthetic examples. The generation and download scripts remain available. Only the public package retains the author information.

## Submission package checks

- Include source, configurations, dependency declarations, reproduction instructions, and tests.
- Remove author names, emails, institutional affiliation, the full paper title in the README, and author package metadata from the anonymous package. Preserve scientific method names and module imports so the code remains usable.
- Exclude Git history and remote configuration, OS metadata, caches, installed package metadata, logs, manuscript files, data, outputs, and nested archives.
- Reject symlinks and selected-file contents with detected credential or private-path markers. Scan anonymous text for direct identity markers as well. Automated scans supplement manual review and cannot identify every possible personal identifier.
- Use neutral archive paths, fixed archive timestamps, standard permissions, and no archive comments or extra fields. The extracted directory's filesystem timestamps are local; submit the generated ZIP directly.
- Run unit tests in a separate extraction when verifying a submission so generated test caches do not enter the final ZIP. Data downloads and research experiment runs are separate actions, not prerequisites for auditing the archive.

The anonymous archive must be reviewed against the specific venue's current submission policy. Removing explicit identities does not prevent someone from matching method names, algorithms, or unchanged code to an already public repository or paper. This package makes no guarantee against that kind of linkage.

## Data distribution boundary

The [MovieLens 10M license](https://files.grouplens.org/datasets/movielens/ml-10m-README.html) requires separate permission to redistribute its data. Use `bash scripts/download_movielens10m.sh` to obtain it directly from GroupLens and retain the upstream license locally. Dataset acknowledgments and third-party source links are retained in the anonymous documentation because they identify the data provider, not the submission authors.

The exclusion policy covers downloaded archives, raw records, processed interaction records, ID maps, embeddings, graphs, arrays, generated toy records, and outputs. It deliberately goes beyond distributing only a raw-data-free package. It does not assert that every aggregate statistic is prohibited by the dataset license.

Local raw data can remain in an ignored working directory. Empty directory placeholders and download instructions are not dataset payloads, but the release builder omits the complete `data/` tree to make the upload boundary unambiguous. Never upload the entire working folder as a ZIP.

## Git history and public hosting

`.gitignore` prevents normal staging of new files, but does not untrack an existing file or erase an earlier commit. Before publishing, inspect both the index (`git ls-files`) and reachable history (`git rev-list --objects --all`) for dataset paths. Examine any separately hosted releases, assets, LFS objects, branches, tags, and forks if they were used. Local inspection alone cannot certify all hosted copies.

If actual data files are tracked, preserve the local data and remove those files from the index before committing. If data were already published in history or hosted assets, plan a scoped cleanup of those copies; do not assume a deletion commit resolves redistribution. The release builder does not alter Git history, remotes, commits, or hosted assets, and does not push changes.

An anonymous review repository, if the venue requires one, must start from the anonymous archive without the original Git metadata and must use an appropriately anonymous account/hosting workflow. Uploading the archive from the original author account or attaching author-identifying submission text can reveal identity independently of the file contents.
