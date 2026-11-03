# Local benchmark manifests

Run `python scripts/rebuild_suites.py` using the repository's `.venv` after placing the original train data in `../train`.

This recreates the 10-question smoke suite, 90-question development suite, 200-question random validation suite, 12-question normalization pilot, 100-question SQL validation and its first-20 paired subset, 40-question local-training evaluation and 40-question verification evaluation. Exclusion manifests cover 290, 302, 402, 442 and 482 viewed questions. Use `seen482.json` for the next fresh sample. JSON files contain dataset identifiers and are excluded from Git. The selection recipes and seeds are in the script; their generated IDs were checked against the actual evaluation manifests.

Development selection uses reference types to cover formats. The 200-question validation sample is uniform random from the remaining questions and does not use reference types or answers for sampling.
