# Documentation

Start with the [project README](../README.md) for an overview. The documents below go deeper; each states where its numbers come from.

| Document | Read it for |
|---|---|
| [architecture.md](architecture.md) | How the app and the research pipeline fit together: runtime flow, offline pipeline, module map and saved files, with the [architecture diagram](assets/architecture.svg). |
| [reproducibility.md](reproducibility.md) | Setting up from a fresh clone, the dataset and its checksum, what is committed versus rebuilt, determinism, and the CI workflow. |
| [methodology.md](methodology.md) | *How* it was done: data-quality audit and cleaning, train/test strategy, repeated cross-validation, calibration, the pre-declared selection protocol, thresholds, uncertainty, subgroups, explainability, reliability and robustness. |
| [evaluation.md](evaluation.md) | *What was measured*: every number, table by table, with the command that produces it. |
| [RESEARCH_VALIDATION.md](RESEARCH_VALIDATION.md) | The evidence chain: each claim traced to the artifact that supports it, plus the Phase 9 audit record (including the test-set leakage it found and fixed). |
| [model_card.md](model_card.md) | Intended and out-of-scope use, data, held-out performance, known limitations and potential bias, ethical considerations and non-clinical status. |

## Suggested reading paths

- **Using or demoing the app:** project README → [architecture.md](architecture.md) → [model_card.md](model_card.md).
- **Reviewing the science:** [methodology.md](methodology.md) → [evaluation.md](evaluation.md) → [RESEARCH_VALIDATION.md](RESEARCH_VALIDATION.md).
- **Rebuilding everything:** [reproducibility.md](reproducibility.md).

## Sources of truth

- Every number in these documents and in the app is read from, or traceable to, a generated artifact in [`../artifacts/`](../artifacts/). Each artifact carries a `provenance` block.
- The model-selection protocol is defined in `src/train_models.py` (the comment block above `ALPHA`) and described in [methodology.md](methodology.md).
- Images: [`assets/`](assets/) holds the banner, the architecture diagram and screenshots of the running app, taken with the synthetic example inputs.

All results describe one public dataset. None of them is clinical validation.
