# Business Entity Resolution

Starter project for inspecting, preprocessing, blocking, feature engineering, and supervised matching of the supplied business entity resolution dataset.

## Problem statement

Given business records from three sources, generate plausible cross-source candidates and use the provided training ground truth to learn whether a Source 1 record and a Source 2/3 record represent the same business. For every test Source 1 record, return zero or more Source 2/3 IDs. Only the supplied dataset is used; there are no external lookups or business data sources.

## Project structure

```text
business_entity_resolution/
├── dataset/
│   ├── train/
│   └── test/
├── src/
│   ├── blocking.py
│   ├── features.py
│   ├── model.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── matching_engine.py
│   ├── predict.py
│   ├── inspect_dataset.py
│   └── preprocessing.py
├── tests/
│   ├── test_blocking.py
│   ├── test_features.py
│   ├── test_model.py
│   ├── test_matching_engine.py
│   └── test_preprocessing.py
├── output/
│   ├── candidate_pairs.tsv
│   └── matching_results.tsv
├── models/
├── requirements.txt
├── README.md
└── Documentation_template.md
```

Place the supplied TSV files in these locations:

| Folder | Files |
| --- | --- |
| `dataset/train/` | `train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv` |
| `dataset/test/` | `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv` |

The project uses only these provided files. It does not call external APIs, databases, geocoding services, or business data sources.

## Set up in VS Code

Open this project folder in VS Code, then run the following in its integrated terminal:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell prevents environment activation, select the `.venv` interpreter in VS Code and run `python -m pip install -r requirements.txt` in that interpreter.

## Inspect the dataset

From the project root, run:

```powershell
python src/inspect_dataset.py
```

The script reads every TSV with `sep="\t"` and reports record counts, column names, up to five sample rows, missing values, duplicate entity IDs in source files, source ID prefix checks, and a train/test summary. It lists any missing files and exits with a nonzero status until all expected files are present.

## Dataset availability

The current checkout contains **0 challenge records and 0 ground-truth rows**: both dataset folders contain only placeholders. No challenge data or labels have been generated. The pipeline is designed for the full supplied dataset, including approximately 15,000 records per source, once the files are placed in the listed folders.

## Preprocessing

`src/preprocessing.py` provides reusable functions for business names, addresses, and countries. `preprocess_business_records` returns a copy, retains all original columns and identifiers, and adds normalized columns. It accepts configurable source column names and does not drop or cap records. Normalization is local and deterministic; it uses no external services or business data.

Example:

| Field | Before | Normalized |
| --- | --- | --- |
| Business name | `North & West Intl., Inc.` | `north and west international inc` |
| Business address | `24-B, Main St., Suite # 5` | `24 b main street suite 5` |
| Country | `  fRaNcE  ` | `france` |

Use the functions on a loaded source table while retaining its source IDs and original fields:

```python
import pandas as pd

from src.preprocessing import preprocess_business_records

source1 = pd.read_csv("dataset/test/test_source1.tsv", sep="\t")
source1 = preprocess_business_records(source1)
```

Run the preprocessing unit tests from the project root:

```powershell
python -m unittest discover -s tests -v
```

scikit-learn, RapidFuzz, NumPy, and Matplotlib are listed in `requirements.txt` for preprocessing, matching, and evaluation.

## Candidate generation

Run candidate generation against the test sources from the project root:

```powershell
python -m src.blocking
```

Use `--split train` to run against training source tables, or `--output path/to/file.tsv` to choose another destination. The program reads Source 1, Source 2, and Source 3 TSVs with `sep="\t"` and writes `output/candidate_pairs.tsv` with exactly `source1_entity_id` and `candidate_entity_id` columns. Candidate IDs retain their original source prefixes so Source 2 and Source 3 remain distinguishable. The checked-in output currently contains only the required header because no source TSV data is available yet.

Blocking uses a union of exact normalized country, business-name tokens, character prefixes, address tokens, and country-plus-field keys. By default, blocks producing more than 5,000 cross-source pairs are skipped; each Source 1 record is limited to 100 candidates per target source. Both settings are configurable through `BlockingConfig` in `src/blocking.py`. The output is only a candidate set; it does not make final match decisions. Console metrics include all possible pairs, candidates, reduction ratio, generation time, per-source coverage, capped Source 1 rows, and oversized blocks skipped.

The unit tests use small hand-authored fixtures solely to test blocking behavior; they are not challenge records or ground-truth labels. A 15,000-by-15,000-by-15,000 scale fixture checks bounded generation. Run the tests with:

```powershell
python -m unittest discover -s tests -v
```

## Pair features

Generate training candidates first, keeping them separate from test candidates:

```powershell
python -m src.blocking --split train --output output/train_candidate_pairs.tsv
python -m src.features --split train --candidates output/train_candidate_pairs.tsv
```

This writes `output/train_features.tsv` for debugging and saves a fitted TF-IDF pipeline to `output/feature_pipeline.joblib`. The numeric columns include RapidFuzz name/address scores, normalized edit similarity, token Jaccard/common-token counts, name and address TF-IDF cosine similarities, country equality, length differences, and missing-value indicators. Candidate IDs are retained as row keys; `is_match` is the binary training target.

The feature pipeline fits TF-IDF vocabularies on training-fold records appearing in training-fold candidates only. Training labels are joined from `train_ground_truth.tsv` after candidate generation; test ground truth is never read or used. Feature computation is batched (50,000 candidate pairs at a time by default), with the batch size configurable through `FeatureConfig`. Use the saved fitted pipeline for test inference:

```powershell
python -m src.blocking --split test
python -m src.features --split test
```

Test features use the training TF-IDF vocabularies and are written to `output/test_features.tsv` without a label column. The feature engineer and batch size are configurable through `FeatureConfig` and `PairFeatureEngineer` in `src/features.py`. The command prints feature distributions and example rows. No real feature dataset can be generated until the supplied TSV files are added to `dataset/`.

## Supervised match model

Run the complete training and validation workflow from the project root:

```powershell
python -m src.train_model
```

The script loads the three training sources and ground truth, preprocesses source fields, generates blocking candidates, and labels candidate pairs from `train_ground_truth.tsv`. It splits by connected components over Source 1 and candidate IDs. Thus candidates for the same Source 1 and any shared Source 2/3 candidate stay together, preventing entity overlap across train and validation. Fold TF-IDF vocabularies use all candidate records in the training fold, including negative examples, and exclude validation-only entities. The models are Logistic Regression and Random Forest, compared on validation precision, recall, F0.5, and confusion matrices.

Validation threshold tuning checks `0.30`, `0.35`, `0.40`, `0.45`, `0.50`, `0.55`, `0.60`, `0.65`, and `0.70`. The best validation F0.5 chooses the model and threshold; the test set is never used for this choice. F0.5 uses `(1.25 * precision * recall) / (0.25 * precision + recall)`.

The model with the best validation F0.5 is selected; precision breaks an F0.5 tie. The threshold is independently selected from the requested grid using validation F0.5 only. After selection, the chosen classifier and TF-IDF transformer are refit on all labeled training candidates for deployment. The bundle retains its separate validation-fold estimator, so the saved validation report stays held out and honest. Training time is recorded in the validation report.

Training saves `models/entity_match_model.pkl` (deployment classifier, validation-fold estimator, training-fitted TF-IDF feature engineer, feature configuration, and selected threshold), `models/feature_config.json`, `output/validation_metrics.json`, and train/validation feature TSVs for debugging. Reprint the held-out validation report without retuning with:

```powershell
python -m src.evaluate_model
```

The evaluator requires labeled validation features and uses the saved validation-fold estimator. It does not read test labels or alter the saved threshold. Dataset inspection found no supplied TSV records in this workspace, so training has not been run on challenge data; no model has been selected and no challenge precision, recall, F0.5, timing, or candidate counts are available. Unit-test fixtures are for code verification only, not challenge records or labels.

Use a different validation fraction or fixed seed if needed. The default random state is 42. The candidate-component split is deterministic and is checked for both classes in each fold. The fold-fitted vectorizers exclude validation-only records. The final deployment refit uses training data only; test sources and any test labels are not used for fitting or threshold selection.

```powershell
python -m src.train_model --validation-size 0.2 --random-state 42
```

## Test-time matching

After training and validation select a model and threshold, run inference:

```powershell
python -m src.predict
```

The inference engine preprocesses all three test source files, generates and saves `output/candidate_pairs.tsv`, calculates features with the saved training-fitted TF-IDF transformer, scores candidates, and applies the selected validation threshold. It writes `output/matching_results.tsv` with exactly one row for every Source 1 ID. `matched_entity_ids` contains comma-separated, de-duplicated Source 2/3 IDs or is blank when no candidate passes the threshold. Every returned ID is checked against the test candidate sources, and predictions can only come from the generated candidate pairs. Console logging includes Source 1 count, candidate count, predicted-pair count, matched Source 1 count, singleton count, and inference time.

The inference inputs and model can be overridden with `--source1`, `--source2`, `--source3`, and `--model`; `--output-dir` changes both output locations. The checked-in `output/matching_results.tsv` currently contains its header only because the test sources and trained model are not present locally. The engine exits with a clear missing-input message rather than creating fictional results.

## Limitations and validation

The provided checkout currently lacks all seven required TSV files, so the actual challenge pipeline cannot be trained, inferred, or scored here. Blocking recall, selected model, selected threshold, validation/test metrics, and runtime on the real 15,000-record data are therefore unknown. A search found no `utils/validate_submission.py`; structural output invariants are covered by `tests/test_matching_engine.py`, but that external submission utility could not be run. `Documentation_template.md` provides a run record for dataset inventory, measured validation results, inference counts, and submission checks.

## Reproducibility

Use the project virtual environment and install the packages from `requirements.txt`; keep the supplied TSV files unchanged, and record the Git revision and resolved Python/package versions for each run. The default model random state and split seed are 42. Training writes the chosen threshold, both model comparison results, split counts, model artifacts, and feature configuration. Test inference reuses those saved artifacts and threshold. The full test suite is:

```powershell
python -m unittest discover -s tests -v
```