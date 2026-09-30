# Business Entity Resolution

Starter project for inspecting, preprocessing, blocking, feature engineering, and supervised matching of the supplied business entity resolution dataset.

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
│   ├── inspect_dataset.py
│   └── preprocessing.py
├── tests/
│   ├── test_blocking.py
│   ├── test_features.py
│   ├── test_model.py
│   └── test_preprocessing.py
├── output/
│   └── candidate_pairs.tsv
├── requirements.txt
└── README.md
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

At the time this preprocessing step was added, both dataset folders contained only their placeholder files: **0 business records and 0 ground-truth rows are available locally**. No challenge data or labels have been generated. Once the complete TSV files are supplied, the preprocessing functions operate on every row they receive, including datasets with approximately 15,000 records.

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

The feature pipeline fits TF-IDF vocabularies on the three training source tables only. Training labels are joined from `train_ground_truth.tsv` after feature calculation; test ground truth is never read or used. Feature computation is batched (50,000 candidate pairs at a time by default), with the batch size configurable through `FeatureConfig`. Use the saved fitted pipeline for test inference:

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

The script loads the three training sources and ground truth, preprocesses source fields, generates blocking candidates, and labels candidate pairs from `train_ground_truth.tsv`. It splits by Source 1 entity ID so all candidate rows for one Source 1 record remain in one fold. The TF-IDF vectorizers are fitted only on source records belonging to training-fold entities; validation rows do not contribute text or labels to model fitting. The models are Logistic Regression and Random Forest, compared on validation precision, recall, F0.5, and confusion matrices.

Validation threshold tuning checks `0.30`, `0.35`, `0.40`, `0.45`, `0.50`, `0.55`, `0.60`, `0.65`, and `0.70`. The best validation F0.5 chooses the model and threshold; the test set is never used for this choice. F0.5 uses `(1.25 * precision * recall) / (0.25 * precision + recall)`.

Training saves `output/entity_match_model.pkl` (classifier, fitted TF-IDF feature engineer, configuration, and selected threshold), `output/feature_config.json`, `output/validation_metrics.json`, and train/validation feature TSVs for debugging. Reprint the saved model's validation report without retuning with:

```powershell
python -m src.evaluate_model
```

The evaluator requires labeled validation features. It does not read test labels or alter the saved threshold. Dataset inspection found no supplied TSV records in this workspace, so training has not been run on challenge data and no challenge performance numbers are available. Unit-test fixtures are for code verification only, not challenge records or labels.

Use a different validation fraction or fixed seed if needed:

```powershell
python -m src.train_model --validation-size 0.2 --random-state 42
```