# Business Entity Resolution

Starter project for inspecting and preprocessing the supplied business entity resolution dataset. The project does not train a machine learning model yet.

## Project structure

```text
business_entity_resolution/
├── dataset/
│   ├── train/
│   └── test/
├── src/
│   ├── inspect_dataset.py
│   └── preprocessing.py
├── tests/
│   └── test_preprocessing.py
├── output/
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

The installed scikit-learn, RapidFuzz, NumPy, and Matplotlib libraries are available for later matching and evaluation work. No model or matching workflow is implemented yet.