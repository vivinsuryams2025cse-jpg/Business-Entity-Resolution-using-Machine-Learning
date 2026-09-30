# Business Entity Resolution

Starter project for inspecting the supplied business entity resolution dataset. The initial script only loads and summarizes the data; it does not train a machine learning model.

## Project structure

```text
business_entity_resolution/
├── dataset/
│   ├── train/
│   └── test/
├── src/
│   └── inspect_dataset.py
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

The installed libraries are intended for the later matching and evaluation work. No model or matching workflow is implemented in this starter step.