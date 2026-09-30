# Business Entity Resolution Run Record

Copy this template for each run using the supplied challenge data. Record values from generated files and logs only; do not estimate missing values.

## Run metadata

- Run date/time:
- Git commit:
- Python version:
- Dependency versions:
- Random state:
- Commands used:

## Dataset inventory

| File | Record count | Columns | Missing/invalid ID notes |
| --- | ---: | --- | --- |
| `train_source1.tsv` | | | |
| `train_source2.tsv` | | | |
| `train_source3.tsv` | | | |
| `train_ground_truth.tsv` | | | |
| `test_source1.tsv` | | | |
| `test_source2.tsv` | | | |
| `test_source3.tsv` | | | |

## Pipeline configuration

- Preprocessing columns and settings:
- Blocking settings and skipped oversized blocks:
- Total possible pairs:
- Candidate pairs and reduction ratio:
- Candidate recall against training ground truth, if measured:
- Feature columns and TF-IDF fitting scope:
- Validation split method and overlap checks:

## Validation model comparison

| Model | Threshold | Precision | Recall | F0.5 | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | | | | | | | | |
| Random Forest | | | | | | | | |

- Threshold values evaluated:
- Selected model and validation-based reason:
- Selected threshold and validation-based reason:
- Training time:
- Validation report path:

## Test inference

- Test Source 1 records:
- Candidate pairs:
- Candidate reduction ratio:
- Predicted matching pairs:
- Matched Source 1 entities:
- Singleton Source 1 predictions:
- Inference time:
- `candidate_pairs.tsv` path:
- `matching_results.tsv` path:

## Submission validation

- Validator path/version, or state that none was available:
- Exactly one output row per Source 1 test ID:
- Duplicate Source 1 rows/matched IDs:
- Every matched ID belongs to test Source 2/3:
- Every final match exists in the candidate table:
- TSV formatting/result:

## Limitations and notes

- Dataset/schema limitations:
- Blocking recall limitations:
- Errors/warnings:
- Reproduction notes: