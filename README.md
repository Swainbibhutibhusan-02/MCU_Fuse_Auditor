# AI-Based MCU Fuse Map Integrity Auditor

A Python and Streamlit application that audits compatible MCU fuse-map records using schema-based decoding, deterministic validation rules, and a machine-learning classifier.

> **Important:** This is a research/demo auditor using the project's supplied dataset and schema. Do not use it as the sole basis for programming, securing, or approving production hardware. Validate all rules against authoritative MCU documentation and independent device data before operational use.

## Features

- Decode hexadecimal fuse maps into schema-defined fields.
- Validate schema structure, field widths, bit positions, and overlapping bit ranges.
- Apply rule-based checks for documented configuration requirements.
- Predict potentially erroneous records with a trained Random Forest model.
- Audit compatible CSV batches in a Streamlit interface.
- Review and export audit results.
- Compare rule-based results with reference labels and held-out model metrics.

## Project structure

```text
MCU_Fuse_Auditor/
├── app.py
├── data/
│   ├── mcu_fuse_map_audit.csv
│   └── mcu_fuse_schema.csv
├── models/
│   ├── best_fuse_error_model.joblib
│   ├── feature_metadata.json
│   └── model_comparison.csv
├── reports/
│   ├── validation_results.csv
│   ├── ml_missed_records.csv
│   └── hybrid_missed_records.csv
├── src/
│   ├── eda.py
│   ├── fuse_decoder.py
│   ├── train_model.py
│   └── validator.py
├── requirements.txt
└── README.md
```

## Requirements

Python 3.10+ recommended, pip, and dependencies listed in `requirements.txt`.

## Setup

From the project directory:

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the application

```bash
streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`.

## Run individual checks

```bash
python src/fuse_decoder.py
python src/validator.py
python src/train_model.py
```

**Note:** Running `train_model.py` retrains the models and updates the saved model, metadata, and model-comparison report.

## Dataset and schema

The supplied labeled dataset contains 1,500 records: 1,094 labeled as no error and 406 labeled as errors.

The schema defines 10 decoded fields:

`BG_TRIM`, `HSI_TRIM`, `LDO_TRIM`, `TS_CAL1`, `FLASH_SIZE`, `PERIPH_EN`, `TEMP_GRADE`, `DEBUG_LOCK`, `RDP`, and `RESERVED`.

The validator uses schema-defined expected values for configuration fields including flash size, peripheral mask, temperature grade, read-out protection, debug lock, and reserved bits. Exact checks depend on the derivative and values in `data/mcu_fuse_schema.csv`.

Trim fields are decoded, but the supplied schema does not define authoritative trim acceptance limits. Statistical patterns must not be presented as official hardware limits.

## Machine-learning evaluation

The current training script uses a stratified 80/20 train/test split with `random_state=42`: 1,200 training records and 300 held-out test records.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Random Forest | 95.00% | 94.59% | 86.42% | 90.32% |
| Logistic Regression | 90.00% | 96.36% | 65.43% | 77.94% |

Random Forest was selected based on its F1-score. On the 300-record test split, it classified 215 clean records correctly, incorrectly flagged 4 clean records, detected 70 error records, and missed 11 error records.

These are results on this project's held-out split, not evidence of production readiness or performance on independent hardware. Independent, representative data is required for further evaluation.

## Rule-based validation results

Latest rule-validator run on the full 1,500-record labeled dataset:

| Outcome | Records |
|---|---:|
| True negatives | 1,094 |
| False positives | 0 |
| False negatives | 61 |
| True positives | 345 |
| Total rule-flagged | 345 |

On this dataset, rule-based precision is 100% and recall is approximately 84.98%. The 61 missed errors are associated with `trim_field_corrupt`, for which authoritative trim limits are not supplied. The reported 100% covered-category match rate applies only to categories covered by implemented rules; it does not mean all records or error types are fully validated.

## Compatible CSV input

The batch interface accepts CSVs containing compatible MCU fuse-map records and can map certain alternate column names. The file must still provide the relevant derivative and hexadecimal fuse-map values; an arbitrary unrelated CSV cannot be audited meaningfully.

Synthetic external fuse maps demonstrate the input workflow only. They do not prove that the values represent real hardware or that the auditor has been validated on independent devices.

## Reports and model artifacts

- `reports/validation_results.csv`: detailed rule-validation outcomes.
- `reports/ml_missed_records.csv`: records missed by the ML evaluation workflow.
- `reports/hybrid_missed_records.csv`: records missed by the hybrid evaluation workflow.
- `models/best_fuse_error_model.joblib`: selected trained model.
- `models/feature_metadata.json`: model feature metadata.
- `models/model_comparison.csv`: model comparison metrics.

Exact contents depend on the latest script run.

## Limitations and responsible use

1. Confirm every field position, width, expected value, and derivative-specific requirement against official MCU documentation.
2. Do not label trim values defective using statistical thresholds unless limits are validated and documented.
3. The model was trained and evaluated on the supplied dataset. Independent testing is needed to estimate generalization.
4. Some error categories are not fully covered by deterministic rules.
5. Treat ML predictions as triage signals; review results using trusted engineering procedures before making hardware decisions.
6. Do not assume a fuse map is safe or secure merely because the application does not flag it.

## Suggested future improvements

- Add automated unit tests for decoder, schema validation, and rule checks.
- Add documented, source-cited rules for each supported derivative.
- Test on independent datasets and report per-category metrics.
- Test malformed hex values, missing columns, unknown derivatives, and out-of-range bit patterns.
- Show rule coverage and model performance separately in the UI.
- Version schemas, datasets, and model artifacts.

## License

Add the license and ownership terms applicable to your project before public distribution.
