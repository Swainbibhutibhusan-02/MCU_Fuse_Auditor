
from pathlib import Path

import pandas as pd

from fuse_decoder import decode_fuse_map, load_schema


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

AUDIT_FILE = DATA_DIR / "mcu_fuse_map_audit.csv"

# Map schema fields to the corresponding dataset error labels.
ERROR_LABELS = {
    "FLASH_SIZE": "wrong_flash_size",
    "PERIPH_EN": "periph_mask_mismatch",
    "TEMP_GRADE": "wrong_temp_grade",
    "DEBUG_LOCK": "missing_debug_lock",
    "RDP": "rdp_missing_automotive",
    "RESERVED": "reserved_bit_set",
}


def expected_value(schema, derivative, field_name):
    """Get a fixed expected value from the derivative's schema."""
    row = schema.loc[schema["field"] == field_name]

    if row.empty:
        return None

    value = str(row.iloc[0][derivative]).strip()

    # "per-die" means the value is not fixed by this schema.
    if value.lower() == "per-die":
        return None

    try:
        return int(value, 0)
    except ValueError:
        return None


def validate_record(record, schema):
    """Decode one record and report schema-supported violations."""
    derivative = record["derivative"]

    if derivative not in {"MCU-A", "MCU-B", "MCU-C"}:
        raise ValueError(f"Unknown derivative: {derivative}")

    decoded = decode_fuse_map(record["fuse_map_hex"], schema)
    findings = []

    # Check fields whose expected values are fixed by the schema.
    for field_name, error_label in [
        ("FLASH_SIZE", "wrong_flash_size"),
        ("PERIPH_EN", "periph_mask_mismatch"),
        ("TEMP_GRADE", "wrong_temp_grade"),
        ("RDP", "rdp_missing_automotive"),
    ]:
        expected = expected_value(schema, derivative, field_name)

        if expected is not None and decoded[field_name] != expected:
            findings.append({
                "rule": f"{field_name}_mismatch",
                "error_type": error_label,
                "actual": decoded[field_name],
                "expected": expected,
            })

    # DEBUG_LOCK is documented as required in production.
    if decoded["DEBUG_LOCK"] != 1:
        findings.append({
            "rule": "DEBUG_LOCK_required",
            "error_type": "missing_debug_lock",
            "actual": decoded["DEBUG_LOCK"],
            "expected": 1,
        })

    # The schema explicitly states that reserved bits must be zero.
    if decoded["RESERVED"] != 0:
        findings.append({
            "rule": "RESERVED_must_be_zero",
            "error_type": "reserved_bit_set",
            "actual": decoded["RESERVED"],
            "expected": 0,
        })

    return decoded, findings


def main():
    audit = pd.read_csv(AUDIT_FILE)
    schema = load_schema()

    required = {
        "die_id", "derivative", "fuse_map_hex",
        "has_error", "error_type"
    }
    missing = required - set(audit.columns)

    if missing:
        raise ValueError(f"Missing audit columns: {missing}")

    results = []

    for _, record in audit.iterrows():
        decoded, findings = validate_record(record, schema)

        detected_types = sorted({
            finding["error_type"] for finding in findings
        })

        results.append({
            "die_id": record["die_id"],
            "derivative": record["derivative"],
            "actual_error_type": record["error_type"],
            "actual_has_error": bool(record["has_error"]),
            "detected_error_types": detected_types,
            "rule_violation_count": len(findings),
            "rule_findings": findings,
        })

    results_df = pd.DataFrame(results)

    # Save a detailed audit report.
    report_path = PROJECT_ROOT / "reports" / "validation_results.csv"

    results_df.to_csv(report_path, index=False)

    # A row is flagged if at least one implemented rule finds a violation.
    results_df["rule_flagged"] = (
        results_df["rule_violation_count"] > 0
    )

    print("=" * 65)
    print("MCU FUSE MAP VALIDATION REPORT")
    print("=" * 65)

    print(f"\nRecords checked: {len(results_df)}")
    print(f"Records flagged by rules: {results_df['rule_flagged'].sum()}")
    print(f"Reference error records: {results_df['actual_has_error'].sum()}")

    print("\nBinary comparison against reference labels:")
    print(pd.crosstab(
        results_df["actual_has_error"],
        results_df["rule_flagged"],
        rownames=["Reference has_error"],
        colnames=["Rule flagged"],
        dropna=False,
    ).to_string())

    # Exact label comparison is meaningful only for categories
    # covered by the implemented rules.
    covered_labels = {
        "wrong_flash_size",
        "periph_mask_mismatch",
        "missing_debug_lock",
        "rdp_missing_automotive",
        "reserved_bit_set",
        "none",
    }

    covered = results_df[
        results_df["actual_error_type"].isin(covered_labels)
    ].copy()

    covered["exact_label_match"] = covered.apply(
        lambda row: (
            row["actual_error_type"] == "none"
            and not row["detected_error_types"]
        ) or (
            row["actual_error_type"] != "none"
            and row["actual_error_type"] in row["detected_error_types"]
        ),
        axis=1,
    )

    print("\nExact-label coverage check (implemented categories only):")
    print(f"Records in covered categories: {len(covered)}")
    print(
        "Records whose reference category was matched: "
        f"{covered['exact_label_match'].sum()}"
    )
    print(
        "Coverage-category match rate: "
        f"{covered['exact_label_match'].mean():.2%}"
    )

    print("\nReference categories not fully covered by current rules:")
    print("- trim_field_corrupt: no valid trim limits are specified.")
    print("- wrong_temp_grade: check whether this label exists in the data.")
    print("- Other categories may need additional documented rules.")

    print(f"\nDetailed report saved to: {report_path}")

    print("\nFirst five flagged records:")
    flagged = results_df[results_df["rule_flagged"]]

    if flagged.empty:
        print("No records flagged by the implemented rules.")
    else:
        print(flagged[
            [
                "die_id",
                "derivative",
                "actual_error_type",
                "detected_error_types",
            ]
        ].head().to_string(index=False))


if __name__ == "__main__":
    main()
