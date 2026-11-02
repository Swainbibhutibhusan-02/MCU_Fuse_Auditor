
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

AUDIT_FILE = DATA_DIR / "mcu_fuse_map_audit.csv"
SCHEMA_FILE = DATA_DIR / "mcu_fuse_schema.csv"


def load_schema():
    """Load and validate the fuse-field schema."""
    schema = pd.read_csv(SCHEMA_FILE)

    required_columns = {"field", "lsb", "nbits"}
    missing = required_columns - set(schema.columns)

    if missing:
        raise ValueError(f"Missing schema columns: {missing}")

    if schema["field"].duplicated().any():
        raise ValueError("Duplicate field names found in schema.")

    if (schema["lsb"] < 0).any():
        raise ValueError("LSB positions cannot be negative.")

    if (schema["nbits"] <= 0).any():
        raise ValueError("Field widths must be greater than zero.")

    # Check that fields do not occupy overlapping bit positions.
    occupied_bits = set()

    for _, row in schema.iterrows():
        start = int(row["lsb"])
        width = int(row["nbits"])
        field_name = row["field"]

        bits = set(range(start, start + width))
        overlap = occupied_bits.intersection(bits)

        if overlap:
            raise ValueError(
                f"Field {field_name} overlaps existing bits: "
                f"{sorted(overlap)}"
            )

        occupied_bits.update(bits)

    return schema


def decode_fuse_map(fuse_map_hex, schema):
    """Extract every defined field from a hexadecimal fuse map."""
    try:
        fuse_value = int(str(fuse_map_hex).strip(), 16)
    except ValueError as exc:
        raise ValueError(
            f"Invalid hexadecimal fuse map: {fuse_map_hex}"
        ) from exc

    if fuse_value < 0:
        raise ValueError("Fuse map cannot be negative.")

    decoded = {}

    for _, row in schema.iterrows():
        field_name = row["field"]
        lsb = int(row["lsb"])
        nbits = int(row["nbits"])

        mask = (1 << nbits) - 1
        decoded[field_name] = (fuse_value >> lsb) & mask

    return decoded


def main():
    audit = pd.read_csv(AUDIT_FILE)
    schema = load_schema()

    print("Schema validation passed.")
    print(f"Fuse fields available: {len(schema)}")

    # Decode one record for each MCU derivative.
    for derivative in sorted(audit["derivative"].unique()):
        sample = audit.loc[
            audit["derivative"] == derivative
        ].iloc[0]

        decoded = decode_fuse_map(
            sample["fuse_map_hex"],
            schema
        )

        print("\n" + "=" * 55)
        print(f"Derivative: {derivative}")
        print(f"Die ID: {sample['die_id']}")
        print(f"Fuse map: {sample['fuse_map_hex']}")
        print(f"Known error label: {sample['error_type']}")
        print("\nDecoded field values:")

        for field_name, value in decoded.items():
            print(f"  {field_name}: {value}")


if __name__ == "__main__":
    main()
