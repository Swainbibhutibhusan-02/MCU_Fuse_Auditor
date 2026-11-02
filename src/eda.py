
from pathlib import Path
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

AUDIT_FILE = DATA_DIR / "mcu_fuse_map_audit.csv"
SCHEMA_FILE = DATA_DIR / "mcu_fuse_schema.csv"


def inspect_dataset(file_path: Path) -> None:
    """Display basic information about a CSV dataset."""

    print("\n" + "=" * 65)
    print(f"DATASET: {file_path.name}")
    print("=" * 65)

    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return

    df = pd.read_csv(file_path)

    print("\nShape (rows, columns):")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst five rows:")
    print(df.head().to_string(index=False))

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nMissing values per column:")
    print(df.isnull().sum().to_string())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nNumerical summary:")
    print(df.describe(include="all").to_string())

    if "has_error" in df.columns:
        print("\nError label distribution:")
        print(df["has_error"].value_counts(dropna=False).to_string())

    if "error_type" in df.columns:
        print("\nError type distribution:")
        print(df["error_type"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    inspect_dataset(AUDIT_FILE)
    inspect_dataset(SCHEMA_FILE)
