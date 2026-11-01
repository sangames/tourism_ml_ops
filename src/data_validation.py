"""Validate and summarize the registered tourism dataset."""
from pathlib import Path
import pandas as pd

EXPECTED_COLUMNS = {
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
}


def load_registered_data(path: str | Path) -> pd.DataFrame:
    """Load the repository CSV and remove only a saved dataframe index column."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = pd.read_csv(path)
    unnamed = [col for col in data.columns if not str(col).strip() or str(col).lower().startswith("unnamed:")]
    if unnamed:
        data = data.drop(columns=unnamed)
    return data


def validate_dataset(path: str | Path) -> pd.DataFrame:
    """Check schema and target values, then print a compact data summary."""
    data = load_registered_data(path)
    missing_columns = EXPECTED_COLUMNS.difference(data.columns)
    extra_columns = set(data.columns).difference(EXPECTED_COLUMNS)
    if missing_columns:
        raise ValueError(f"Missing expected columns: {sorted(missing_columns)}")
    if extra_columns:
        raise ValueError(f"Unexpected columns: {sorted(extra_columns)}")
    if data.empty:
        raise ValueError("Dataset has no rows.")
    target_values = set(pd.to_numeric(data["ProdTaken"], errors="coerce").dropna().unique())
    if target_values != {0, 1}:
        raise ValueError(f"ProdTaken must contain the binary labels 0 and 1; found {target_values}")
    print(f"Validated dataset: {len(data):,} rows × {len(data.columns)} columns")
    print(f"Exact duplicate rows: {data.duplicated().sum():,}")
    print("Missing values by column:")
    print(data.isna().sum().to_string())
    print("Target distribution:")
    print(data["ProdTaken"].value_counts().sort_index().to_string())
    print("Target share:")
    print((data["ProdTaken"].value_counts(normalize=True).sort_index() * 100).round(2).to_string())
    return data


if __name__ == "__main__":
    validate_dataset(Path(__file__).resolve().parents[1] / "data" / "tourism.csv")
