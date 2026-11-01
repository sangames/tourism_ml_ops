"""Clean data and create stratified train/test splits for the next workflow job."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_validation import load_registered_data

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "data" / "tourism.csv"
OUTPUT_DIR = ROOT / "artifacts" / "processed"
TARGET = "ProdTaken"
IDENTIFIER_COLUMNS = ["CustomerID"]
# These fields are only known after a contact or sales pitch and would leak future information.
POST_CONTACT_COLUMNS = [
    "TypeofContact", "DurationOfPitch", "NumberOfFollowups",
    "ProductPitched", "PitchSatisfactionScore",
]


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Standardize obvious label variants and remove duplicates without arbitrary outlier deletion."""
    cleaned = data.drop_duplicates().copy()
    cleaned["Gender"] = cleaned["Gender"].astype("string").str.strip().str.lower().replace({"fe male": "female"})
    cleaned["Occupation"] = cleaned["Occupation"].astype("string").str.strip().str.lower().replace({"free lancer": "freelancer"})
    cleaned["MaritalStatus"] = cleaned["MaritalStatus"].astype("string").str.strip()
    cleaned["Designation"] = cleaned["Designation"].astype("string").str.strip()
    cleaned[TARGET] = pd.to_numeric(cleaned[TARGET], errors="raise").astype("int8")
    return cleaned.reset_index(drop=True)


def make_splits(data: pd.DataFrame, output_dir: str | Path = OUTPUT_DIR, test_size: float = 0.20, seed: int = 42):
    """Remove IDs and post-contact fields, then save stratified train/test CSVs."""
    cleaned = clean_data(data)
    drop_columns = [c for c in IDENTIFIER_COLUMNS + POST_CONTACT_COLUMNS if c in cleaned.columns]
    modeling_data = cleaned.drop(columns=drop_columns)
    if TARGET not in modeling_data:
        raise ValueError(f"Target column {TARGET} is missing after cleaning.")
    train, test = train_test_split(
        modeling_data,
        test_size=test_size,
        random_state=seed,
        stratify=modeling_data[TARGET],
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path, test_path = output_dir / "train.csv", output_dir / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    print(f"Removed identifier fields: {IDENTIFIER_COLUMNS}")
    print(f"Excluded post-contact fields to prevent leakage: {POST_CONTACT_COLUMNS}")
    print(f"Cleaned rows: {len(cleaned):,}; modeling predictors: {modeling_data.shape[1]-1}")
    print(f"Train split: {train.shape}; Test split: {test.shape}")
    print(f"Train target share: {train[TARGET].mean():.3f}; Test target share: {test[TARGET].mean():.3f}")
    print(f"Saved splits to {output_dir}")
    return train, test


if __name__ == "__main__":
    make_splits(load_registered_data(SOURCE_PATH))
