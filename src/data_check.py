from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

FILES = [
    "sales_daily.csv",
    "sku_master.csv",
    "calendar.csv",
    "inventory_snapshots.csv"
]


def load_and_check(filename):
    path = DATA_DIR / filename

    if not path.exists():
        print(f"ERROR: {filename} not found")
        return

    df = pd.read_csv(path)

    print("\n" + "=" * 60)
    print(filename)
    print("=" * 60)

    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    print("\nFirst 5 rows:")
    print(df.head())


def main():
    for file in FILES:
        load_and_check(file)


if __name__ == "__main__":
    main()