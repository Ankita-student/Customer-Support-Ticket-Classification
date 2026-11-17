"""Task 1 - Data preparation.

Reads data/tickets.csv, inspects it, cleans it and writes data/tickets_clean.csv.
`clean_text()` is also imported by train.py / predict.py / app.py so that the
SAME cleaning is applied at training time and at prediction time.

Run:  python src/preprocessing.py
"""
import json
import re

import pandas as pd

from config import CLEAN_CSV, OUTPUT_DIR, RAW_CSV, ensure_dirs


# ----------------------------------------------------------------------------
# Text cleaning
# ----------------------------------------------------------------------------
def clean_text(text):
    """Lower-case, remove punctuation/extra spaces, expand common contractions.

    Negations ("not", "cannot") are deliberately KEPT because they carry meaning
    ("password is not working"). Digits are kept too (error codes like 500, 404).
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.replace("\u2019", "'")                       # curly apostrophe
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)       # URLs
    text = re.sub(r"\S+@\S+", " ", text)                     # e-mail addresses
    text = text.replace("can't", "cannot").replace("won't", "will not")
    text = re.sub(r"n't\b", " not", text)                    # isn't -> is not
    text = text.replace("'", "")                             # remaining apostrophes
    text = re.sub(r"[^a-z0-9\s]", " ", text)                 # punctuation / symbols
    text = re.sub(r"\s+", " ", text).strip()                 # extra spaces
    return text


# ----------------------------------------------------------------------------
# Loading / inspection helpers
# ----------------------------------------------------------------------------
def load_raw(path=RAW_CSV):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'python src/generate_dataset.py' first "
            "(or place your own CSV there)."
        )
    return pd.read_csv(path)


def inspect(df, title="DATASET"):
    print(f"\n----- {title} -----")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nMissing values per column:")
    print(df.isnull().sum().to_string())
    print(f"\nExact duplicate rows: {df.duplicated().sum()}")


# ----------------------------------------------------------------------------
# Cleaning pipeline
# ----------------------------------------------------------------------------
def clean_dataframe(raw):
    """Return (clean_df, summary_dict). Every step is logged in the summary."""
    df = raw.copy()
    s = {"raw_rows": int(len(df))}
    s["missing_before"] = {c: int(v) for c, v in df.isnull().sum().items()}
    s["exact_duplicates"] = int(df.duplicated().sum())

    # 1. tidy whitespace / spelling of text columns
    for col in ["ticket_description", "category", "priority", "status", "customer_name"]:
        if col in df.columns:
            df[col] = df[col].astype("object").where(df[col].notnull(), None)
            df[col] = df[col].apply(lambda v: re.sub(r"\s+", " ", v).strip() if isinstance(v, str) else v)
    for col in ["category", "priority", "status"]:
        df[col] = df[col].apply(lambda v: v.title() if isinstance(v, str) else v)

    # 2. rows without description or category cannot be used for learning -> drop
    before = len(df)
    df = df.dropna(subset=["ticket_description", "category"])
    df = df[df["ticket_description"].str.len() > 0]
    s["dropped_missing_description_or_category"] = int(before - len(df))

    # 3. fill other missing values
    s["filled_priority"] = int(df["priority"].isnull().sum())
    s["filled_status"] = int(df["status"].isnull().sum())
    priority_mode = df["priority"].mode().iloc[0]
    df["priority"] = df["priority"].fillna(priority_mode)
    df["status"] = df["status"].fillna("Unknown")
    s["priority_filled_with"] = str(priority_mode)

    # 4. remove exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    s["removed_exact_duplicates"] = int(before - len(df))

    # 5. text cleaning
    df["clean_description"] = df["ticket_description"].apply(clean_text)
    df = df[df["clean_description"].str.len() > 0]

    # 6. remove re-submitted tickets (same cleaned text) - keep the first one
    before = len(df)
    df = df.drop_duplicates(subset=["clean_description"], keep="first")
    s["removed_duplicate_descriptions"] = int(before - len(df))

    # 7. proper data types
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.reset_index(drop=True)
    s["clean_rows"] = int(len(df))
    s["n_categories"] = int(df["category"].nunique())
    return df, s


def main():
    ensure_dirs()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 70)

    raw = load_raw()
    print("=" * 70)
    print("TASK 1 - DATA PREPARATION")
    print("=" * 70)
    print("\nFirst 10 rows of the raw dataset:")
    print(raw.head(10).to_string(index=False))

    inspect(raw, "RAW DATA CHECKS")
    dup_ids = raw["ticket_id"].duplicated().sum()
    print(f"Duplicate ticket_ids: {dup_ids}")

    clean, summary = clean_dataframe(raw)

    print("\n----- CLEANING LOG -----")
    print(f"Rows dropped (missing description/category) : {summary['dropped_missing_description_or_category']}")
    print(f"Missing priority filled with mode ({summary['priority_filled_with']:<6})   : {summary['filled_priority']}")
    print(f"Missing status filled with 'Unknown'         : {summary['filled_status']}")
    print(f"Exact duplicate rows removed                 : {summary['removed_exact_duplicates']}")
    print(f"Re-submitted (same text) tickets removed     : {summary['removed_duplicate_descriptions']}")
    print(f"Rows: {summary['raw_rows']} -> {summary['clean_rows']}")

    print("\n----- TEXT CLEANING EXAMPLES -----")
    for orig, cleaned in clean[["ticket_description", "clean_description"]].head(5).values:
        print(f"  before: {orig}")
        print(f"  after : {cleaned}\n")

    inspect(clean, "CLEANED DATA CHECKS")
    print("\nDataset info:")
    clean.info()
    lens = clean["clean_description"].str.split().str.len()
    print(f"\nWords per description -> min {lens.min()}, mean {lens.mean():.1f}, max {lens.max()}")

    clean.to_csv(CLEAN_CSV, index=False, encoding="utf-8")
    with open(OUTPUT_DIR / "preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved cleaned data -> {CLEAN_CSV}")


if __name__ == "__main__":
    main()
