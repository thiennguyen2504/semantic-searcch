"""Download, clean, and prepare Stack Overflow quality questions dataset."""
import os
import glob
import json
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
import kagglehub

DATASET_NAME = "imoore/60k-stack-overflow-questions-with-quality-rate"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"


def clean_html(text: str) -> str:
    """Strip HTML tags using BeautifulSoup with space separator."""
    if not isinstance(text, str) or not text.strip():
        return ""
    soup = BeautifulSoup(text, "html.parser")
    # Collapse multiple whitespace
    cleaned = soup.get_text(separator=" ")
    return " ".join(cleaned.split())


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dev_output_path = OUTPUT_DIR / "documents_dev.json"
    full_output_path = OUTPUT_DIR / "documents_full.json"

    print(f"[1/7] Downloading dataset '{DATASET_NAME}' via kagglehub...")
    dataset_path = kagglehub.dataset_download(DATASET_NAME)
    print(f"      Dataset downloaded to: {dataset_path}")

    # Inspect files in the downloaded directory
    files = os.listdir(dataset_path)
    print(f"      Files found in directory: {files}")

    csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))
    print(f"      CSV files located: {[os.path.basename(f) for f in csv_files]}")

    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in dataset path: {dataset_path}")

    # Prioritize train.csv if multiple, or combine/select the main dataset file
    target_csv = None
    for f in csv_files:
        if "train" in os.path.basename(f).lower():
            target_csv = f
            break
    if not target_csv:
        target_csv = csv_files[0]

    print(f"[2/7] Reading CSV file: {target_csv}")
    df = pd.read_csv(target_csv)
    initial_count = len(df)
    print(f"      Initial total rows: {initial_count}")

    # Filter Y == "HQ"
    print("[3/7] Filtering rows where Y == 'HQ'...")
    df = df[df["Y"] == "HQ"].copy()
    hq_count = len(df)
    print(f"      Rows remaining after Y == 'HQ' filter: {hq_count}")

    # Strip HTML from Body
    print("[4/7] Stripping HTML tags from 'Body' column using BeautifulSoup...")
    df["content"] = df["Body"].apply(clean_html)

    # Filter rows where content length >= 50
    print("[5/7] Filtering rows where content length >= 50 characters...")
    df = df[df["content"].str.len() >= 50].copy()
    len_filtered_count = len(df)
    print(f"      Rows remaining after length filter (>= 50 chars): {len_filtered_count}")

    # Rename Title -> title
    df = df.rename(columns={"Title": "title"})
    df = df[["title", "content"]].reset_index(drop=True)

    # Save full cleaned HQ dataset
    print(f"[6/7] Saving full cleaned HQ dataset to '{full_output_path}'...")
    records_full = df.to_dict(orient="records")
    with open(full_output_path, "w", encoding="utf-8") as f:
        json.dump(records_full, f, ensure_ascii=False, indent=2)
    print(f"      Saved {len(records_full)} documents to {full_output_path}")

    # Sample 3000 rows for dev
    sample_size = min(3000, len(df))
    print(f"[7/7] Sampling {sample_size} rows (random_state=42) for dev set...")
    df_dev = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    records_dev = df_dev.to_dict(orient="records")
    with open(dev_output_path, "w", encoding="utf-8") as f:
        json.dump(records_dev, f, ensure_ascii=False, indent=2)
    print(f"      Saved {len(records_dev)} documents to {dev_output_path}")

    print("\nData preparation completed successfully!")
    print(f"- Initial rows: {initial_count}")
    print(f"- HQ rows: {hq_count}")
    print(f"- Content >= 50 chars rows: {len_filtered_count}")
    print(f"- dev documents: {len(records_dev)}")
    print(f"- full documents: {len(records_full)}")


if __name__ == "__main__":
    main()
