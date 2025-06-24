from langdetect import detect
import pandas as pd
import re
from cleantext import clean
from nltk.tokenize import word_tokenize
import nltk
import os

nltk.download("punkt")

# === Text preprocessing ===
def preprocess_amharic(text):
    text = clean(
        text,
        fix_unicode=True,
        to_ascii=False,
        lower=True,
        no_line_breaks=True,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_digits=False,
        no_currency_symbols=False,
        no_punct=True,
    )
    return re.sub(r"\s+", " ", text).strip()

def tokenize_amharic(text):
    return word_tokenize(text)

def is_amharic(text):
    if not isinstance(text, str):
        return False
    return any("\u1200" <= c <= "\u137F" for c in text)

def detect_language(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return "unknown"
    try:
        lang = detect(text)
        if lang == "am" or is_amharic(text):
            return "am"
        return lang
    except:
        return "am" if is_amharic(text) else "unknown"

# === Process a single CSV file ===
def process_file(file_path):
    print(f"[⏳] Processing: {file_path}")
    df = pd.read_csv(file_path)

    if "text" not in df.columns:
        print(f"[⚠️] 'text' column missing in: {file_path}")
        return

    # Detect language
    df["language"] = df["text"].astype(str).apply(detect_language)

    # Filter to Amharic
    amharic_df = df[df["language"] == "am"].copy()

    if amharic_df.empty:
        print(f"[⚠️] No Amharic content found in: {file_path}")
        return

    # Preprocess and tokenize
    amharic_df["preprocessed_text"] = amharic_df["text"].apply(preprocess_amharic)
    amharic_df["tokens"] = amharic_df["preprocessed_text"].apply(tokenize_amharic)

    # Save processed file
    out_path = file_path.replace(".csv", "_processed.csv")
    amharic_df.to_csv(out_path, index=False)
    print(f"[✅] Processed file saved to: {out_path} ({len(amharic_df)} Amharic messages)")

# === Run preprocessing on all files in the text folder ===
if __name__ == "__main__":
    text_folder = "data/text"
    if not os.path.exists(text_folder):
        print("[❌] Text folder does not exist:", text_folder)
    else:
        for file in os.listdir(text_folder):
            if file.endswith("_messages.csv"):
                process_file(os.path.join(text_folder, file))
