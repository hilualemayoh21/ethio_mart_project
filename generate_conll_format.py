import pandas as pd
import os
import re
import json
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# === Paths ===
INPUT_CSV = "data/text/Shageronlinestore_messages_processed.csv"
OUTPUT_CONLL = "data/amharic_ner_conll_auto.txt"
OUTPUT_JSON = "data/amharic_ner_labeled.json"
os.makedirs("data", exist_ok=True)

# === Load and sample data ===
df = pd.read_csv(INPUT_CSV)
df = df[df["preprocessed_text"].notna() & df["preprocessed_text"].str.strip().ne("")]
sampled = df.sample(n=min(30, len(df)), random_state=42)

# === Labeling Function ===
def label_tokens(text):
    tokens = word_tokenize(text)
    labels = ["O"] * len(tokens)

    price_number_pattern = re.compile(r"^\d+([.,]\d+)?$")

    # --- PRICE detection ---
    for i in range(len(tokens)):
        if i + 2 < len(tokens):
            if tokens[i] in ["ዋጋ", "በ"] and price_number_pattern.match(tokens[i+1]) and tokens[i+2] == "ብር":
                labels[i] = "B-PRICE"
                labels[i+1] = "I-PRICE"
                labels[i+2] = "I-PRICE"
                continue
        if i + 1 < len(tokens):
            if price_number_pattern.match(tokens[i]) and tokens[i+1] == "ብር":
                labels[i] = "B-PRICE"
                labels[i+1] = "I-PRICE"

    # --- PRODUCT detection ---
    product_keywords = {
        "መክፈቻ", "ቦትል", "ካሜራ", "ጫማ", "ስትሮ", "መጠጣት", "ጌጥ", "አልባስ",
        "jar", "mason", "glass", "bottle", "phone", "watch", "shoes"
    }
    i = 0
    while i < len(tokens):
        if tokens[i].lower() in product_keywords or tokens[i] in product_keywords:
            if labels[i] == "O":
                labels[i] = "B-Product"
                j = i + 1
                while j < len(tokens) and tokens[j].isalpha() and labels[j] == "O":
                    labels[j] = "I-Product"
                    j += 1
                i = j
                continue
        i += 1

    # --- LOCATION detection ---
    location_keywords = {
        "አዲስ", "አበባ", "ቦሌ", "ሃዋሳ", "ጅማ", "ባህር", "ዳር", "ጎንደር",
        "ቤተክርስቲያን", "ህንፃ", "ፊት", "መንገድ", "ማዕከል", "ግቢ", "አቅራቢያ",
        "ቤት", "ዩኒቨርሲቲ", "ስትሪት", "ቢሮ", "ቅርንጫፍ", "ቁ", "ቁጥር", "አድራሻ",
        "ክፍል", "ፎሎር", "መደብ", "መስኮት", "ቦርድ", "ማዕከላዊ", "ደጅ", "መስክ"
    }

    i = 0
    while i < len(tokens):
        if tokens[i] in location_keywords or re.match(r"^(ቁ|ቁጥር)?\d+$", tokens[i]):
            if labels[i] == "O":
                labels[i] = "B-LOC"
                j = i + 1
                while j < len(tokens):
                    if (tokens[j] in location_keywords or
                        re.match(r"^[\u1200-\u137F]+$", tokens[j]) or
                        re.match(r"^\d+$", tokens[j])):
                        if labels[j] == "O":
                            labels[j] = "I-LOC"
                            j += 1
                        else:
                            break
                    else:
                        break
                # Label 1-2 more tokens after address if they're not already labeled
                k = j
                while k < len(tokens) and k < j + 2 and labels[k] == "O":
                    labels[k] = "I-LOC"
                    k += 1
                i = k
                continue
        i += 1

    return tokens, labels

# === Labeling and Export ===
conll_lines = []
json_data = []

for _, row in sampled.iterrows():
    tokens, tags = label_tokens(row["preprocessed_text"])
    for t, tag in zip(tokens, tags):
        conll_lines.append(f"{t} {tag}")
    conll_lines.append("")  # sentence separator
    json_data.append([{"token": t, "label": tag} for t, tag in zip(tokens, tags)])

# Save to CoNLL
with open(OUTPUT_CONLL, "w", encoding="utf-8") as f:
    f.write("\n".join(conll_lines))
print(f"[✔] CoNLL format saved to: {OUTPUT_CONLL}")

# Save to JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)
print(f"[✔] JSON format saved to: {OUTPUT_JSON}")
