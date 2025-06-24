import pandas as pd
import json
import os

def convert_raw_csv_to_jsonl(csv_file, jsonl_file):
    df = pd.read_csv(csv_file)
    with open(jsonl_file, "w", encoding="utf-8") as out:
        for _, row in df.iterrows():
            record = {
                "message_id": row.get("message_id"),
                "timestamp": row.get("date"),
                "sender": row.get("sender_id"),
                "text": row.get("text"),
                "image_path": row.get("image_path") if not pd.isna(row.get("image_path")) else None,
                # No preprocessed_text or tokens here for raw data
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[✓] Raw JSONL file created at {jsonl_file}")

def convert_processed_csv_to_jsonl(csv_file, jsonl_file):
    df = pd.read_csv(csv_file)
    with open(jsonl_file, "w", encoding="utf-8") as out:
        for _, row in df.iterrows():
            record = {
                "message_id": row.get("message_id"),
                "timestamp": row.get("date"),
                "sender": row.get("sender_id"),
                "text": row.get("text"),
                "preprocessed_text": row.get("preprocessed_text"),
                "tokens": eval(row.get("tokens")) if pd.notna(row.get("tokens")) else [],
                "image_path": row.get("image_path") if not pd.isna(row.get("image_path")) else None
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[✓] Processed JSONL file created at {jsonl_file}")

def convert_all_files(folder_path="data/text", output_folder="data/jsonl"):
    os.makedirs(output_folder, exist_ok=True)
    for file in os.listdir(folder_path):
        if file.endswith("_messages.csv"):
            # Raw scraped file
            csv_path = os.path.join(folder_path, file)
            jsonl_name = file.replace("_messages.csv", "_raw.jsonl")
            jsonl_path = os.path.join(output_folder, jsonl_name)
            convert_raw_csv_to_jsonl(csv_path, jsonl_path)
        elif file.endswith("_processed.csv"):
            # Processed file
            csv_path = os.path.join(folder_path, file)
            jsonl_name = file.replace("_processed.csv", ".jsonl")
            jsonl_path = os.path.join(output_folder, jsonl_name)
            convert_processed_csv_to_jsonl(csv_path, jsonl_path)

convert_all_files()