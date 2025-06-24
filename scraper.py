import os
import asyncio
from dotenv import load_dotenv
from telethon import TelegramClient, events
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.tl.types import Channel
import pandas as pd
from datetime import datetime
from cleantext import clean
from nltk.tokenize import word_tokenize
import re
import nltk

nltk.download('punkt')

# Load credentials
load_dotenv()
api_id = int(os.getenv("API_ID"))
api_hash = os.getenv("API_HASH")

# Paths
TEXT_PATH = "data/text"
IMAGE_PATH = "data/images"
DOC_PATH = "data/documents"
os.makedirs(TEXT_PATH, exist_ok=True)
os.makedirs(IMAGE_PATH, exist_ok=True)
os.makedirs(DOC_PATH, exist_ok=True)

# Init Telegram client
client = TelegramClient('ethio_mart_scraper', api_id, api_hash)

# ✅ Verified marketing channels only
channels = [
    'Shageronlinestore',
    'OnlineShoppingEthiopia',
    'AddisMart',
    'NEW BRAND'
]

# Text preprocessing
def preprocess_amharic(text):
    if not text:
        return ""
    text = clean(text, fix_unicode=True, to_ascii=False, lower=True,
                 no_line_breaks=True, no_urls=True, no_emails=True,
                 no_phone_numbers=True, no_digits=False,
                 no_currency_symbols=False, no_punct=True)
    return re.sub(r"\s+", " ", text).strip()

def tokenize_amharic(text):
    return word_tokenize(text)

# Download media safely
async def safe_download_media(message, save_path):
    try:
        print(f"[🖼] Attempting media download from message {message.id} to {save_path}")
        result_path = await client.download_media(message, file=save_path)
        if result_path and os.path.isfile(result_path):
            print(f"[✔️] Downloaded media to {result_path}")
            return result_path
        else:
            print(f"[FAIL] Media download returned invalid path: {result_path}")
            return None
    except Exception as e:
        print(f"[ERROR] Exception during media download: {e}")
        return None

# Get channel entity by name/title
async def resolve_entity(identifier):
    try:
        return await client.get_entity(identifier)
    except:
        dialogs = await client.get_dialogs()
        for dialog in dialogs:
            entity = dialog.entity
            if isinstance(entity, Channel):
                if entity.title.strip().lower() == identifier.strip().lower():
                    return entity
        print(f"[✖️] Channel '{identifier}' not found.")
        return None

# Historical message fetching with max 30 image downloads
async def fetch_channel_data(identifier):
    entity = await resolve_entity(identifier)
    if not entity:
        return

    print(f"[ℹ️] Fetching history for '{identifier}'")
    history = await client(GetHistoryRequest(
        peer=entity,
        limit=200,
        offset_date=None,
        offset_id=0,
        max_id=0,
        min_id=0,
        add_offset=0,
        hash=0
    ))

    rows = []
    image_download_count = 0
    MAX_IMAGES = 30

    for msg in history.messages:
        if msg.message:
            image_path = None
            doc_path = None

            if msg.media:
                if msg.photo and image_download_count < MAX_IMAGES:
                    image_file = os.path.join(IMAGE_PATH, f"{msg.id}.jpg")
                    image_path = await safe_download_media(msg, image_file)
                    if image_path:
                        image_download_count += 1
                elif msg.document:
                    doc_file = os.path.join(DOC_PATH, f"{msg.id}.dat")
                    doc_path = await safe_download_media(msg, doc_file)

            row = {
                'message_id': msg.id,
                'date': msg.date,
                'sender_id': getattr(msg.from_id, 'user_id', None),
                'text': msg.message,
                'preprocessed_text': preprocess_amharic(msg.message),
                'tokens': tokenize_amharic(preprocess_amharic(msg.message)),
                'image_path': image_path,
                'document_path': doc_path
            }

            rows.append(row)

    df = pd.DataFrame(rows)
    filename = os.path.join(TEXT_PATH, f"{identifier.replace(' ', '_')}_messages.csv")
    df.to_csv(filename, index=False)
    print(f"[+] Saved {len(rows)} messages from '{identifier}' to {filename}")

# Global for real-time image tracking
image_download_count_realtime = 0
MAX_IMAGES_REALTIME = 30

# Real-time message handler
@client.on(events.NewMessage)
async def handler(event):
    global image_download_count_realtime
    sender = await event.get_chat()

    # ✅ Filter: only messages from marketing-related channels
    if isinstance(sender, Channel) and any(kw in sender.title.lower() for kw in ["mart", "shop", "store", "brand"]):
        name = sender.username if sender.username else sender.title
        image_path = None
        doc_path = None

        if event.message.media:
            if event.message.photo and image_download_count_realtime < MAX_IMAGES_REALTIME:
                image_file = os.path.join(IMAGE_PATH, f"{event.message.id}.jpg")
                image_path = await safe_download_media(event.message, image_file)
                if image_path:
                    image_download_count_realtime += 1
            elif event.message.document:
                doc_file = os.path.join(DOC_PATH, f"{event.message.id}.dat")
                doc_path = await safe_download_media(event.message, doc_file)

        row = {
            'message_id': event.message.id,
            'date': event.message.date,
            'sender_id': getattr(event.message.from_id, 'user_id', None),
            'text': event.message.message,
            'preprocessed_text': preprocess_amharic(event.message.message),
            'tokens': tokenize_amharic(preprocess_amharic(event.message.message)),
            'image_path': image_path,
            'document_path': doc_path
        }

        file = os.path.join(TEXT_PATH, f"{name.replace(' ', '_')}_messages.csv")
        if os.path.exists(file):
            df = pd.read_csv(file)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(file, index=False)
        print(f"[📥] Real-time message saved from '{name}'")

# Main runner
async def main():
    await client.start()
    for ch in channels:
        await fetch_channel_data(ch)
    print("[✔️] Listening for new messages...")
    await client.run_until_disconnected()

# Run client
with client:
    client.loop.run_until_complete(main())
