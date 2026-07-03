import argparse
import csv
import json
from pathlib import Path


TEXT_FIELDS = ("text", "tweet_text", "full_text", "content", "comment", "body")
SENTIMENT_FIELDS = ("sentiment", "label", "polarity", "class")
SENTIMENT_MAP = {
    "positive": "positive",
    "pos": "positive",
    "1": "positive",
    "negative": "negative",
    "neg": "negative",
    "0": "negative",
    "-1": "negative",
    "neutral": "neutral",
    "neu": "neutral",
    "2": "neutral",
}


def pick_value(row, fields):
    for field in fields:
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def normalize_sentiment(value):
    key = str(value).strip().lower()
    return SENTIMENT_MAP.get(key, "")


def load_json_records(path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        for key in ("data", "results", "tweets", "items", "rows"):
            value = raw.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [raw]
    return []


def load_jsonl_records(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
    return rows


def load_csv_records(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_records(path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_csv_records(path)
    if suffix in {".jsonl", ".ndjson"}:
        return load_jsonl_records(path)
    if suffix == ".json":
        return load_json_records(path)
    raise ValueError("Use a CSV, JSON, JSONL, or NDJSON TweetClaw export.")


def convert_rows(records, default_sentiment):
    rows = []
    skipped = 0
    for record in records:
        text = pick_value(record, TEXT_FIELDS)
        sentiment = normalize_sentiment(pick_value(record, SENTIMENT_FIELDS))
        if not sentiment and default_sentiment:
            sentiment = default_sentiment
        if not text or not sentiment:
            skipped += 1
            continue
        rows.append({"text": text, "sentiment": sentiment})
    return rows, skipped


def write_rows(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("text", "sentiment"))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a TweetClaw export into data/social_media_data.csv."
    )
    parser.add_argument("input", type=Path, help="TweetClaw CSV, JSON, JSONL, or NDJSON export")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/social_media_data.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--default-sentiment",
        choices=("positive", "negative", "neutral"),
        help="Use this label for unlabeled rows instead of skipping them",
    )
    args = parser.parse_args()

    rows, skipped = convert_rows(load_records(args.input), args.default_sentiment)
    write_rows(rows, args.output)
    print(f"Converted {len(rows)} rows to {args.output}. Skipped {skipped} rows.")


if __name__ == "__main__":
    main()
