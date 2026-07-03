# Social Media Sentiment Analysis Dashboard

## Overview
This project analyzes social media text and predicts sentiment using ML.

## Tech Stack
Python, Scikit-learn, TF-IDF, Streamlit

## Features
- Text preprocessing
- ML model
- Dashboard

## Run
python main.py
streamlit run app/app.py

## TweetClaw Export Data

To train with reviewed TweetClaw exports, convert the export into the existing
`data/social_media_data.csv` schema:

```bash
python tweetclaw_to_social_media_csv.py exports/tweetclaw.json
python main.py
```

The converter accepts TweetClaw CSV, JSON, JSONL, and NDJSON exports. It writes
`text,sentiment` rows and skips rows without text or a usable sentiment label.
Use `--default-sentiment neutral` only when you intentionally want unlabeled rows
to train as neutral examples.

## Github Repo Link
https://github.com/adityatamhanedev/Social-Media-Sentiment-Analysis-Dashboard
