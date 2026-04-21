#!/usr/bin/env bash
set -e
cd /Users/kingisaac/Github/Brand-Perception-and-Sentiment-Analysis/services/reddit_data
# activate venv
source /Users/kingisaac/Github/Brand-Perception-and-Sentiment-Analysis/.venv/bin/activate
# choose environment vars (example list). Adjust as needed.
export SUBREDDITS="pasta,news,technology,science,funny"
export SUBREDDITS_PER_RUN=1
export APIFY_SEARCH="lasagna"
export DATABASE_FILE="/Users/kingisaac/Github/Brand-Perception-and-Sentiment-Analysis/services/reddit_data/master.db"
# run
python apify_scraper.py >> /Users/kingisaac/Github/Brand-Perception-and-Sentiment-Analysis/services/reddit_data/scraper.log 2>&1