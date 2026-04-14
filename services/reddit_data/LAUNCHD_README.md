**Install and run the Reddit Apify scraper via launchd (macOS)**

Place the plist in your user LaunchAgents and load it to run the scraper every 5 minutes.

1. Ensure the wrapper is executable and configured:

```bash
chmod +x /Users/kingisaac/Github/Brand-Perception-and-Sentiment-Analysis/services/reddit_data/run_apify_scraper.sh
# edit .env or export env vars as needed, example:
cat > /Users/kingisaac/Github/Brand-Perception-and-Sentiment-Analysis/services/reddit_data/.env <<EOF
SUBREDDITS=pasta,news,technology,science,funny
SUBREDDITS_PER_RUN=1
APIFY_SEARCH=lasagna
DATABASE_FILE=/Users/kingisaac/Github/Brand-Perception-and-Sentiment-Analysis/services/reddit_data/master.db
END_PAGE=10
MAX_ITEMS=500
EOF
```

2. Copy the plist into your LaunchAgents folder and load it:

```bash
cp /Users/kingisaac/Github/Brand-Perception-and-Sentiment-Analysis/services/reddit_data/com.username.apifyscraper.plist ~/Library/LaunchAgents/
launchctl unload ~/Library/LaunchAgents/com.username.apifyscraper.plist 2>/dev/null || true
launchctl load ~/Library/LaunchAgents/com.username.apifyscraper.plist
```

3. Useful `launchctl` commands:

```bash
# show status
launchctl list | grep com.username.apifyscraper

# stop
launchctl unload ~/Library/LaunchAgents/com.username.apifyscraper.plist

# start (reload)
launchctl load ~/Library/LaunchAgents/com.username.apifyscraper.plist
```

4. Logs:

- Stdout: `/Users/kingisaac/Github/Brand-Perception-and-Sentiment-Analysis/services/reddit_data/scraper.log`
- Stderr: `/Users/kingisaac/Github/Brand-Perception-and-Sentiment-Analysis/services/reddit_data/scraper.err`

Notes:
- `StartInterval` is set to 300 seconds (5 minutes). Edit `com.username.apifyscraper.plist` if you need a different interval.
- The plist runs the `run_apify_scraper.sh` wrapper which activates the project's `.venv` and runs `apify_scraper.py`.
- `SUBREDDITS` and other config can be set in the `.env` file or exported in the wrapper.
