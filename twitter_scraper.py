import asyncio
from apify_client import ApifyClient
import pandas as pd
import time
from datetime import datetime

# DO NOT hardcode your API key in production code
# Instead, use environment variables or a secure configuration file
# api_key = "YOUR_API_KEY_HERE"  # Replace with environment variable

# Define your hashtag groups
misinformation_hashtags = [
    "#vaccinemisinformation", "#vaccinedebate", "#vaccinesafety", 
    "#vaccinesideeffects", "#donttakethevaccine", "#covidvaccinetruth", 
    "#vaccinemandates", "#bigpharma", "#vaccineconspiracy", "#stoptheshots"
]

factual_hashtags = [
    "#COVID19vaccine", "#getvaccinated", "#vaccinatefortheworld", 
    "#vaccineswork", "#COVIDvaccinessavelives", "#publichealth", 
    "#sciencebased", "#trustthescience", "#vaccineequity", "#cdc"
]

# Date range for data collection (COVID vaccine rollout period)
start_date = "2020-12-01"
end_date = "2021-03-31"

async def collect_tweets_for_hashtags(hashtags, category_name, api_key):
    """Collect tweets for a list of hashtags using Apify's Twitter Scraper"""
    client = ApifyClient(api_key)
    
    all_tweets = []
    
    for hashtag in hashtags:
        print(f"Collecting tweets for {hashtag}...")
        
        # Start the Twitter Scraper actor
        run_input = {
            "searchTerms": [hashtag],
            "maxTweets": 5000,  # Adjust as needed
            "dateFrom": start_date,
            "dateTo": end_date,
            "languageFilter": "en",
            "proxyConfiguration": {"useApifyProxy": True},
        }
        
        # Run the actor and wait for it to finish
        run = client.actor("quacker/twitter-scraper").call(run_input=run_input)
        
        # Fetch the actor's output
        dataset = client.dataset(run["defaultDatasetId"])
        tweets = dataset.list_items().items
        
        for tweet in tweets:
            tweet['category'] = category_name
            tweet['search_hashtag'] = hashtag
            all_tweets.append(tweet)
        
        # Be nice to the API and avoid rate limits
        time.sleep(2)
    
    # Save to CSV with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"twitter_data_{category_name}_{timestamp}.csv"
    
    df = pd.DataFrame(all_tweets)
    df.to_csv(filename, index=False)
    print(f"Saved {len(all_tweets)} tweets to {filename}")
    
    return all_tweets

async def main():
    # Get API key securely (from environment or config)
    import os
    api_key = os.environ.get("APIFY_API_KEY")  # Set this in your environment
    
    if not api_key:
        raise ValueError("APIFY_API_KEY not found in environment variables")
    
    # Collect tweets for both hashtag groups
    misinfo_tweets = await collect_tweets_for_hashtags(
        misinformation_hashtags, "misinformation", api_key
    )
    
    factual_tweets = await collect_tweets_for_hashtags(
        factual_hashtags, "factual", api_key
    )
    
    print(f"Total tweets collected: {len(misinfo_tweets) + len(factual_tweets)}")

if __name__ == "__main__":
    asyncio.run(main())
