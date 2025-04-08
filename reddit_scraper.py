import praw
import pandas as pd
import datetime
import time
import os

reddit = praw.Reddit(
    client_id="ID_GOES_HERE",
    client_secret="SECRET_GOES_HERE",
    user_agent="script:misinformation_analysis:v1.0 (by u/YOUR_USER_NAME)"
)

misinformation_subreddits = [
    "NoNewNormal", "Conspiracy", "DebateVaccines",
    "ChurchOfCOVID", "LockdownSkepticism"
]

factual_subreddits = [
    "Coronavirus", "COVID19", "science", "medicine",
    "Health", "askscience"
]

covid_keywords = [
    "covid", "coronavirus", "pandemic", "vaccine", "pfizer", "moderna", 
    "johnson", "j&j", "astrazeneca", "vaccination", "vax", "vaxx", 
    "mrna", "covax", "sars-cov-2", "immunization", "booster",
    "lockdown", "mandate", "jab", "fauci", "cdc", "who", "masks"
]

def is_covid_related(text):
    if not text:
        return False
    
    text = text.lower()
    return any(keyword in text for keyword in covid_keywords)

def collect_reddit_data(subreddits, category, limit=100):
    all_posts = []
    all_comments = []
    all_users = set()
    
    for subreddit_name in subreddits:
        print(f"Collecting from r/{subreddit_name}...")
        
        try:
            subreddit = reddit.subreddit(subreddit_name)
            
            for submission in subreddit.search("covid OR vaccine", sort="top", time_filter="year", limit=limit):
                post_data = {
                    'id': submission.id,
                    'type': 'submission',
                    'title': submission.title,
                    'author': str(submission.author),
                    'author_id': str(submission.author) if submission.author else "[deleted]",
                    'created_utc': datetime.datetime.fromtimestamp(submission.created_utc),
                    'score': submission.score,
                    'upvote_ratio': submission.upvote_ratio,
                    'num_comments': submission.num_comments,
                    'is_self': submission.is_self,
                    'selftext': submission.selftext if submission.is_self else "",
                    'url': submission.url,
                    'permalink': submission.permalink,
                    'subreddit': subreddit_name,
                    'category': category
                }
                all_posts.append(post_data)
                
                if submission.author:
                    all_users.add(str(submission.author))
                
                submission.comments.replace_more(limit=0)
                
                def process_comments(comment, level=0):
                    if not comment.author:
                        return
                    
                    comment_data = {
                        'id': comment.id,
                        'type': 'comment',
                        'parent_id': comment.parent_id,
                        'submission_id': submission.id,
                        'author': str(comment.author),
                        'author_id': str(comment.author),
                        'created_utc': datetime.datetime.fromtimestamp(comment.created_utc),
                        'score': comment.score,
                        'body': comment.body,
                        'permalink': comment.permalink,
                        'subreddit': subreddit_name,
                        'category': category,
                        'comment_level': level
                    }
                    all_comments.append(comment_data)
                    
                    all_users.add(str(comment.author))
                    
                    for reply in comment.replies:
                        process_comments(reply, level+1)
                
                for comment in submission.comments:
                    process_comments(comment)
                
                print(f"  Processed submission '{submission.title[:30]}...' with {submission.num_comments} comments")
            
            print(f"Collected data from r/{subreddit_name}")
            
        except Exception as e:
            print(f"Error collecting from r/{subreddit_name}: {e}")
        
        os.makedirs("reddit_data", exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if all_posts:
            posts_df = pd.DataFrame(all_posts)
            posts_filename = f"reddit_data/posts_{subreddit_name}_{timestamp}.csv"
            posts_df.to_csv(posts_filename, index=False)
            print(f"Saved {len(all_posts)} posts to {posts_filename}")
        
        if all_comments:
            comments_df = pd.DataFrame(all_comments)
            comments_filename = f"reddit_data/comments_{subreddit_name}_{timestamp}.csv"
            comments_df.to_csv(comments_filename, index=False)
            print(f"Saved {len(all_comments)} comments to {comments_filename}")
        
        time.sleep(2)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    all_content = all_posts + all_comments
    content_df = pd.DataFrame(all_content)
    content_filename = f"reddit_data/all_content_{category}_{timestamp}.csv"
    content_df.to_csv(content_filename, index=False)
    print(f"Saved {len(all_content)} total items to {content_filename}")
    
    users_df = pd.DataFrame(list(all_users), columns=['username'])
    users_filename = f"reddit_data/users_{category}_{timestamp}.csv"
    users_df.to_csv(users_filename, index=False)
    print(f"Saved {len(all_users)} unique users to {users_filename}")
    
    network_edges = []
    
    for comment in all_comments:
        parent_id = comment['parent_id']
        
        parent_author = None
        
        if parent_id.startswith('t3_'):
            post_id = parent_id[3:]
            for post in all_posts:
                if post['id'] == post_id:
                    parent_author = post['author']
                    break
        elif parent_id.startswith('t1_'):
            comment_id = parent_id[3:]
            for c in all_comments:
                if c['id'] == comment_id:
                    parent_author = c['author']
                    break
        
        if parent_author and comment['author'] != parent_author:
            edge = {
                'source': comment['author'],
                'target': parent_author,
                'interaction_type': 'comment_reply',
                'comment_id': comment['id'],
                'parent_id': parent_id,
                'subreddit': comment['subreddit'],
                'category': comment['category'],
                'created_utc': comment['created_utc']
            }
            network_edges.append(edge)
    
    if network_edges:
        edges_df = pd.DataFrame(network_edges)
        edges_filename = f"reddit_data/network_edges_{category}_{timestamp}.csv"
        edges_df.to_csv(edges_filename, index=False)
        print(f"Saved {len(network_edges)} network edges to {edges_filename}")
    
    return all_content, all_users, network_edges

def main():
    print("Starting collection for misinformation subreddits...")
    misinfo_content, misinfo_users, misinfo_edges = collect_reddit_data(
        misinformation_subreddits, "misinformation"
    )
    
    print("Starting collection for factual information subreddits...")
    factual_content, factual_users, factual_edges = collect_reddit_data(
        factual_subreddits, "factual"
    )
    
    print("Data collection complete!")
    print(f"Collected {len(misinfo_content)} items from misinformation subreddits")
    print(f"Collected {len(factual_content)} items from factual subreddits")
    print(f"Identified {len(misinfo_users) + len(factual_users)} unique users")
    print(f"Captured {len(misinfo_edges) + len(factual_edges)} network interactions")
    
if __name__ == "__main__":
    main()
