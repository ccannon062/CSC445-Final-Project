import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import Counter

def analyze_crossposters(misinfo_graph, factual_graph, misinfo_content, factual_content):
    print("Analyzing cross-posting users...")
    
    misinfo_users = set(misinfo_graph.nodes())
    factual_users = set(factual_graph.nodes())
    crossposters = misinfo_users.intersection(factual_users)
    
    print(f"Found {len(crossposters)} users who post in both misinformation and factual communities")
    print(f"This represents {len(crossposters)/len(misinfo_users.union(factual_users)):.2%} of all users")
    
    misinfo_pagerank = nx.pagerank(misinfo_graph, max_iter=100)
    factual_pagerank = nx.pagerank(factual_graph, max_iter=100)
    
    misinfo_degree = dict(misinfo_graph.degree())
    factual_degree = dict(factual_graph.degree())
    
    crossposter_data = []
    for user in crossposters:
        crossposter_data.append({
            'user': user,
            'misinfo_pagerank': misinfo_pagerank.get(user, 0),
            'factual_pagerank': factual_pagerank.get(user, 0),
            'misinfo_degree': misinfo_degree.get(user, 0),
            'factual_degree': factual_degree.get(user, 0),
            'total_pagerank': misinfo_pagerank.get(user, 0) + factual_pagerank.get(user, 0),
            'total_degree': misinfo_degree.get(user, 0) + factual_degree.get(user, 0)
        })
    
    if crossposter_data:
        crossposter_df = pd.DataFrame(crossposter_data)
        
        top_crossposters = crossposter_df.sort_values('total_pagerank', ascending=False).head(20)
        
        crossposter_df.to_csv('results/crossposters_analysis.csv', index=False)
        top_crossposters.to_csv('results/top_crossposters.csv', index=False)
        print(f"Crossposters analysis saved to results/crossposters_analysis.csv")
        print(f"Top crossposters saved to results/top_crossposters.csv")
        
        visualize_crossposter_influence(crossposter_df)
        
        subreddit_analysis = analyze_crossposter_subreddits(crossposters, misinfo_content, factual_content)
        
        return {
            'crossposters': list(crossposters),
            'crossposter_df': crossposter_df,
            'top_crossposters': top_crossposters,
            'subreddit_analysis': subreddit_analysis
        }
    else:
        print("No crossposters found for analysis")
        return {
            'crossposters': [],
            'crossposter_df': None,
            'top_crossposters': None,
            'subreddit_analysis': None
        }

def visualize_crossposter_influence(crossposter_df):
    if crossposter_df.empty:
        print("No data for crossposter influence visualization")
        return
    
    plt.figure(figsize=(10, 8))
    plt.scatter(
        crossposter_df['misinfo_pagerank'], 
        crossposter_df['factual_pagerank'],
        alpha=0.7,
        s=crossposter_df['total_degree'] * 3,
        c=crossposter_df['misinfo_pagerank'] / (crossposter_df['misinfo_pagerank'] + crossposter_df['factual_pagerank']),
        cmap='coolwarm'
    )
    
    cbar = plt.colorbar()
    cbar.set_label('Proportion of Influence in Misinformation Network')
    
    top_users = crossposter_df.sort_values('total_pagerank', ascending=False).head(10)
    for _, user in top_users.iterrows():
        plt.annotate(
            user['user'],
            (user['misinfo_pagerank'], user['factual_pagerank']),
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    plt.xlabel('Influence in Misinformation Network (PageRank)')
    plt.ylabel('Influence in Factual Network (PageRank)')
    plt.title('Cross-Posting Users: Influence in Both Networks')
    
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/crossposter_influence.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Crossposter influence visualization saved to results/crossposter_influence.png")
    
    top_users = crossposter_df.sort_values('total_pagerank', ascending=False).head(10)
    
    plt.figure(figsize=(12, 6))
    
    bars = plt.bar(top_users['user'], top_users['misinfo_pagerank'], color='red', label='Misinformation')
    plt.bar(top_users['user'], top_users['factual_pagerank'], bottom=top_users['misinfo_pagerank'], color='blue', label='Factual')
    
    plt.xlabel('User')
    plt.ylabel('PageRank')
    plt.title('Top 10 Cross-Posting Users by Total Influence')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("results/top_crossposters_influence.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Top crossposters visualization saved to results/top_crossposters_influence.png")

def analyze_crossposter_subreddits(crossposters, misinfo_content, factual_content):
    misinfo_crossposter_content = misinfo_content[misinfo_content['author'].isin(crossposters)]
    factual_crossposter_content = factual_content[factual_content['author'].isin(crossposters)]
    
    misinfo_sub_counts = misinfo_crossposter_content['subreddit'].value_counts()
    factual_sub_counts = factual_crossposter_content['subreddit'].value_counts()
    
    sub_pairs = []

    misinfo_by_author = misinfo_crossposter_content.groupby('author')['subreddit'].unique()
    factual_by_author = factual_crossposter_content.groupby('author')['subreddit'].unique()
    
    for author in crossposters:
        if author in misinfo_by_author and author in factual_by_author:
            misinfo_subs = misinfo_by_author[author]
            factual_subs = factual_by_author[author]
            
            for m_sub in misinfo_subs:
                for f_sub in factual_subs:
                    sub_pairs.append((m_sub, f_sub))
    
    pair_counts = Counter(sub_pairs)
    
    visualize_crossposter_subreddits(misinfo_sub_counts, factual_sub_counts, pair_counts)
    
    pair_df = pd.DataFrame([
        {'misinfo_subreddit': m, 'factual_subreddit': f, 'count': count}
        for (m, f), count in pair_counts.most_common()
    ])
    
    if not pair_df.empty:
        pair_df.to_csv('results/subreddit_pairs.csv', index=False)
        print("Subreddit pair analysis saved to results/subreddit_pairs.csv")
    
    return {
        'misinfo_subreddits': misinfo_sub_counts,
        'factual_subreddits': factual_sub_counts,
        'subreddit_pairs': pair_counts
    }

def visualize_crossposter_subreddits(misinfo_counts, factual_counts, pair_counts):
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 1, 1)
    misinfo_df = pd.DataFrame({
        'Subreddit': misinfo_counts.index,
        'Count': misinfo_counts.values
    })
    
    ax1 = sns.barplot(x='Subreddit', y='Count', data=misinfo_df, color='red')
    plt.title('Cross-Poster Participation in Misinformation Subreddits')
    plt.xticks(rotation=45, ha='right')
    for i, v in enumerate(misinfo_df['Count']):
        ax1.text(i, v + 5, str(v), ha='center')
    
    plt.subplot(2, 1, 2)
    factual_df = pd.DataFrame({
        'Subreddit': factual_counts.index,
        'Count': factual_counts.values
    })
    
    ax2 = sns.barplot(x='Subreddit', y='Count', data=factual_df, color='blue')
    plt.title('Cross-Poster Participation in Factual Subreddits')
    plt.xticks(rotation=45, ha='right')
    for i, v in enumerate(factual_df['Count']):
        ax2.text(i, v + 5, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig("results/crossposter_subreddits.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Crossposter subreddit participation saved to results/crossposter_subreddits.png")
    
    if len(pair_counts) > 1:
        top_misinfo = misinfo_counts.head(5).index
        top_factual = factual_counts.head(5).index
        
        matrix = np.zeros((len(top_misinfo), len(top_factual)))
        
        for i, m_sub in enumerate(top_misinfo):
            for j, f_sub in enumerate(top_factual):
                matrix[i, j] = pair_counts.get((m_sub, f_sub), 0)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            matrix,
            annot=True,
            fmt='g',
            xticklabels=top_factual,
            yticklabels=top_misinfo,
            cmap='YlOrRd'
        )
        
        plt.xlabel('Factual Subreddits')
        plt.ylabel('Misinformation Subreddits')
        plt.title('Cross-Posting Between Top Subreddits')
        
        plt.tight_layout()
        plt.savefig("results/subreddit_pair_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Subreddit pair heatmap saved to results/subreddit_pair_heatmap.png")
        
        top_pairs = pd.DataFrame([
            {'Pair': f"{m} → {f}", 'Count': count}
            for (m, f), count in pair_counts.most_common(10)
        ])
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Pair', y='Count', data=top_pairs)
        plt.title('Top 10 Subreddit Cross-Posting Pairs')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig("results/top_subreddit_pairs.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Top subreddit pairs saved to results/top_subreddit_pairs.png")