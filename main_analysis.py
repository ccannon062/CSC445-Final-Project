"""
CSC445 - Social Networks Analytics
COVID-19 Misinformation Network Analysis
Main Analysis Script

This script loads the collected Reddit data, builds the networks, 
and performs the primary analysis for the project.
"""

import pandas as pd
import networkx as nx
import os
import time
from network_metrics import calculate_network_metrics, detect_communities
from visualization import (
    visualize_networks_comparison, 
    visualize_combined_network,
    visualize_community_sizes,
    visualize_subreddit_participation
)
from cross_posting_analysis import analyze_crossposters

#Replace the file names here with your files.
MISINFO_EDGES_PATH = 'reddit_data/network_edges_misinformation_20250408_140600.csv'
FACTUAL_EDGES_PATH = 'reddit_data/network_edges_factual_20250408_141247.csv'
MISINFO_CONTENT_PATH = 'reddit_data/all_content_misinformation_20250408_140600.csv'
FACTUAL_CONTENT_PATH = 'reddit_data/all_content_factual_20250408_141247.csv'

def load_data():
    print("Loading network edge data...")
    misinfo_edges = pd.read_csv(MISINFO_EDGES_PATH)
    factual_edges = pd.read_csv(FACTUAL_EDGES_PATH)
    
    print("Loading content data...")
    misinfo_content = pd.read_csv(MISINFO_CONTENT_PATH)
    factual_content = pd.read_csv(FACTUAL_CONTENT_PATH)
    
    return misinfo_edges, factual_edges, misinfo_content, factual_content

def build_networks(misinfo_edges, factual_edges):
    """Build the directed networks from edge lists"""
    print("Building networks...")
    
    # Create directed graphs
    misinfo_graph = nx.from_pandas_edgelist(
        misinfo_edges, 
        source='source', 
        target='target', 
        edge_attr=['subreddit', 'created_utc'],
        create_using=nx.DiGraph()
    )
    
    factual_graph = nx.from_pandas_edgelist(
        factual_edges, 
        source='source', 
        target='target', 
        edge_attr=['subreddit', 'created_utc'],
        create_using=nx.DiGraph()
    )
    
    all_edges = pd.concat([misinfo_edges, factual_edges])
    combined_graph = nx.from_pandas_edgelist(
        all_edges, 
        source='source', 
        target='target', 
        edge_attr=['subreddit', 'category', 'created_utc'],
        create_using=nx.DiGraph()
    )
    
    print(f"Misinformation network: {misinfo_graph.number_of_nodes()} nodes, {misinfo_graph.number_of_edges()} edges")
    print(f"Factual network: {factual_graph.number_of_nodes()} nodes, {factual_graph.number_of_edges()} edges")
    print(f"Combined network: {combined_graph.number_of_nodes()} nodes, {combined_graph.number_of_edges()} edges")
    
    return misinfo_graph, factual_graph, combined_graph

def main():
    os.makedirs('results', exist_ok=True)
    
    misinfo_edges, factual_edges, misinfo_content, factual_content = load_data()
    
    misinfo_graph, factual_graph, combined_graph = build_networks(misinfo_edges, factual_edges)
    
    print("\nCalculating network metrics...")
    misinfo_metrics, factual_metrics = calculate_network_metrics(misinfo_graph, factual_graph)
    
    metrics_comparison = pd.DataFrame({
        'Metric': list(misinfo_metrics.keys()),
        'Misinformation': list(misinfo_metrics.values()),
        'Factual': list(factual_metrics.values())
    })
    metrics_comparison.to_csv('results/network_metrics_comparison.csv', index=False)
    print("Network metrics saved to results/network_metrics_comparison.csv")
    
    print("\nDetecting communities...")
    misinfo_communities, factual_communities = detect_communities(misinfo_graph, factual_graph)
    
    print("\nAnalyzing cross-posting users...")
    crossposter_results = analyze_crossposters(
        misinfo_graph, 
        factual_graph, 
        misinfo_content, 
        factual_content
    )
    
    print("\nCreating visualizations...")
    visualize_networks_comparison(misinfo_graph, factual_graph)
    visualize_combined_network(combined_graph, misinfo_graph, factual_graph)
    visualize_community_sizes(misinfo_communities, factual_communities)
    visualize_subreddit_participation(misinfo_content, factual_content)
    
    print("\nAnalysis complete! All results saved to the 'results' directory.")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")
