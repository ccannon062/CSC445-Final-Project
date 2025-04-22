"""
network_metrics_report.py - A standalone script to generate a comprehensive 
network metrics report for COVID-19 misinformation analysis.

This script reads the previously saved network data and outputs a detailed
metrics report to help quantify the differences between misinformation and
factual information networks.
"""

import pandas as pd
import networkx as nx
import numpy as np
import os
import time

# Define paths to the saved network data
MISINFO_EDGES_PATH = 'reddit_data/network_edges_misinformation_20250408_140600.csv'
FACTUAL_EDGES_PATH = 'reddit_data/network_edges_factual_20250408_141247.csv'

def load_networks():
    """Load the network data and construct NetworkX graph objects."""
    print("Loading network edge data...")
    
    misinfo_edges = pd.read_csv(MISINFO_EDGES_PATH)
    factual_edges = pd.read_csv(FACTUAL_EDGES_PATH)
    
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

def generate_network_metrics_report(misinfo_graph, factual_graph, combined_graph, output_path="results/network_metrics_report.txt"):
    """
    Generate a comprehensive report of network metrics for both networks
    and save it to a text file for easy reference.
    """
    start_time = time.time()
    print("Generating network metrics report...")
    
    with open(output_path, 'w') as f:
        # Title
        f.write("=====================================================\n")
        f.write("COVID-19 MISINFORMATION NETWORK ANALYSIS - METRICS REPORT\n")
        f.write("=====================================================\n\n")
        
        f.write(f"Report generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Basic Network Statistics
        f.write("1. BASIC NETWORK STATISTICS\n")
        f.write("---------------------------\n\n")
        
        f.write("Misinformation Network:\n")
        f.write(f"  - Nodes: {misinfo_graph.number_of_nodes()}\n")
        f.write(f"  - Edges: {misinfo_graph.number_of_edges()}\n")
        f.write(f"  - Density: {nx.density(misinfo_graph):.6f}\n")
        
        misinfo_degrees = [d for n, d in misinfo_graph.degree()]
        f.write(f"  - Average Degree: {np.mean(misinfo_degrees):.2f}\n")
        f.write(f"  - Median Degree: {np.median(misinfo_degrees):.2f}\n")
        f.write(f"  - Max Degree: {np.max(misinfo_degrees)}\n")
        
        # In-degree and out-degree for directed graph
        misinfo_in_degrees = [d for n, d in misinfo_graph.in_degree()]
        misinfo_out_degrees = [d for n, d in misinfo_graph.out_degree()]
        f.write(f"  - Average In-Degree: {np.mean(misinfo_in_degrees):.2f}\n")
        f.write(f"  - Average Out-Degree: {np.mean(misinfo_out_degrees):.2f}\n")
        
        f.write("\nFactual Information Network:\n")
        f.write(f"  - Nodes: {factual_graph.number_of_nodes()}\n")
        f.write(f"  - Edges: {factual_graph.number_of_edges()}\n")
        f.write(f"  - Density: {nx.density(factual_graph):.6f}\n")
        
        factual_degrees = [d for n, d in factual_graph.degree()]
        f.write(f"  - Average Degree: {np.mean(factual_degrees):.2f}\n")
        f.write(f"  - Median Degree: {np.median(factual_degrees):.2f}\n")
        f.write(f"  - Max Degree: {np.max(factual_degrees)}\n")
        
        # In-degree and out-degree for directed graph
        factual_in_degrees = [d for n, d in factual_graph.in_degree()]
        factual_out_degrees = [d for n, d in factual_graph.out_degree()]
        f.write(f"  - Average In-Degree: {np.mean(factual_in_degrees):.2f}\n")
        f.write(f"  - Average Out-Degree: {np.mean(factual_out_degrees):.2f}\n")
        
        f.write("\nCombined Network:\n")
        f.write(f"  - Nodes: {combined_graph.number_of_nodes()}\n")
        f.write(f"  - Edges: {combined_graph.number_of_edges()}\n")
        f.write(f"  - Density: {nx.density(combined_graph):.6f}\n\n")
        
        # Network Centralization Metrics
        f.write("2. CENTRALIZATION METRICS\n")
        f.write("-------------------------\n\n")
        
        # PageRank
        f.write("PageRank Statistics:\n")
        print("  Calculating PageRank...")
        misinfo_pagerank = nx.pagerank(misinfo_graph, max_iter=100)
        factual_pagerank = nx.pagerank(factual_graph, max_iter=100)
        
        f.write("  Misinformation Network:\n")
        f.write(f"    - Max PageRank: {max(misinfo_pagerank.values()):.6f}\n")
        f.write(f"    - Average PageRank: {np.mean(list(misinfo_pagerank.values())):.6f}\n")
        f.write("    - Top 10 Users by PageRank:\n")
        
        for i, (user, pr) in enumerate(sorted(misinfo_pagerank.items(), key=lambda x: x[1], reverse=True)[:10], 1):
            f.write(f"      {i}. {user}: {pr:.6f}\n")
        
        f.write("\n  Factual Information Network:\n")
        f.write(f"    - Max PageRank: {max(factual_pagerank.values()):.6f}\n")
        f.write(f"    - Average PageRank: {np.mean(list(factual_pagerank.values())):.6f}\n")
        f.write("    - Top 10 Users by PageRank:\n")
        
        for i, (user, pr) in enumerate(sorted(factual_pagerank.items(), key=lambda x: x[1], reverse=True)[:10], 1):
            f.write(f"      {i}. {user}: {pr:.6f}\n")
        
        f.write("\n")
        
        # Approximate Betweenness Centrality
        f.write("Approximate Betweenness Centrality:\n")
        print("  Calculating approximate betweenness centrality (this may take a while)...")
        
        try:
            # Sample k=500 nodes for approximate betweenness
            misinfo_betweenness = nx.betweenness_centrality(misinfo_graph, k=500, normalized=True)
            factual_betweenness = nx.betweenness_centrality(factual_graph, k=500, normalized=True)
            
            f.write("  Misinformation Network:\n")
            f.write(f"    - Max Betweenness: {max(misinfo_betweenness.values()):.6f}\n")
            f.write(f"    - Average Betweenness: {np.mean(list(misinfo_betweenness.values())):.6f}\n")
            f.write("    - Top 10 Users by Betweenness:\n")
            
            for i, (user, bc) in enumerate(sorted(misinfo_betweenness.items(), key=lambda x: x[1], reverse=True)[:10], 1):
                f.write(f"      {i}. {user}: {bc:.6f}\n")
            
            f.write("\n  Factual Information Network:\n")
            f.write(f"    - Max Betweenness: {max(factual_betweenness.values()):.6f}\n")
            f.write(f"    - Average Betweenness: {np.mean(list(factual_betweenness.values())):.6f}\n")
            f.write("    - Top 10 Users by Betweenness:\n")
            
            for i, (user, bc) in enumerate(sorted(factual_betweenness.items(), key=lambda x: x[1], reverse=True)[:10], 1):
                f.write(f"      {i}. {user}: {bc:.6f}\n")
        except Exception as e:
            f.write(f"  Could not compute betweenness centrality due to: {e}\n")
        
        f.write("\n")
        
        # Connected Components
        f.write("3. CONNECTED COMPONENTS\n")
        f.write("-----------------------\n\n")
        print("  Analyzing connected components...")
        
        misinfo_wcc = list(nx.weakly_connected_components(misinfo_graph))
        factual_wcc = list(nx.weakly_connected_components(factual_graph))
        
        f.write("Misinformation Network:\n")
        f.write(f"  - Number of weakly connected components: {len(misinfo_wcc)}\n")
        f.write(f"  - Size of largest component: {len(max(misinfo_wcc, key=len))}\n")
        f.write(f"  - Percentage of nodes in largest component: {len(max(misinfo_wcc, key=len)) / misinfo_graph.number_of_nodes():.2%}\n")
        
        # Component size distribution
        misinfo_cc_sizes = [len(cc) for cc in misinfo_wcc]
        f.write("  - Component size distribution:\n")
        f.write(f"    * Min: {min(misinfo_cc_sizes)}\n")
        f.write(f"    * 25th percentile: {np.percentile(misinfo_cc_sizes, 25):.1f}\n")
        f.write(f"    * Median: {np.median(misinfo_cc_sizes):.1f}\n")
        f.write(f"    * 75th percentile: {np.percentile(misinfo_cc_sizes, 75):.1f}\n")
        f.write(f"    * Max: {max(misinfo_cc_sizes)}\n")
        
        f.write("\nFactual Information Network:\n")
        f.write(f"  - Number of weakly connected components: {len(factual_wcc)}\n")
        f.write(f"  - Size of largest component: {len(max(factual_wcc, key=len))}\n")
        f.write(f"  - Percentage of nodes in largest component: {len(max(factual_wcc, key=len)) / factual_graph.number_of_nodes():.2%}\n")
        
        # Component size distribution
        factual_cc_sizes = [len(cc) for cc in factual_wcc]
        f.write("  - Component size distribution:\n")
        f.write(f"    * Min: {min(factual_cc_sizes)}\n")
        f.write(f"    * 25th percentile: {np.percentile(factual_cc_sizes, 25):.1f}\n")
        f.write(f"    * Median: {np.median(factual_cc_sizes):.1f}\n")
        f.write(f"    * 75th percentile: {np.percentile(factual_cc_sizes, 75):.1f}\n")
        f.write(f"    * Max: {max(factual_cc_sizes)}\n\n")
        
        # Clustering
        f.write("4. CLUSTERING METRICS\n")
        f.write("---------------------\n\n")
        print("  Calculating clustering coefficients...")
        
        misinfo_undirected = misinfo_graph.to_undirected()
        factual_undirected = factual_graph.to_undirected()
        
        f.write("Misinformation Network:\n")
        f.write(f"  - Average clustering coefficient: {nx.average_clustering(misinfo_undirected):.6f}\n")
        
        f.write("\nFactual Information Network:\n")
        f.write(f"  - Average clustering coefficient: {nx.average_clustering(factual_undirected):.6f}\n\n")
        
        # Path Length Analysis
        f.write("5. PATH LENGTH ANALYSIS\n")
        f.write("----------------------\n\n")
        print("  Analyzing path lengths...")
        
        # Sample nodes from largest components for path length analysis
        misinfo_largest_cc = max(nx.weakly_connected_components(misinfo_graph), key=len)
        factual_largest_cc = max(nx.weakly_connected_components(factual_graph), key=len)
        
        try:
            if len(misinfo_largest_cc) > 1000:
                # Sample 500 nodes for path length calculation
                misinfo_sample = list(misinfo_largest_cc)[:500]
                misinfo_sample_graph = misinfo_graph.subgraph(misinfo_sample).copy()
                avg_path_length_misinfo = nx.average_shortest_path_length(misinfo_sample_graph)
                f.write(f"Misinformation Network (sampled 500 nodes): {avg_path_length_misinfo:.4f}\n")
            else:
                misinfo_largest_graph = misinfo_graph.subgraph(misinfo_largest_cc).copy()
                avg_path_length_misinfo = nx.average_shortest_path_length(misinfo_largest_graph)
                f.write(f"Misinformation Network: {avg_path_length_misinfo:.4f}\n")
        except Exception as e:
            f.write(f"Could not compute average path length for misinformation network: {e}\n")
            
        try:
            if len(factual_largest_cc) > 1000:
                # Sample 500 nodes for path length calculation
                factual_sample = list(factual_largest_cc)[:500]
                factual_sample_graph = factual_graph.subgraph(factual_sample).copy()
                avg_path_length_factual = nx.average_shortest_path_length(factual_sample_graph)
                f.write(f"Factual Network (sampled 500 nodes): {avg_path_length_factual:.4f}\n\n")
            else:
                factual_largest_graph = factual_graph.subgraph(factual_largest_cc).copy()
                avg_path_length_factual = nx.average_shortest_path_length(factual_largest_graph)
                f.write(f"Factual Network: {avg_path_length_factual:.4f}\n\n")
        except Exception as e:
            f.write(f"Could not compute average path length for factual network: {e}\n\n")
        
        # Cross-posting analysis
        f.write("6. CROSS-POSTING ANALYSIS\n")
        f.write("-------------------------\n\n")
        print("  Analyzing cross-posting behavior...")
        
        crossposters = set(misinfo_graph.nodes()).intersection(set(factual_graph.nodes()))
        
        f.write(f"Number of cross-posting users: {len(crossposters)}\n")
        f.write(f"Percentage of all users: {len(crossposters) / combined_graph.number_of_nodes():.2%}\n\n")
        
        # Add top cross-posters by influence
        f.write("Top 10 cross-posters by combined influence (PageRank):\n")
        
        crossposter_data = []
        for user in crossposters:
            crossposter_data.append({
                'user': user,
                'misinfo_pagerank': misinfo_pagerank.get(user, 0),
                'factual_pagerank': factual_pagerank.get(user, 0),
                'total_pagerank': misinfo_pagerank.get(user, 0) + factual_pagerank.get(user, 0)
            })
        
        if crossposter_data:
            sorted_crossposters = sorted(crossposter_data, key=lambda x: x['total_pagerank'], reverse=True)[:10]
            
            for i, data in enumerate(sorted_crossposters, 1):
                f.write(f"  {i}. {data['user']}:\n")
                f.write(f"     - Misinformation PageRank: {data['misinfo_pagerank']:.6f}\n")
                f.write(f"     - Factual PageRank: {data['factual_pagerank']:.6f}\n")
                f.write(f"     - Total PageRank: {data['total_pagerank']:.6f}\n")
                
                # Calculate influence proportion
                misinfo_proportion = data['misinfo_pagerank'] / data['total_pagerank'] if data['total_pagerank'] > 0 else 0
                f.write(f"     - Proportion of influence in misinformation network: {misinfo_proportion:.2%}\n\n")
                
        # Subreddit analysis
        f.write("7. SUBREDDIT PARTICIPATION ANALYSIS\n")
        f.write("----------------------------------\n\n")
        
        # Count edges by subreddit
        misinfo_subreddit_counts = {}
        for _, _, attrs in misinfo_graph.edges(data=True):
            if 'subreddit' in attrs:
                subreddit = attrs['subreddit']
                misinfo_subreddit_counts[subreddit] = misinfo_subreddit_counts.get(subreddit, 0) + 1
                
        factual_subreddit_counts = {}
        for _, _, attrs in factual_graph.edges(data=True):
            if 'subreddit' in attrs:
                subreddit = attrs['subreddit']
                factual_subreddit_counts[subreddit] = factual_subreddit_counts.get(subreddit, 0) + 1
        
        f.write("Misinformation Subreddits Participation:\n")
        for subreddit, count in sorted(misinfo_subreddit_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  - r/{subreddit}: {count} interactions\n")
            
        f.write("\nFactual Subreddits Participation:\n")
        for subreddit, count in sorted(factual_subreddit_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  - r/{subreddit}: {count} interactions\n")
        
        # Execution time
        end_time = time.time()
        f.write(f"\nReport generation completed in {end_time - start_time:.2f} seconds.\n")
    
    print(f"Network metrics report saved to {output_path}")
    return output_path

def main():
    """Main function to execute the metrics report generation."""
    start_time = time.time()
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Load networks
    misinfo_graph, factual_graph, combined_graph = load_networks()
    
    # Generate the metrics report
    report_path = generate_network_metrics_report(misinfo_graph, factual_graph, combined_graph)
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print(f"Report saved to: {report_path}")

if __name__ == "__main__":
    main()