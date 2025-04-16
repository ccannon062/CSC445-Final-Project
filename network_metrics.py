import networkx as nx
import pandas as pd
import numpy as np
import community as community_louvain
import matplotlib.pyplot as plt

def calculate_network_metrics(misinfo_graph, factual_graph):

    misinfo_metrics = {}
    factual_metrics = {}
    
    misinfo_metrics['nodes'] = misinfo_graph.number_of_nodes()
    misinfo_metrics['edges'] = misinfo_graph.number_of_edges()
    misinfo_metrics['density'] = nx.density(misinfo_graph)
    
    factual_metrics['nodes'] = factual_graph.number_of_nodes()
    factual_metrics['edges'] = factual_graph.number_of_edges()
    factual_metrics['density'] = nx.density(factual_graph)
    
    misinfo_degrees = [d for n, d in misinfo_graph.degree()]
    factual_degrees = [d for n, d in factual_graph.degree()]
    
    misinfo_metrics['avg_degree'] = np.mean(misinfo_degrees)
    factual_metrics['avg_degree'] = np.mean(factual_degrees)
    
    print("Calculating PageRank...")
    misinfo_pagerank = nx.pagerank(misinfo_graph, max_iter=100)
    factual_pagerank = nx.pagerank(factual_graph, max_iter=100)
    
    misinfo_metrics['max_pagerank'] = max(misinfo_pagerank.values())
    factual_metrics['max_pagerank'] = max(factual_pagerank.values())
    
    top_misinfo_users = sorted(misinfo_pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
    top_factual_users = sorted(factual_pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
    
    top_users_df = pd.DataFrame({
        'Rank': list(range(1, 11)),
        'Misinfo_User': [user for user, _ in top_misinfo_users],
        'Misinfo_PageRank': [score for _, score in top_misinfo_users],
        'Factual_User': [user for user, _ in top_factual_users],
        'Factual_PageRank': [score for _, score in top_factual_users]
    })
    top_users_df.to_csv('results/top_influential_users.csv', index=False)
    print("Top influential users saved to results/top_influential_users.csv")
    
    print("Calculating approximate betweenness centrality (this may take a while)...")
    misinfo_betweenness = nx.betweenness_centrality(misinfo_graph, k=500, normalized=True)
    factual_betweenness = nx.betweenness_centrality(factual_graph, k=500, normalized=True)
    
    misinfo_metrics['max_betweenness'] = max(misinfo_betweenness.values())
    factual_metrics['max_betweenness'] = max(factual_betweenness.values())
    
    top_misinfo_bridges = sorted(misinfo_betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
    top_factual_bridges = sorted(factual_betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
    
    top_bridges_df = pd.DataFrame({
        'Rank': list(range(1, 11)),
        'Misinfo_User': [user for user, _ in top_misinfo_bridges],
        'Misinfo_Betweenness': [score for _, score in top_misinfo_bridges],
        'Factual_User': [user for user, _ in top_factual_bridges],
        'Factual_Betweenness': [score for _, score in top_factual_bridges]
    })
    top_bridges_df.to_csv('results/top_bridge_users.csv', index=False)
    print("Top bridge users saved to results/top_bridge_users.csv")

    print("Calculating clustering coefficients...")
    misinfo_undirected = misinfo_graph.to_undirected()
    factual_undirected = factual_graph.to_undirected()
    
    misinfo_metrics['clustering_coefficient'] = nx.average_clustering(misinfo_undirected)
    factual_metrics['clustering_coefficient'] = nx.average_clustering(factual_undirected)
    
    misinfo_largest_wcc = max(nx.weakly_connected_components(misinfo_graph), key=len)
    factual_largest_wcc = max(nx.weakly_connected_components(factual_graph), key=len)
    
    misinfo_metrics['largest_component_size'] = len(misinfo_largest_wcc)
    factual_metrics['largest_component_size'] = len(factual_largest_wcc)
    
    misinfo_metrics['largest_component_percentage'] = len(misinfo_largest_wcc) / misinfo_graph.number_of_nodes()
    factual_metrics['largest_component_percentage'] = len(factual_largest_wcc) / factual_graph.number_of_nodes()
    
    try:
        if len(misinfo_largest_wcc) < 1000:
            misinfo_largest_graph = misinfo_graph.subgraph(misinfo_largest_wcc).copy()
            misinfo_metrics['avg_path_length'] = nx.average_shortest_path_length(misinfo_largest_graph)
        else:
            sample_nodes = list(misinfo_largest_wcc)[:500]
            misinfo_largest_graph = misinfo_graph.subgraph(sample_nodes).copy()
            misinfo_metrics['avg_path_length'] = nx.average_shortest_path_length(misinfo_largest_graph)
            misinfo_metrics['avg_path_length_note'] = 'Estimated from sample of 500 nodes'
    except:
        misinfo_metrics['avg_path_length'] = None
        misinfo_metrics['avg_path_length_note'] = 'Could not compute'
        
    try:
        if len(factual_largest_wcc) < 1000:
            factual_largest_graph = factual_graph.subgraph(factual_largest_wcc).copy()
            factual_metrics['avg_path_length'] = nx.average_shortest_path_length(factual_largest_graph)
        else:
            sample_nodes = list(factual_largest_wcc)[:500]
            factual_largest_graph = factual_graph.subgraph(sample_nodes).copy()
            factual_metrics['avg_path_length'] = nx.average_shortest_path_length(factual_largest_graph)
            factual_metrics['avg_path_length_note'] = 'Estimated from sample of 500 nodes'
    except:
        factual_metrics['avg_path_length'] = None
        factual_metrics['avg_path_length_note'] = 'Could not compute'
        
    return misinfo_metrics, factual_metrics

def detect_communities(misinfo_graph, factual_graph):
    print("Detecting communities using Louvain algorithm...")
    
    misinfo_undirected = misinfo_graph.to_undirected()
    factual_undirected = factual_graph.to_undirected()
    
    misinfo_communities = community_louvain.best_partition(misinfo_undirected)
    factual_communities = community_louvain.best_partition(factual_undirected)
    
    misinfo_community_sizes = {}
    for node, community_id in misinfo_communities.items():
        if community_id not in misinfo_community_sizes:
            misinfo_community_sizes[community_id] = 0
        misinfo_community_sizes[community_id] += 1
    
    factual_community_sizes = {}
    for node, community_id in factual_communities.items():
        if community_id not in factual_community_sizes:
            factual_community_sizes[community_id] = 0
        factual_community_sizes[community_id] += 1
    
    print(f"Misinformation communities: {len(misinfo_community_sizes)}")
    print(f"Factual communities: {len(factual_community_sizes)}")
    
    misinfo_community_df = pd.DataFrame(misinfo_communities.items(), columns=['User', 'Community_ID'])
    misinfo_community_df.to_csv('results/misinfo_communities.csv', index=False)
    
    factual_community_df = pd.DataFrame(factual_communities.items(), columns=['User', 'Community_ID'])
    factual_community_df.to_csv('results/factual_communities.csv', index=False)
    
    misinfo_sizes_df = pd.DataFrame(misinfo_community_sizes.items(), columns=['Community_ID', 'Size'])
    misinfo_sizes_df = misinfo_sizes_df.sort_values('Size', ascending=False)
    misinfo_sizes_df.to_csv('results/misinfo_community_sizes.csv', index=False)
    
    factual_sizes_df = pd.DataFrame(factual_community_sizes.items(), columns=['Community_ID', 'Size'])
    factual_sizes_df = factual_sizes_df.sort_values('Size', ascending=False)
    factual_sizes_df.to_csv('results/factual_community_sizes.csv', index=False)
    
    print("Community detection results saved to results directory")
    
    return misinfo_communities, factual_communities