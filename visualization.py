import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter

def visualize_networks_comparison(misinfo_graph, factual_graph):
    misinfo_largest_cc = max(nx.weakly_connected_components(misinfo_graph), key=len)
    factual_largest_cc = max(nx.weakly_connected_components(factual_graph), key=len)
    
    if len(misinfo_largest_cc) > 1000:
        misinfo_sample = list(misinfo_largest_cc)[:1000]
        misinfo_viz_graph = misinfo_graph.subgraph(misinfo_sample).copy()
        sample_note_misinfo = f" (showing 1000 of {len(misinfo_largest_cc)} nodes)"
    else:
        misinfo_viz_graph = misinfo_graph.subgraph(misinfo_largest_cc).copy()
        sample_note_misinfo = ""
        
    if len(factual_largest_cc) > 1000:
        factual_sample = list(factual_largest_cc)[:1000]
        factual_viz_graph = factual_graph.subgraph(factual_sample).copy()
        sample_note_factual = f" (showing 1000 of {len(factual_largest_cc)} nodes)"
    else:
        factual_viz_graph = factual_graph.subgraph(factual_largest_cc).copy()
        sample_note_factual = ""
    
    plt.figure(figsize=(18, 9))
    
    plt.subplot(1, 2, 1)
    pos_misinfo = nx.spring_layout(misinfo_viz_graph, seed=42)
    node_sizes = [misinfo_graph.degree(node) * 3 for node in misinfo_viz_graph.nodes()]
    nx.draw_networkx(
        misinfo_viz_graph,
        pos=pos_misinfo,
        with_labels=False,
        node_size=node_sizes,
        node_color='red',
        alpha=0.7,
        edge_color='lightgray',
        arrows=False
    )
    plt.title(f"Misinformation Network{sample_note_misinfo}")
    
    plt.subplot(1, 2, 2)
    pos_factual = nx.spring_layout(factual_viz_graph, seed=42)
    node_sizes = [factual_graph.degree(node) * 3 for node in factual_viz_graph.nodes()]
    nx.draw_networkx(
        factual_viz_graph,
        pos=pos_factual,
        with_labels=False,
        node_size=node_sizes,
        node_color='blue',
        alpha=0.7,
        edge_color='lightgray',
        arrows=False
    )
    plt.title(f"Factual Information Network{sample_note_factual}")
    
    plt.tight_layout()
    plt.savefig("results/network_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Network comparison visualization saved to results/network_comparison.png")

def visualize_combined_network(combined_graph, misinfo_graph, factual_graph):
    largest_cc = max(nx.weakly_connected_components(combined_graph), key=len)
    
    misinfo_nodes = set(misinfo_graph.nodes())
    factual_nodes = set(factual_graph.nodes())
    crossposters = misinfo_nodes.intersection(factual_nodes)
    crossposters_in_cc = list(crossposters.intersection(largest_cc))
    
    sample_size = 1000
    
    if len(largest_cc) > sample_size:
        pagerank = nx.pagerank(combined_graph, max_iter=100)
        
        other_nodes = list(set(largest_cc) - set(crossposters_in_cc))
        
        other_nodes_sorted = sorted(other_nodes, key=lambda n: pagerank.get(n, 0), reverse=True)
        
        remaining_needed = min(sample_size - len(crossposters_in_cc), len(other_nodes))
        sample_nodes = list(crossposters_in_cc) + other_nodes_sorted[:remaining_needed]
        
        sample_note = f" (showing {len(sample_nodes)} of {len(largest_cc)} nodes, prioritizing cross-posters)"
    else:
        sample_nodes = list(largest_cc)
        sample_note = ""
    
    viz_graph = combined_graph.subgraph(sample_nodes).copy()
    
    plt.figure(figsize=(16, 16), dpi=300)
    
    G_undirected = viz_graph.to_undirected()
    
    try:
        import community as community_louvain
        communities = community_louvain.best_partition(G_undirected)
    except ImportError:
        from networkx.algorithms import community
        communities_generator = community.greedy_modularity_communities(G_undirected)
        communities = {}
        for i, comm in enumerate(communities_generator):
            for node in comm:
                communities[node] = i
    
    unique_communities = set(communities.values())
    num_communities = len(unique_communities)
    community_colors = plt.cm.tab20(np.linspace(0, 1, num_communities))
    community_color_map = {comm_id: community_colors[i] for i, comm_id in enumerate(unique_communities)}
    
    try:
        from scipy.spatial import ConvexHull
        pos = nx.fruchterman_reingold_layout(viz_graph, k=0.5, iterations=100, seed=42)
        
        for comm_id in unique_communities:
            comm_nodes = [node for node, c_id in communities.items() if c_id == comm_id]
            if len(comm_nodes) > 3:
                comm_pos = {node: pos[node] for node in comm_nodes}
                x_coords = [p[0] for p in comm_pos.values()]
                y_coords = [p[1] for p in comm_pos.values()]
                if len(x_coords) > 2:
                    hull = ConvexHull(np.column_stack([x_coords, y_coords]))
                    hull_points = [np.array([x_coords[i], y_coords[i]]) for i in hull.vertices]
                    hull_points.append(hull_points[0])
                    x_hull = [p[0] for p in hull_points]
                    y_hull = [p[1] for p in hull_points]
                    plt.fill(x_hull, y_hull, color=community_color_map[comm_id], alpha=0.1)
    except ImportError:
        pos = nx.fruchterman_reingold_layout(viz_graph, k=0.5, iterations=100, seed=42)
        print("Warning: scipy not installed, skipping community highlighting")
    
    nx.draw_networkx_edges(
        viz_graph,
        pos=pos,
        alpha=0.3,
        width=0.7,
        arrowsize=10,
        arrowstyle='->',
        edge_color='gray'
    )
    
    node_colors = []
    node_sizes = []
    
    for node in viz_graph.nodes():
        if node in misinfo_nodes and node in factual_nodes:
            color = 'purple'
            size = min(300, viz_graph.degree(node) * 3)
        elif node in misinfo_nodes:
            color = 'red'
            size = min(150, viz_graph.degree(node) * 1.5)
        else:
            color = 'blue'
            size = min(150, viz_graph.degree(node) * 1.5)
        
        node_colors.append(color)
        node_sizes.append(size)
    
    nx.draw_networkx_nodes(
        viz_graph,
        pos=pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.8
    )
    
    pagerank = nx.pagerank(viz_graph, max_iter=100)
    top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
    
    node_labels = {node: node for node, _ in top_nodes}
    
    nx.draw_networkx_labels(
        viz_graph,
        pos=pos,
        labels=node_labels,
        font_size=10,
        font_weight='bold',
        font_color='black',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
    )
    
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Misinformation User'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Factual Information User'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Cross-poster'),
        plt.Line2D([0], [0], color='gray', lw=1, marker='>', markersize=5, label='Information Flow Direction')
    ]
    
    plt.legend(handles=legend_elements, loc='lower right', fontsize=12)
    
    plt.title(f"Combined COVID-19 Information Network{sample_note}", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    plt.savefig("results/enhanced_combined_network.png", dpi=300, bbox_inches='tight')
    print("Enhanced network visualization saved to results/enhanced_combined_network.png")
    plt.close()

def visualize_community_sizes(misinfo_communities, factual_communities):
    misinfo_sizes = Counter(misinfo_communities.values())
    factual_sizes = Counter(factual_communities.values())
    
    misinfo_df = pd.DataFrame({
        'Community': list(misinfo_sizes.keys()),
        'Size': list(misinfo_sizes.values())
    }).sort_values('Size', ascending=False)
    
    factual_df = pd.DataFrame({
        'Community': list(factual_sizes.keys()),
        'Size': list(factual_sizes.values())
    }).sort_values('Size', ascending=False)
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    top_misinfo = misinfo_df.head(10)
    sns.barplot(x='Community', y='Size', data=top_misinfo, color='red')
    plt.title('Top 10 Misinformation Communities by Size')
    plt.yscale('log')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 1, 2)
    top_factual = factual_df.head(10)
    sns.barplot(x='Community', y='Size', data=top_factual, color='blue')
    plt.title('Top 10 Factual Communities by Size')
    plt.yscale('log')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("results/community_sizes.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Community size visualization saved to results/community_sizes.png")
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    misinfo_sizes_list = list(misinfo_sizes.values())
    plt.hist(misinfo_sizes_list, bins=30, color='red', alpha=0.7)
    plt.title('Misinformation Community Size Distribution')
    plt.xlabel('Community Size')
    plt.ylabel('Frequency')
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    factual_sizes_list = list(factual_sizes.values())
    plt.hist(factual_sizes_list, bins=30, color='blue', alpha=0.7)
    plt.title('Factual Community Size Distribution')
    plt.xlabel('Community Size')
    plt.ylabel('Frequency')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig("results/community_size_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Community size distribution saved to results/community_size_distribution.png")

def visualize_subreddit_participation(misinfo_content, factual_content):
    misinfo_counts = misinfo_content['subreddit'].value_counts()
    factual_counts = factual_content['subreddit'].value_counts()
    
    misinfo_df = pd.DataFrame({
        'Subreddit': misinfo_counts.index,
        'Count': misinfo_counts.values
    })
    
    factual_df = pd.DataFrame({
        'Subreddit': factual_counts.index,
        'Count': factual_counts.values
    })
    
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 1, 1)
    ax1 = sns.barplot(x='Subreddit', y='Count', data=misinfo_df, color='red')
    plt.title('Participation in Misinformation Subreddits')
    plt.xticks(rotation=45)
    for i, v in enumerate(misinfo_df['Count']):
        ax1.text(i, v + 10, str(v), ha='center')
    
    plt.subplot(2, 1, 2)
    ax2 = sns.barplot(x='Subreddit', y='Count', data=factual_df, color='blue')
    plt.title('Participation in Factual Subreddits')
    plt.xticks(rotation=45)
    for i, v in enumerate(factual_df['Count']):
        ax2.text(i, v + 10, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig("results/subreddit_participation.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Subreddit participation visualization saved to results/subreddit_participation.png")
    
    plt.figure(figsize=(16, 8))
    
    plt.subplot(1, 2, 1)
    plt.pie(misinfo_counts.values, labels=misinfo_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Content in Misinformation Subreddits')
    
    plt.subplot(1, 2, 2)
    plt.pie(factual_counts.values, labels=factual_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Content in Factual Subreddits')
    
    plt.tight_layout()
    plt.savefig("results/subreddit_distribution.png", dpi=300, bbox_inches='tight')
    plt