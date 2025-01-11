import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import re
from hiveplotlib import hive_plot_n_axes
from hiveplotlib.converters import networkx_to_nodes_edges
from hiveplotlib.viz import hive_plot_viz
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d

def calculate_string_attraction(pos1, pos2, rest_length=0.3, k=0.5):
    """Calculate spring-like attraction force between two points with stronger bundling"""
    diff = np.array(pos2) - np.array(pos1)
    distance = np.linalg.norm(diff)
    # Increased force constant (k) and reduced rest length for tighter bundling
    force = k * (distance - rest_length)
    # Add exponential component for stronger bundling at larger distances
    force *= (1 + 0.2 * np.exp(distance - 1))
    # Normalize and scale
    return (diff / distance) * force if distance > 0 else np.zeros(2)

def get_curved_path(start, end, tension=1.2):
    """Generate straight radial paths from center"""
    center = np.array([0, 0])
    start = np.array(start)
    end = np.array(end)
    
    # Calculate angles
    start_angle = np.arctan2(start[1], start[0])
    end_angle = np.arctan2(end[1], end[0])
    start_dist = np.linalg.norm(start)
    end_dist = np.linalg.norm(end)
    
    # Create points along path
    t = np.linspace(0, 1, 100)
    
    # First segment: from center to midpoint of target angle
    t1 = t[t <= 0.5]
    mid_angle = (start_angle + end_angle) / 2
    r1 = np.interp(t1, [0, 0.5], [0, end_dist])
    x1 = r1 * np.cos(mid_angle)
    y1 = r1 * np.sin(mid_angle)
    
    # Second segment: from midpoint to end
    t2 = t[t > 0.5]
    angle_interp = np.interp(t2, [0.5, 1], [mid_angle, end_angle])
    r2 = np.interp(t2, [0.5, 1], [end_dist, end_dist])
    x2 = r2 * np.cos(angle_interp)
    y2 = r2 * np.sin(angle_interp)
    
    # Combine segments
    curve_x = np.concatenate([x1, x2])
    curve_y = np.concatenate([y1, y2])
    
    return curve_x, curve_y

def get_curved_path_with_gradient(start, end, tension=1.2):
    """Generate straight radial paths from center with color gradient"""
    center = np.array([0, 0])
    start = np.array(start)
    end = np.array(end)
    
    # Calculate angles
    start_angle = np.arctan2(start[1], start[0])
    end_angle = np.arctan2(end[1], end[0])
    end_dist = np.linalg.norm(end)
    
    # Create points along path with more segments for smoother gradient
    num_segments = 50
    t = np.linspace(0, 1, num_segments)
    
    # Generate path coordinates
    mid_angle = (start_angle + end_angle) / 2
    r = np.interp(t, [0, 1], [0, end_dist])
    angle = np.interp(t, [0, 1], [mid_angle, end_angle])
    
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    
    # Create color gradient from red to blue
    colors = np.zeros((len(t), 4))
    colors[:, 0] = np.interp(t, [0, 1], [1, 0])  # Red component
    colors[:, 2] = np.interp(t, [0, 1], [0, 1])  # Blue component
    colors[:, 3] = 0.3  # Alpha
    
    return x, y, colors

def connection_bundle(from_nodes, to_nodes, pos, alpha=0.2, color='skyblue', width=0.9, tension=1):
    """Create connection bundle visualization"""
    for start, end in zip(from_nodes, to_nodes):
        start_pos = pos[start]
        end_pos = pos[end]
        curve_x, curve_y = get_curved_path(start_pos, end_pos, tension)
        plt.plot(curve_x, curve_y, color=color, alpha=alpha, linewidth=width)

# Load and process raw data
df = pd.read_csv('IMGPR/IMGPR_plasmid_data.tsv', sep='\t')

# Create direct connections from raw data
G = nx.Graph()

# Process each row directly
for _, row in df.iterrows():
    if isinstance(row['arg_genes'], str) and isinstance(row['host_taxonomy'], str):
        # Extract genus
        genus_match = re.search(r'g__([^;]+)', row['host_taxonomy'])
        genus = genus_match.group(1) if genus_match else 'unclassified'
        
        # Process ARGs
        args = row['arg_genes'].split(';') if ';' in row['arg_genes'] else [row['arg_genes']]
        ecosystem = row['ecosystem'] if isinstance(row['ecosystem'], str) else 'unknown'
        
        # Add edges for each ARG
        for arg in args:
            if arg.strip():  # only process non-empty ARGs
                G.add_edge(f"arg_{arg.strip()}", f"genus_{genus.strip()}")
                G.add_edge(f"genus_{genus.strip()}", f"eco_{ecosystem.strip()}")
                G.add_edge(f"eco_{ecosystem.strip()}", f"arg_{arg.strip()}")
                # Add direct environment-taxa connections
                G.add_edge(f"eco_{ecosystem.strip()}", f"genus_{genus.strip()}")

# Get all nodes by type
arg_nodes = sorted([n for n in G.nodes() if n.startswith('arg_')], 
                  key=lambda x: -G.degree[x])
genus_nodes = sorted([n for n in G.nodes() if n.startswith('genus_')], 
                    key=lambda x: -G.degree[x])
eco_nodes = sorted([n for n in G.nodes() if n.startswith('eco_')], 
                  key=lambda x: -G.degree[x])

print("\nRaw data node counts:")
print(f"ARG nodes: {len(arg_nodes)}")
print(f"Genus nodes: {len(genus_nodes)}")
print(f"Ecosystem nodes: {len(eco_nodes)}")
print(f"Total edges: {G.number_of_edges()}")

# Create both visualizations
# 1. Hive Plot
nodes, edges = networkx_to_nodes_edges(G)

# Add node data for hive plot
for node in nodes:
    prefix = node.unique_id.split('_')[0]
    node.add_data(data={
        "type": prefix,
        "degree": len(G[node.unique_id])
    })

# Create hive plot
hp = hive_plot_n_axes(
    node_list=nodes,
    edges=edges,
    axes_assignments=[arg_nodes, genus_nodes, eco_nodes],
    sorting_variables=["degree"] * 3,
    axes_names=["ARGs", "Genus", "Ecosystem"],
    all_edge_kwargs={"lw": 0.5, "alpha": 0.3},
    repeat_axes=[False] * 3,
    orient_angle=0,
)

# Customize hive plot edge colors
hp.add_edge_kwargs(axis_id_1="ARGs", axis_id_2="Genus", color="orange")
hp.add_edge_kwargs(axis_id_1="Genus", axis_id_2="Ecosystem", color="green")
hp.add_edge_kwargs(axis_id_1="Ecosystem", axis_id_2="ARGs", color="blue")

# Create and save hive plot
hp_fig, hp_ax = hive_plot_viz(hp)
hp_fig.suptitle("ARGs-Taxonomy-Ecosystem Network\nHive Plot", fontsize=18, y=1.1)
hp_fig.savefig('hive_plot.png', 
              format='png', 
              bbox_inches='tight', 
              dpi=300,
              facecolor='white',
              edgecolor='none')
plt.close(hp_fig)

# Add pathogenic species dictionary before genus_groups
pathogenic_species = {
    'Escherichia',
    'Salmonella',
    'Klebsiella',
    'Pseudomonas',
    'Staphylococcus',
    'Streptococcus',
    'Enterococcus',
    'Acinetobacter'
}

# Add after pathogenic_species dictionary
genus_groups = {
    'Enterobacteriaceae': ['Escherichia', 'Klebsiella', 'Salmonella', 'Enterobacter', 'Citrobacter'],
    'Non-fermenting': ['Acinetobacter', 'Pseudomonas'],
    'Gram-positive': ['Staphylococcus', 'Streptococcus', 'Enterococcus'],
    'Other': []  # will contain all other genera
}

# Update grouped_circular_layout function
def grouped_circular_layout(G, node_groups, genus_groups, pathogenic_species):
    """Create circular layout with nodes grouped by type and similar genera together"""
    pos = {}
    
    # Calculate total nodes for spacing
    total_nodes = sum(len(group) for group in node_groups)
    angle_ratio = 2.0 * np.pi / total_nodes
    curr_pos = 0
    
    # Position ARGs
    arg_group = node_groups[0]
    for node in arg_group:
        angle = curr_pos * angle_ratio
        pos[node] = np.array([np.cos(angle), np.sin(angle)])
        curr_pos += 1
    
    # Position genera by groups
    genus_nodes = node_groups[1]
    grouped_genera = {k: [] for k in genus_groups.keys()}
    
    # Sort genera into their groups
    for node in genus_nodes:
        genus = node.split('_')[1]
        placed = False
        for group_name, members in genus_groups.items():
            if genus in members:
                grouped_genera[group_name].append(node)
                placed = True
                break
        if not placed:
            grouped_genera['Other'].append(node)
    
    # Position each genus group
    for group_name, nodes in grouped_genera.items():
        nodes.sort(key=lambda x: x.split('_')[1])  # sort within group
        for node in nodes:
            angle = curr_pos * angle_ratio
            pos[node] = np.array([np.cos(angle), np.sin(angle)])
            curr_pos += 1
    
    # Position ecosystems by categories
    eco_nodes = node_groups[2]
    grouped_ecosystems = {
        'Host-associated': [],
        'Engineered': [],
        'Environmental': [],
        'unknown': []
    }
    
    # Sort ecosystems into their groups
    for node in eco_nodes:
        eco_name = node.split('_')[1]
        placed = False
        for category, keywords in ecosystem_categories.items():
            if any(keyword.lower() in eco_name.lower() for keyword in keywords):
                grouped_ecosystems[category].append(node)
                placed = True
                break
        if not placed:
            grouped_ecosystems['unknown'].append(node)
    
    # Position each ecosystem group
    for category, nodes in grouped_ecosystems.items():
        nodes.sort(key=lambda x: x.split('_')[1])  # sort within category
        for node in nodes:
            angle = curr_pos * angle_ratio
            pos[node] = np.array([np.cos(angle), np.sin(angle)])
            curr_pos += 1
    
    return pos

# 2. Edge Bundle Plot
# Update node colors with simplified scheme
node_colors = {
    'arg_': '#FFA500',  # orange for ARGs
    'taxa_': '#95a5a6',  # grey for all taxa
    'eco_Host-associated': '#ff7f0e',    # orange
    'eco_Engineered': '#2ca02c',         # green
    'eco_Environmental': '#1f77b4',       # blue
    'eco_unknown': '#7f7f7f'             # gray
}

# Add after node_colors definition
ecosystem_categories = {
    'Host-associated': ['Host-associated', 'Arthropoda', 'Plants', 'Mammals', 'Human'],
    'Engineered': ['Engineered', 'Wastewater', 'Bioreactor'],
    'Environmental': ['Environmental', 'Aquatic', 'Terrestrial', 'Marine', 'Freshwater', 'Soil']
}

# ...existing code...

# Create visualization
fig = plt.figure(figsize=(15, 15))
fig.patch.set_facecolor('white')

# Create subgraph with all nodes
G_sub = G

# Create grouped circular layout
pos = grouped_circular_layout(G_sub, [arg_nodes, genus_nodes, eco_nodes], genus_groups, pathogenic_species)

# Draw edges with bundling first (so they're behind nodes)
# First draw ARG connections
for start in arg_nodes:
    for end in G.neighbors(start):
        if end in genus_nodes or end in eco_nodes:
            curve_x, curve_y, colors = get_curved_path_with_gradient(pos[start], pos[end])
            # Draw gradient line segment by segment
            for i in range(len(curve_x)-1):
                plt.plot(curve_x[i:i+2], curve_y[i:i+2], 
                        color=colors[i],
                        linewidth=0.3)

# Then draw environment-taxa connections
for eco_node in eco_nodes:
    eco_name = eco_node.split('_')[1]
    eco_category = 'unknown'
    for category, keywords in ecosystem_categories.items():
        if any(keyword.lower() in eco_name.lower() for keyword in keywords):
            eco_category = category
            break
    
    # Connect to all neighbor genera
    for genus_node in G.neighbors(eco_node):
        if genus_node in genus_nodes:  # ensure it's a genus node
            curve_x, curve_y, colors = get_curved_path_with_gradient(pos[eco_node], pos[genus_node])
            # Draw gradient line segment by segment
            for i in range(len(curve_x)-1):
                plt.plot(curve_x[i:i+2], curve_y[i:i+2], 
                        color=colors[i],
                        linewidth=0.2)

# Draw nodes with type-specific colors
for node in G_sub.nodes():
    if node.startswith('genus_'):
        genus = node.split('_')[1]
        # Use red for pathogenic species, grey for others
        color = '#e74c3c' if genus in pathogenic_species else node_colors['taxa_']
        nx.draw_networkx_nodes(G_sub, pos, 
                             nodelist=[node],
                             node_color=color,
                             node_size=100,
                             edgecolors='black',
                             linewidths=0.5)
    elif node.startswith('eco_'):
        eco_name = node.split('_')[1]
        eco_category = 'unknown'
        for category, keywords in ecosystem_categories.items():
            if any(keyword.lower() in eco_name.lower() for keyword in keywords):
                eco_category = category
                break
        color = node_colors[f'eco_{eco_category}']
        nx.draw_networkx_nodes(G_sub, pos,
                             nodelist=[node],
                             node_color=color,
                             node_size=100,
                             edgecolors='black',
                             linewidths=0.5)
    else:
        # Draw ARGs
        color = node_colors['arg_']
        nx.draw_networkx_nodes(G_sub, pos,
                             nodelist=[node],
                             node_color=color,
                             node_size=100,
                             edgecolors='black',
                             linewidths=0.5)

# ...existing code...

# Before drawing the phylogenetic arcs, extract taxonomy from data
taxonomy_hierarchy = {}
taxonomy_colors = {}

# Process taxonomy from the data
for _, row in df.iterrows():
    if isinstance(row['host_taxonomy'], str):
        taxa = row['host_taxonomy'].split(';')
        current_dict = taxonomy_hierarchy
        
        for taxon in taxa:
            if not taxon:
                continue
            level, name = taxon.split('__')
            if name:  # only process non-empty names
                # Create path in hierarchy
                if level not in current_dict:
                    current_dict[level] = {'children': {}, 'name': name}
                current_dict = current_dict[level]['children']
                
                # Assign colors if not already assigned
                if name not in taxonomy_colors:
                    if level == 'p':  # phylum level
                        taxonomy_colors[name] = plt.cm.Set3(len(taxonomy_colors) % 12)

# Define radii for each taxonomic level
radii = {
    'd': 1.5,  # domain
    'p': 1.4,  # phylum
    'c': 1.3,  # class
    'o': 1.2,  # order
    'f': 1.1,  # family
    'g': 1.0   # genus
}

def draw_phylo_arc(start_angle, end_angle, radius=1, color='gray', alpha=0.2, linewidth=1):
    """Draw an arc for phylogenetic visualization"""
    theta = np.linspace(start_angle, end_angle, 100)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    plt.plot(x, y, color=color, alpha=alpha, linewidth=linewidth)

# Modify the draw_taxonomy_arcs function to remove labels
def draw_taxonomy_arcs(node_angles, taxonomy_dict, level='d', parent_color=None, min_angle=0, max_angle=2*np.pi):
    """Recursively draw taxonomy arcs"""
    if not taxonomy_dict:
        return
    
    for taxa_level, taxa_info in taxonomy_dict.items():
        if taxa_level == 'children':
            continue
            
        # Get all genus angles that belong to this taxon
        taxa_angles = []
        def collect_genus_angles(tax_dict):
            if not tax_dict:
                return
            if 'g' in tax_dict:
                genus = tax_dict['g']['name']
                if genus in node_angles:
                    taxa_angles.append(node_angles[genus])
            for k, v in tax_dict.items():
                if k != 'name':
                    collect_genus_angles(v['children'])
        
        collect_genus_angles(taxa_info['children'])
        
        if taxa_angles:
            min_taxa = min(taxa_angles)
            max_taxa = max(taxa_angles)
            
            # Use parent color with different alpha, or get new color for phyla
            color = parent_color
            if level == 'p':
                color = taxonomy_colors[taxa_info['name']]
            
            # Draw arc without using ax parameter
            draw_phylo_arc(min_taxa, max_taxa,
                          radius=radii[level],
                          color=color,
                          alpha=0.2 + (0.1 * list(radii.keys()).index(level)),
                          linewidth=2)
            
            # Recursively draw children
            draw_taxonomy_arcs(node_angles, 
                             taxa_info['children'],
                             level=list(radii.keys())[list(radii.keys()).index(level) + 1],
                             parent_color=color,
                             min_angle=min_taxa,
                             max_angle=max_taxa)

# Create node_angles dictionary from pos
node_angles = {}
for node in G_sub.nodes():
    if node.startswith('genus_'):
        x, y = pos[node]
        angle = np.arctan2(y, x)
        if angle < 0:
            angle += 2 * np.pi
        node_angles[node.split('_')[1]] = angle

# Draw taxonomy arcs
draw_taxonomy_arcs(node_angles, taxonomy_hierarchy)

# Update title
plt.title("Edge Bundle Visualization\n" +
          "Orange: ARGs | Red: Pathogenic Taxa | Grey: Other Taxa\n" +
          "Ecosystem colors: Orange: Host-associated | Green: Engineered | Blue: Environmental\n" +
          "Outer arcs show taxonomic relationships | Edge colors: Red → Blue shows direction",
          fontsize=16, pad=20)

# Save and close
plt.savefig('edge_bundle.png', 
           format='png', 
           bbox_inches='tight', 
           dpi=300,
           facecolor='white',
           edgecolor='none')
plt.close()