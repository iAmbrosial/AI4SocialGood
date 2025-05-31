import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Azrieli Centre for Autism Research (ACAR)", layout="wide")
st.header("Azrieli Centre for Autism Research (ACAR)")
st.subheader("Social Network Clustering from Node2Vec")
from pyvis.network import Network
import networkx as nx
import streamlit.components.v1 as components

embeddings = pd.read_csv('./data/embeddings_ASF.csv')

with open('./data/graph_ACAR.pkl', 'rb') as f:
    G = pickle.load(f)

with open('./data/semantic_dict_ACAR.pkl', 'rb') as f:
    dict_mapping = pickle.load(f)

from pyvis.network import Network
import networkx as nx
import json

# === Legend HTML (top-right)
legend_html = """
<div style="
  position: absolute;
  top: 10px;
  right: 10px;
  background-color: white;
  border: 1px solid #ccc;
  padding: 10px;
  font-size: 14px;
  z-index: 1000;
  box-shadow: 0 2px 6px rgba(0,0,0,0.15);
">
  <strong>Legend</strong><br>
  <span style='color:red;'>●</span> ORG (center)<br>
  <span style='color:blue;'>●</span> 1st Degree<br>
  <span style='color:green;'>●</span> 2nd Degree<br><br>
  <span style='display:inline-block; width:12px; height:12px; background:#555; border-radius:6px;'></span> Cluster 1 (dot)<br>
  <span style='display:inline-block; width:12px; height:12px; background:#555;'></span> Cluster 2 (square)
</div>
"""

root_org_user = 'NeuroACAR'.lower()

# === Graph Setup
ORG = root_org_user.lower()
first_degree = set(G.successors(ORG)) | set(G.predecessors(ORG))

second_degree_raw = set()
for node in first_degree:
    second_degree_raw |= set(G.successors(node))
    second_degree_raw |= set(G.predecessors(node))

second_degree = second_degree_raw - {ORG} - first_degree

degree_centrality = nx.degree_centrality(G)

# Sidebar sliders
st.sidebar.title("Graph Controls")
st.sidebar.subheader("Node Structure Clustering")

# Slider to control max number of 1st-degree and 2nd-degree nodes
MAX_FIRST_DEGREE = st.sidebar.slider("Max First-Degree Nodes (Node2Vec)", min_value=10, max_value=len(first_degree), value=10, step=10)
MAX_SECOND_DEGREE = st.sidebar.slider("Max Second-Degree Nodes (Node2Vec)", min_value=10, max_value=len(second_degree), value=500, step=10)

st.sidebar.subheader("Semantic Structure Clustering")

# Slider to control max number of 1st-degree and 2nd-degree nodes
MAX_FIRST_DEGREE2 = st.sidebar.slider("Max First-Degree Nodes (Semantics)", min_value=10, max_value=len(first_degree), value=20, step=10)
MAX_SECOND_DEGREE2 = st.sidebar.slider("Max Second-Degree Nodes (Semantics)", min_value=10, max_value=len(second_degree), value=400, step=10)

print(f"Degree {len(first_degree)}/{MAX_FIRST_DEGREE}")
print(f"Degree {len(second_degree)}/{MAX_SECOND_DEGREE}")

first_degree = set(sorted(first_degree, key=lambda n: degree_centrality.get(n, 0), reverse=True)[:MAX_FIRST_DEGREE])
second_degree = set(sorted(second_degree, key=lambda n: degree_centrality.get(n, 0), reverse=True)[:MAX_SECOND_DEGREE])

all_nodes = {ORG} | first_degree | second_degree
H = G.subgraph(all_nodes)

# === PyVis Network with physics config below
net = Network(height="100vh", width="100%", notebook=False, directed=True)

net.set_options("""
{
  "configure": {
    "enabled": true,
    "filter": "physics"
  },
  "physics": {
    "stabilization": true
  },
  "nodes": {
    "font": {
      "size": 14
    }
  },
  "edges": {
    "arrows": {
      "to": { "enabled": true, "scaleFactor": 0.4 }
    },
    "smooth": true
  }
}
""")

# === Add nodes and edges
node_clusters = embeddings['cluster'].to_dict()
cluster_shapes = { 0: 'dot', 1: 'square', 2: 'triangle' }

for node in H.nodes():
    bio = str(G.nodes[node].get('bio', '') or '')
    color = (
        'red' if node == ORG else
        'blue' if node in first_degree else
        'green' if node in second_degree else
        'gray'
    )
    cluster_id = node_clusters.get(node, -1)
    shape = cluster_shapes.get(cluster_id, 'dot')
    net.add_node(node, label=node, title=bio, color=color, shape=shape)

for source, target in H.edges():
    net.add_edge(source, target)

# === Save and inject legend
output_path = "asf_network_with_clean_legend.html"
net.save_graph(output_path)

with open(output_path, "r", encoding="utf-8") as f:
    html = f.read()

html = html.replace("</body>", f"{legend_html}</body>")

with open(output_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"✅ Saved to: {output_path} — with floating legend only.")

# Display in Streamlit
with open("asf_network_with_clean_legend.html", "r", encoding="utf-8") as f:
    html = f.read()
    components.html(html, height=550, width=800)

st.subheader("Social Network Clustering from SentenceTransformer")
# === Legend HTML (top-right)
legend_html = """
<div style="
  position: absolute;
  top: 10px;
  right: 10px;
  background-color: white;
  border: 1px solid #ccc;
  padding: 10px;
  font-size: 14px;
  z-index: 1000;
  box-shadow: 0 2px 6px rgba(0,0,0,0.15);
">
  <strong>Legend</strong><br>
  <span style='color:red;'>●</span> Cluster 1<br>
  <span style='color:blue;'>●</span> Cluster 2<br>
</div>
"""

# === Graph Setup
ORG = root_org_user.lower()
first_degree = set(G.successors(ORG)) | set(G.predecessors(ORG))

second_degree_raw = set()
for node in first_degree:
    second_degree_raw |= set(G.successors(node))
    second_degree_raw |= set(G.predecessors(node))

second_degree = second_degree_raw - {ORG} - first_degree

degree_centrality = nx.degree_centrality(G)

print(f"Degree {len(first_degree)}/{MAX_FIRST_DEGREE2}")
print(f"Degree {len(second_degree)}/{MAX_SECOND_DEGREE2}")

first_degree = set(sorted(first_degree, key=lambda n: degree_centrality.get(n, 0), reverse=True)[:MAX_FIRST_DEGREE2])
second_degree = set(sorted(second_degree, key=lambda n: degree_centrality.get(n, 0), reverse=True)[:MAX_SECOND_DEGREE2])

all_nodes = {ORG} | first_degree | second_degree
H = G.subgraph(all_nodes)

# === PyVis Network with physics config below
net = Network(height="100vh", width="100%", notebook=False, directed=True)

net.set_options("""
{
  "configure": {
    "enabled": true,
    "filter": "physics"
  },
  "physics": {
    "stabilization": true
  },
  "nodes": {
    "font": {
      "size": 14
    }
  },
  "edges": {
    "arrows": {
      "to": { "enabled": true, "scaleFactor": 0.4 }
    },
    "smooth": true
  }
}
""")

# === Add nodes and edges
node_clusters = dict_mapping
cluster_shapes = { 0: 'dot', 1: 'square', 2: 'triangle' }

for node in H.nodes():
    bio = str(G.nodes[node].get('bio', '') or '')
    color = (
        'red' if node_clusters.get(node, -1) == 0 else
        'blue' if node_clusters.get(node, -1) ==1 else
        'green' if node_clusters.get(node, -1) ==2 else
        'gray'
    )
    cluster_id = node_clusters.get(node, -1)
    shape = cluster_shapes.get(cluster_id, 'dot')
    net.add_node(node, label=node, color=color, title=bio)

for source, target in H.edges():
    net.add_edge(source, target)

# === Save and inject legend
output_path = "asf_network_semantic_network.html"
net.save_graph(output_path)

with open(output_path, "r", encoding="utf-8") as f:
    html = f.read()

html = html.replace("</body>", f"{legend_html}</body>")

with open(output_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"✅ Saved to: {output_path} — with floating legend only.")

# Display in Streamlit
with open("asf_network_semantic_network.html", "r", encoding="utf-8") as f:
    html = f.read()
    components.html(html, height=550, width=800)