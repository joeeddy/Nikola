# viz.py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import streamlit as st

# These imports assume you're exposing internal states from other files
from topology import get_topology_graph
from node import get_node_activity_map
from helpers import get_evolution_trace

st.set_page_config(layout="wide")
st.title("🧠 Nikola Visualization Dashboard")

# 🔧 Sidebar Control Panel
st.sidebar.header("Control Panel")
learning_rate = st.sidebar.slider("Learning Rate", 0.001, 1.0, 0.05)
depth = st.sidebar.selectbox("Network Depth", options=[2, 4, 8, 16], index=2)
entropy_threshold = st.sidebar.slider("Entropy Threshold", 0.0, 1.0, 0.5)
st.sidebar.button("Apply Changes")

# 🌐 Fractal Topology Map
st.subheader("Fractal Topology Overview")
G = get_topology_graph(depth=depth)
fig1, ax1 = plt.subplots()
nx.draw(G, ax=ax1, node_size=100, node_color="skyblue", with_labels=True)
st.pyplot(fig1)

# 📈 Node Activity Visualization
st.subheader("Live Node Activity Map")
activity_matrix = get_node_activity_map()
fig2, ax2 = plt.subplots()
ax2.imshow(activity_matrix, cmap="plasma")
ax2.set_title("Node Activity Levels")
st.pyplot(fig2)

# 🧬 Evolution Trail Plot
st.subheader("Evolution Trail Over Time")
evolution_log = get_evolution_trace()
fig3, ax3 = plt.subplots()
ax3.plot(evolution_log, color="green")
ax3.set_xlabel("Time")
ax3.set_ylabel("Structural Events")
ax3.set_title("Neural Evolution Progression")
st.pyplot(fig3)
