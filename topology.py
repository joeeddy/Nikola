import random
import math
from node import FractalNode

def create_fractal_network(depth, inputs_per_node, hidden_size, output_size):
    """Builds layered fractal nodes and initializes connections."""
    nodes = []
    connections = []
    node_id_counter = 0
    for level in range(depth):
        level_nodes = []
        num_nodes = 2 ** level
        for _ in range(num_nodes):
            input_size = inputs_per_node if level == 0 else output_size
            node = FractalNode(input_size, hidden_size=hidden_size, output_size=output_size, node_id=node_id_counter)
            level_nodes.append(node)
            node_id_counter += 1
        nodes.append(level_nodes)

    for level in range(1, depth):
        for node in nodes[level]:
            prev_ids = [n.node_id for n in nodes[level - 1]]
            connections.append((node.node_id, prev_ids[:min(inputs_per_node, len(prev_ids))]))
    
    return nodes, connections, node_id_counter

def self_organize(nodes, connections, temperature, inputs_per_node, output_size, node_id_counter, evolution_cfg):
    """Rewires connections, prunes underperforming nodes, and spawns new ones."""
    new_connections = []

    for level in range(1, len(nodes)):
        for node in nodes[level]:
            conn = [c for c in connections if c[0] == node.node_id]
            if conn:
                conn = conn[0]
                current_conn_ids = conn[1]
                prev_nodes = [n.node_id for n in nodes[level - 1]]
                available = [nid for nid in prev_nodes if nid not in current_conn_ids]

                new_conn_ids = current_conn_ids
                if random.random() < math.exp(-1 / temperature) and available:
                    new_conn_ids = random.sample(prev_nodes, min(inputs_per_node, len(prev_nodes)))
                
                current_score = sum(n.connection_weights.get(i, 0) for i in current_conn_ids for n in nodes[level - 1])
                new_score = sum(n.connection_weights.get(i, 0) for i in new_conn_ids for n in nodes[level - 1])
                
                if new_score > current_score or random.random() < math.exp((new_score - current_score) / temperature):
                    new_connections.append((node.node_id, new_conn_ids))
                else:
                    new_connections.append(conn)
    
    temperature *= evolution_cfg["temperature_decay"]
    connections[:] = new_connections

    for level in range(len(nodes)):
        to_remove = []
        to_add = []
        for node in nodes[level]:
            if len(node.activity_log) > 20 and sum(node.activity_log[-20:]) / 20 < evolution_cfg["prune_threshold"]:
                to_remove.append(node)
            elif node.performance > evolution_cfg["spawn_threshold"] and random.random() < evolution_cfg["spawn_chance"]:
                new_node = FractalNode(node.input_size, hidden_size=node.hidden_size, output_size=output_size, node_id=node_id_counter)
                node_id_counter += 1
                to_add.append(new_node)
                if level < len(nodes) - 1:
                    connections.append((new_node.node_id, [n.node_id for n in nodes[level]]))
        
        for dead in to_remove:
            nodes[level] = [n for n in nodes[level] if n.node_id != dead.node_id]
            connections[:] = [c for c in connections if c[0] != dead.node_id]
        
        nodes[level].extend(to_add)

    return temperature, node_id_counter
    def get_topology_graph(depth=4):
    import networkx as nx
    G = nx.Graph()
    
    def add_fractal_nodes(parent, level):
        if level > depth:
            return
        for i in range(2):
            child = f"{parent}.{i}" if parent else str(i)
            G.add_edge(parent, child) if parent else G.add_node(child)
            add_fractal_nodes(child, level + 1)
    
    add_fractal_nodes("", 1)
    return G
