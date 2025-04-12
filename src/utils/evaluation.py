from graphviz import Digraph
import matplotlib.pyplot as plt
from sklearn import tree
import os

def visualize_tree(node, graph, parent_id=None, edge_label=None, node_id=None):
    if node_id is None:
        node_id = 0
    current_id = str(node_id)
    
    # Create node label
    if node.label is not None:
        # Convert boolean/int to more readable format
        label_text = "Transported" if node.label == 1 else "Not Transported"
        label = f'Class: {label_text}'
    else:
        label = f'{node.attribute}'
        # Add threshold only for C4.5 numerical splits
        if hasattr(node, 'threshold') and node.threshold is not None:
            label += f'\n≤ {node.threshold:.2f}'
    
    # Style the node
    if node.label is not None:
        # Leaf nodes in light blue
        graph.node(current_id, label, style='filled', fillcolor='lightblue')
    else:
        # Decision nodes in light green
        graph.node(current_id, label, style='filled', fillcolor='lightgreen')
    
    # Connect to parent if exists
    if parent_id is not None:
        if edge_label == "left":
            edge_label = "≤"
        elif edge_label == "right":
            edge_label = ">"
        graph.edge(parent_id, current_id, edge_label or '')
    
    # Recursively visualize children
    next_id = node_id + 1
    for branch, child in node.branches.items():
        next_id = visualize_tree(child, graph, current_id, str(branch), next_id)
    
    return next_id

def save_tree_visualization(model, filename="decision_tree"):
    """
    Creates and saves a visualization of the decision tree model
    
    Args:
        model: Decision tree model (ID3, C4.5, or sklearn)
        filename: Name of the output file (without extension)
    """
    if isinstance(model, (tree.DecisionTreeClassifier, tree.DecisionTreeRegressor)):
        # Handle sklearn tree visualization
        plt.figure(figsize=(20,10))
        tree.plot_tree(model, 
                      feature_names=model.feature_names_in_,
                      class_names=['Not Transported', 'Transported'],
                      filled=True,
                      rounded=True)
        plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nDecision tree visualization saved as '{filename}.png'")
    else:
        # Handle custom tree visualization (ID3/C4.5)
        dot = Digraph(comment='Decision Tree Visualization')
        dot.attr(rankdir='TB')
        dot.attr('node', shape='box')
        
        # Add title to the graph
        model_type = "C4.5" if hasattr(model, 'numerical_features') else "ID3"
        dot.attr(label=f'\n{model_type} Decision Tree\n')
        dot.attr(labelloc='t')
        
        visualize_tree(model.root, dot)
        dot.render(filename, format="png", cleanup=True)
        print(f"\nDecision tree visualization saved as '{filename}.png'")