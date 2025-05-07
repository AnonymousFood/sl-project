import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.data_preprocessing as dp
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib import rcParams

# Publication-ready styling
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['axes.linewidth'] = 1.5
rcParams['figure.dpi'] = 300

def get_probability_scores(model, X, model_type):
    """
    Get probability scores for positive class.
    For models that don't return probabilities directly, 
    we implement appropriate handling.
    """
    proba = np.zeros(len(X))
    
    if model_type in ["id3", "c45", "bat"]:
        # For custom tree-based models that don't have built-in predict_proba
        # We use the proportion of "transported" leaf nodes in each path
        for i, (_, row) in enumerate(X.iterrows()):
            if model_type in ["id3", "c45"]:
                node = model.root
                while node.label is None:
                    attribute = node.attribute
                    value = row[attribute]
                    
                    if hasattr(node, "threshold") and node.threshold is not None:
                        branch = "left" if value <= node.threshold else "right"
                    else:
                        branch = value
                        
                    if branch not in node.branches:
                        # Use fallback if branch is missing
                        branches_values = list(node.branches.values())
                        leaf_counts = {0: 0, 1: 0}
                        for branch_node in branches_values:
                            if branch_node.label is not None:
                                leaf_counts[branch_node.label] += 1
                        proba[i] = leaf_counts[1] / sum(leaf_counts.values()) if sum(leaf_counts.values()) > 0 else 0.5
                        break
                    
                    node = node.branches[branch]
                
                if node.label is not None:
                    proba[i] = float(node.label)
            
            elif model_type == "bat":
                # For Born Again Tree
                node = model.tree
                while node.label is None:
                    attribute = node.attribute
                    value = row[attribute]
                    
                    branch = "left" if value <= node.threshold else "right"
                    node = node.branches[branch]
                
                proba[i] = float(node.label)
                
    elif model_type == "rf":
        # For random forest, calculate the proportion of trees predicting class 1
        all_votes = []
        for tree_obj, feature_indices, feature_names in model.trees:
            # Create a DataFrame with just the features this tree uses
            sample_features = pd.DataFrame([row[feature_names].values for _, row in X.iterrows()],
                                           columns=feature_names)
            
            # Get predictions from this tree
            tree_preds = tree_obj.predict(sample_features)
            all_votes.append(tree_preds)
            
        # Convert to numpy array for easier calculations
        all_votes = np.array(all_votes)
        
        # Calculate proportion of 'True' votes for each sample
        proba = np.mean(all_votes, axis=0)
    
    return proba

def plot_roc_curves(models_dict, X, y, title='ROC Curve Comparison', figsize=(12, 10), filename="model_comparison_roc"):
    """Plot ROC curves with publication-ready styling"""
    plt.figure(figsize=figsize)
    
    # Professional color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    linestyles = ['-', '-', '-', '--', '--', '--']
    linewidths = [3, 3, 3, 2.5, 2.5, 2.5]
    
    # Plot each model with enhanced styling
    for i, (model_name, (model, model_type)) in enumerate(models_dict.items()):
        # Get probability scores
        y_score = get_probability_scores(model, X, model_type)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve with publication styling
        plt.plot(fpr, tpr, 
                 linestyle=linestyles[i % len(linestyles)],
                 linewidth=linewidths[i % len(linewidths)], 
                 color=colors[i % len(colors)],
                 label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    # Plot the diagonal (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.7)
    
    # Enhanced styling
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=22, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=22, fontweight='bold')
    plt.title(title, fontsize=24, fontweight='bold', pad=20)
    
    # Enhanced legend
    plt.legend(loc="lower right", fontsize=18, frameon=True, 
               fancybox=True, framealpha=0.95, 
               shadow=True, borderpad=1)
    
    # Add grid but make it subtle
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Increase tick size
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    # Save in multiple formats for publication
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{filename}.svg", bbox_inches='tight')
    plt.savefig(f"{filename}.pdf", bbox_inches='tight')
    
    print(f"ROC curves comparison saved as '{filename}' in PNG, SVG, and PDF formats")
    plt.show()

# Add a new function to create comparison confusion matrices
def plot_confusion_matrices(models_dict, X, y, filename="model_comparison_cm"):
    """Create a grid of confusion matrices for all models"""
    n_models = len(models_dict)
    
    # Calculate rows and columns for the grid
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), dpi=300)
    
    # Flatten axes for easier indexing if we have multiple rows
    if n_rows > 1:
        axes = axes.flatten()
    elif n_cols == 1:  # Handle single subplot case
        axes = [axes]
    
    # For each model, create a confusion matrix
    for i, (model_name, (model, model_type)) in enumerate(models_dict.items()):
        # Get predictions
        if model_type == "rf":
            y_pred = model.predict(X)
        elif model_type == "bat":
            y_pred = model.predict(X)
        elif model_type in ["id3", "c45"]:
            y_pred = model.predict(X)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Plot on the appropriate subplot
        ax = axes[i] if i < len(axes) else axes[-1]
        
        # Try to calculate spacing
        row_sums = np.sum(cm, axis=1, keepdims=True)
        cm_percentages = cm / row_sums * 100
        
        # Put count and percentage in each cell
        annot = np.array([[f"{val}\n({pct:.1f}%)" for val, pct in zip(row, row_pct)] 
                        for row, row_pct in zip(cm, cm_percentages)])
        
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=ax,
                    xticklabels=["Not Transported", "Transported"],
                    yticklabels=["Not Transported", "Transported"],
                    annot_kws={"size": 16, "weight": "bold"},
                    cbar=False)
        
        ax.set_xlabel('Predicted', fontsize=18, fontweight='bold')
        ax.set_ylabel('True', fontsize=18, fontweight='bold')
        ax.set_title(f"{model_name}", fontsize=20, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Remove any unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout(pad=2.0)
    
    # Save in multiple formats
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{filename}.svg", bbox_inches='tight')
    plt.savefig(f"{filename}.pdf", bbox_inches='tight')
    
    print(f"Confusion matrices saved as '{filename}' in PNG, SVG, and PDF formats")
    plt.show()

def compare_models(models, train_data_path, test_path=None):
    """Compare multiple models using ROC curves and confusion matrices"""
    # Read data
    train_df = pd.read_csv(train_data_path)
    
    # Preprocess data
    X = dp.preprocess_data(train_df)
    y = train_df['Transported'].astype(int)
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Plot ROC curves using validation set
    plot_roc_curves(models, X_test, y_test, 
                    title='ROC Curve Comparison - Validation Set',
                    filename="model_comparison_validation_roc")
    
    # Plot confusion matrices using validation set
    plot_confusion_matrices(models, X_test, y_test,
                          filename="model_comparison_validation_cm")
    
    # If test data is provided, also evaluate on it
    if test_path:
        test_df = pd.read_csv(test_path)
        X_test_external = dp.preprocess_data(test_df)
        if 'Transported' in test_df.columns:  # Only if labels are available
            y_test_external = test_df['Transported'].astype(int)
            
            plot_roc_curves(models, X_test_external, y_test_external,
                          title='ROC Curve Comparison - Test Set',
                          filename="model_comparison_test_roc")
            
            plot_confusion_matrices(models, X_test_external, y_test_external,
                                  filename="model_comparison_test_cm")