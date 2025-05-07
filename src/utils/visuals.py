from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
import numpy as np

# Receiver Operating Characteristic Curve - useful for evaluating binary classifiers
def roc(y_true, y_pred, title):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Transportation Prediction Rate')
    plt.ylabel('True Transportation Prediction Rate')
    plt.title('Receiver Operating Characteristic' + title)
    plt.legend(loc='lower right')
    plt.show()
    
# roc - set
def roc_compare(y_true, y_pred_list, model_names, title):
    plt.figure()
    
    for y_pred, model_name in zip(y_pred_list, model_names):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')

    # Plot the diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison' + title + " - Validation Set")
    plt.legend(loc='lower right')
    plt.show()
    
# Confusion Matrix - Shows Both classes
def cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    # Plot using Seaborn heatmap
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Transported', 'Transported'], yticklabels=['Not Transported', 'Transported'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix' + title)
    plt.show()
    
# Attribute Importance Bar Chart Comparison
# value_comparing is expected to be a 2D array : [label][attribute]
def attribute_bar_compare(attributes, value_comparing, value_being_compared_str):
    # Plotting the means for each class
    fig, ax = plt.subplots(figsize=(8, 6))
    width = 0.35  # Bar width
    x = np.arange(len(attributes))  # Attribute positions

    bars1 = ax.bar(x - width/2, value_comparing[0], width, label='Not Transported')
    bars2 =ax.bar(x + width/2, value_comparing[1], width, label='Transported')
    
    # Adding text annotations above the bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', 
                ha='center', va='bottom', fontsize=10)  # Displaying value for 'Not Transported'

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', 
                ha='center', va='bottom', fontsize=10)  # Displaying value for 'Transported'

    ax.set_xlabel('Attributes')
    ax.set_ylabel(f'{value_being_compared_str} Value')
    ax.set_title(f'Attributes {value_being_compared_str} for Each Class')
    ax.set_xticks(x)
    ax.set_xticklabels(attributes)
    ax.legend()

    plt.show()
    
def visualize_heatmap(value_comparing, value_being_compared_str):
    # Convert y_attribute_probs to a DataFrame for easy visualization
    data = {}
    for label, attribute_probs in value_comparing.items():
        c_label = 'Not Transported' if label == 0 else 'Transported'
        for attribute, probs in attribute_probs.items():
            if attribute not in data:
                data[attribute] = {}
            
            data[attribute][c_label] = probs

    df = pd.DataFrame(data)
    
    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df, annot=False, cmap='Reds', fmt='.2f', cbar=True, annot_kws={"size": 8})
    plt.title(f'Conditional Probability Distribution for {value_being_compared_str} of Attributes by Class')
    plt.ylabel('Attributes')
    plt.xlabel('Classes')
    plt.show()
    
def visualize_feature_influence(y_attribute_probs, attributes, classes):
    # Create a plot for each class to show feature influence
    fig, axes = plt.subplots(nrows=1, ncols=len(classes), figsize=(15, 6))
    
    for i, class_label in enumerate(classes):
        c_label = 'Not Transported' if class_label == 0 else 'Transported'
        axes[i].set_title(f'Class: {c_label}')
        axes[i].set_ylabel('Feature Influence')
        feature_influence = []
        
        # Calculate the product of probabilities for each feature for this class
        for feature in y_attribute_probs[class_label]:
            feature_probs = y_attribute_probs[class_label].get(feature, {})
            influence = 1
            for feature_value, prob in feature_probs.items():
                influence *= prob
            feature_influence.append(influence)
        
        axes[i].bar(attributes, feature_influence, color='skyblue')
        axes[i].set_xticklabels(attributes, rotation=45)
    
    plt.tight_layout()
    plt.show()
    
def visualize_stacked_feature_contributions(y_attribute_probs, features, classes):
    print(y_attribute_probs)
    
    # Prepare the data for the stacked bar chart
    fcs = {}
    
    for class_label in classes:
        feature_contributions = {feature: [] for feature in features}
        for feature in features:
            feature_probs = y_attribute_probs[class_label].get(feature, {})
            contribution = 1  # Initialize the product
            for feature_value, prob in feature_probs.items():
                contribution *= prob
            feature_contributions[feature].append(contribution)
        fcs[class_label] = feature_contributions
    
    # Create a stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the stacked bars
    bottom_values = np.zeros(len(features))
    for i, class_label in enumerate(classes):
        c_label = 'Not Transported' if class_label == 0 else 'Transported'
        ax.bar(features, fcs[class_label][feature], label=f'Class {c_label}', bottom=bottom_values)
        bottom_values += fcs[class_label][feature]  # Stack the contributions

    ax.set_xlabel('Features')
    ax.set_ylabel('Feature Contribution to Classification')
    ax.set_title('Stacked Feature Contributions by Class')
    ax.legend()

    plt.show()