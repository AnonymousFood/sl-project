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

    ax.bar(x - width/2, value_comparing[0], width, label='Not Transported')
    ax.bar(x + width/2, value_comparing[1], width, label='Transported')

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
        for attribute, probs in attribute_probs.items():
            if attribute not in data:
                data[attribute] = {}
            data[attribute][f'Class_{label}'] = probs

    df = pd.DataFrame(data)
    
    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df, annot=True, cmap='Blues', fmt='.2f', cbar=True)
    plt.title(f'Conditional Probability Distribution for {value_being_compared_str} of Attributes by Class')
    plt.ylabel('Attributes')
    plt.xlabel('Classes')
    plt.show()