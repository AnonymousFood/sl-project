import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
from collections import Counter

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.data_preprocessing as dp

class Node:
    def __init__(self, attribute=None, label=None):
        self.attribute = attribute  # splitting attribute
        self.label = label  # leaf node label for prediction
        self.branches = {}  # dictionary to store child nodes
        
class ID3DecisionTree:
    def __init__(self, max_depth=None):
        self.root = None
        self.max_depth = max_depth
        
    def calculate_entropy(self, y):
        # Calculate entropy of a dataset
        counts = Counter(y)
        entropy = 0
        total = len(y)
        for count in counts.values():
            prob = count / total
            entropy -= prob * np.log2(prob)
        return entropy
    
    def calculate_information_gain(self, X, y, attribute):
        # Calculate information gain for an attribute
        parent_entropy = self.calculate_entropy(y)
        
        # Calculate weighted entropy of children
        values = X[attribute].unique()
        weighted_entropy = 0
        total = len(y)
        
        for value in values:
            mask = X[attribute] == value
            subset_y = y[mask]
            if len(subset_y) > 0:
                weight = len(subset_y) / total
                weighted_entropy += weight * self.calculate_entropy(subset_y)
                
        information_gain = parent_entropy - weighted_entropy
        return information_gain
    
    def find_best_attribute(self, X, y, attributes):
        # Find attribute with highest information gain
        gains = {attr: self.calculate_information_gain(X, y, attr) for attr in attributes}
        return max(gains.items(), key=lambda x: x[1])[0]
    
    def id3(self, X, y, attributes, depth=0):
        # Create a leaf node if all examples have same class
        if len(set(y)) == 1:
            return Node(label=y.iloc[0])
        
        # Create a leaf node if no attributes remain or max depth reached
        if not attributes or (self.max_depth and depth >= self.max_depth):
            return Node(label=Counter(y).most_common(1)[0][0])
        
        # Find best attribute to split on
        best_attribute = self.find_best_attribute(X, y, attributes)
        
        # Create node with best attribute
        node = Node(attribute=best_attribute)
        
        # Create child nodes for each value of best attribute
        remaining_attributes = [attr for attr in attributes if attr != best_attribute]
        for value in X[best_attribute].unique():
            # Get subset of examples with this attribute value
            mask = X[best_attribute] == value
            subset_X = X[mask]
            subset_y = y[mask]
            
            if len(subset_X) == 0:
                # If no examples with this value, create leaf with most common class
                node.branches[value] = Node(label=Counter(y).most_common(1)[0][0])
            else:
                # Recursively build subtree
                node.branches[value] = self.id3(subset_X, subset_y, remaining_attributes, depth + 1)
                
        return node
    
    def fit(self, X, y):
        # Convert to pandas if numpy arrays
        X = pd.DataFrame(X) if isinstance(X, np.ndarray) else X
        y = pd.Series(y) if isinstance(y, np.ndarray) else y
        
        attributes = list(X.columns)
        self.root = self.id3(X, y, attributes)
        return self
    
    def predict_one(self, x, node):
        # Reach leaf node
        if node.label is not None:
            return node.label
        
        # Get value of splitting attribute for this example
        value = x[node.attribute]
        
        # If value not seen during training, return most common branch prediction
        if value not in node.branches:
            # Get most common prediction from all branches
            predictions = [self.predict_one(x, branch) for branch in node.branches.values()]
            return Counter(predictions).most_common(1)[0][0]
            
        return self.predict_one(x, node.branches[value])
    
    def predict(self, X):
        # Convert to pandas if numpy array
        X = pd.DataFrame(X) if isinstance(X, np.ndarray) else X
        return [self.predict_one(x, self.root) for _, x in X.iterrows()]

def train_decision_tree(train_data_path): # training part
    # Read training data
    train_df = pd.read_csv(train_data_path)
    
    # Preprocess data
    X = dp.preprocess_data(train_df)
    y = train_df['Transported'].astype(int)
    
    # Split training data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train model
    dt_classifier = ID3DecisionTree(max_depth=10)
    dt_classifier.fit(X_train, y_train)
    
    # Make predictions on validation set
    y_pred = dt_classifier.predict(X_val)
    
    # Print performance
    print("\nID3 Decision Tree Model Performance:")
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, digits=4))
    
    return dt_classifier

def predict(model, test_data_path): # testing part
    # Read test data
    test_df = pd.read_csv(test_data_path)
    passenger_ids = test_df['PassengerId']
    
    # Preprocess test data
    X_test = dp.preprocess_data(test_df)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Store passenger IDs mapped to predictions
    prediction_df = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Transported': pd.Series(predictions).astype(bool)
    })
    
    return prediction_df