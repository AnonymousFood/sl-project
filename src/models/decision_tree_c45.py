import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.data_preprocessing as dp

class Node:
    def __init__(self, attribute=None, threshold=None, label=None):
        self.attribute = attribute  # splitting attribute
        self.threshold = threshold  # threshold for continuous attributes
        self.label = label  # leaf node label
        self.branches = {}  # dictionary to store child nodes

class C45DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=5, class_weight=None):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.numerical_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num', 'TotalSpent']
        self.class_weight = class_weight
        self.class_weights = None
        
    def calculate_entropy(self, y):
        counts = Counter(y)
        entropy = 0
        total = len(y)
        
        # Initialize class_weights if not already done
        if self.class_weights is None:
            if self.class_weight == 'balanced':
                n_samples = len(y)
                self.class_weights = {cls: n_samples / (len(counts) * count) 
                                    for cls, count in counts.items()}
            elif isinstance(self.class_weight, dict):
                self.class_weights = self.class_weight
            else:
                self.class_weights = {cls: 1.0 for cls in set(y)}
        
        # Apply class weights to probability calculation
        weighted_counts = {cls: count * self.class_weights.get(cls, 1.0) 
                          for cls, count in counts.items()}
        weighted_total = sum(weighted_counts.values())
        
        for cls, weighted_count in weighted_counts.items():
            prob = weighted_count / weighted_total if weighted_total > 0 else 0
            entropy -= prob * np.log2(prob) if prob > 0 else 0
            
        return entropy

    def calculate_split_info(self, X, attribute, threshold=None):
        if attribute in self.numerical_features:
            mask = X[attribute] <= threshold
            splits = [mask, ~mask]
        else:
            splits = [X[attribute] == value for value in X[attribute].unique()]
            
        split_info = 0
        total = len(X)
        
        for split in splits:
            subset_size = sum(split)
            if subset_size > 0:
                prob = subset_size / total
                split_info -= prob * np.log2(prob)
        
        return split_info

    def calculate_gain_ratio(self, X, y, attribute, threshold=None):
        parent_entropy = self.calculate_entropy(y)
        weighted_entropy = 0
        total = len(y)

        if attribute in self.numerical_features:
            mask = X[attribute] <= threshold
            splits = [(mask, "left"), (~mask, "right")]
        else:
            splits = [(X[attribute] == value, value) for value in X[attribute].unique()]

        for split, _ in splits:
            subset_y = y[split]
            if len(subset_y) > 0:
                weight = len(subset_y) / total
                weighted_entropy += weight * self.calculate_entropy(subset_y)

        information_gain = parent_entropy - weighted_entropy
        split_info = self.calculate_split_info(X, attribute, threshold)
        
        return information_gain / split_info if split_info > 0 else 0

    def find_best_split(self, X, y, attributes):
        best_gain_ratio = -1
        best_attribute = None
        best_threshold = None

        for attribute in attributes:
            if attribute in self.numerical_features:
                unique_values = sorted(X[attribute].unique())
                thresholds = [(a + b) / 2 for a, b in zip(unique_values[:-1], unique_values[1:])]
                
                for threshold in thresholds:
                    gain_ratio = self.calculate_gain_ratio(X, y, attribute, threshold)
                    if gain_ratio > best_gain_ratio:
                        best_gain_ratio = gain_ratio
                        best_attribute = attribute
                        best_threshold = threshold
            else:
                gain_ratio = self.calculate_gain_ratio(X, y, attribute)
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_attribute = attribute
                    best_threshold = None

        return best_attribute, best_threshold

    def c45(self, X, y, attributes, depth=0):
        # Create leaf node if stopping criteria met
        if len(set(y)) == 1:
            return Node(label=y.iloc[0])
        
        if not attributes or (self.max_depth and depth >= self.max_depth) or len(X) < self.min_samples_split:
            return Node(label=Counter(y).most_common(1)[0][0])

        # Find best attribute and split
        best_attribute, threshold = self.find_best_split(X, y, attributes)
        if best_attribute is None:
            return Node(label=Counter(y).most_common(1)[0][0])

        # Create node with best split
        node = Node(attribute=best_attribute, threshold=threshold)
        remaining_attributes = [attr for attr in attributes if attr != best_attribute]

        if best_attribute in self.numerical_features:
            # Binary split for numerical attributes based on threshold
            # Should probably also have multiple thresholds
            mask = X[best_attribute] <= threshold
            left_mask = mask
            right_mask = ~mask
            
            if sum(left_mask) > 0:
                node.branches["left"] = self.c45(X[left_mask], y[left_mask], 
                                               remaining_attributes, depth + 1)
            if sum(right_mask) > 0:
                node.branches["right"] = self.c45(X[right_mask], y[right_mask], 
                                                remaining_attributes, depth + 1)
        else:
            # Multi-way split for categorical attributes
            for value in X[best_attribute].unique():
                mask = X[best_attribute] == value
                if sum(mask) > 0:
                    node.branches[value] = self.c45(X[mask], y[mask], 
                                                  remaining_attributes, depth + 1)

        return node

    def fit(self, X, y):
        X = pd.DataFrame(X) if isinstance(X, np.ndarray) else X
        y = pd.Series(y) if isinstance(y, np.ndarray) else y
        
        # Process class weights
        if self.class_weight == 'balanced':
            # Calculate balanced weights
            class_counts = Counter(y)
            n_samples = len(y)
            self.class_weights = {cls: n_samples / (len(class_counts) * count) 
                                 for cls, count in class_counts.items()}
        elif isinstance(self.class_weight, dict):
            self.class_weights = self.class_weight
        else:
            self.class_weights = {cls: 1.0 for cls in set(y)}  # Default equal weights
            
        attributes = list(X.columns)
        self.root = self.c45(X, y, attributes)
        return self

    def predict_one(self, x, node):
        if node.label is not None:
            return node.label

        value = x[node.attribute]
        
        if node.threshold is not None:  # Numerical attribute
            branch = "left" if value <= node.threshold else "right"
        else:  # Categorical attribute
            branch = value
            
        if branch not in node.branches:
            # Handle unseen values by returning most common prediction
            predictions = [self.predict_one(x, b) for b in node.branches.values()]
            return Counter(predictions).most_common(1)[0][0]
            
        return self.predict_one(x, node.branches[branch])

    def predict(self, X):
        X = pd.DataFrame(X) if isinstance(X, np.ndarray) else X
        return [self.predict_one(x, self.root) for _, x in X.iterrows()]

def train_decision_tree(train_data_path):
    # Read training data
    train_df = pd.read_csv(train_data_path)
    
    # Preprocess data
    X = dp.preprocess_data(train_df)
    y = train_df['Transported'].astype(int)
    
    # Split training data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train model
    dt_classifier = C45DecisionTree(max_depth=4, min_samples_split=5)
    dt_classifier.fit(X_train, y_train)
    
    # Make predictions on validation set
    y_pred = dt_classifier.predict(X_val)
    
    # Print performance
    print("\nC4.5 Decision Tree Model Performance:")
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, digits=4))
    
    return dt_classifier

def predict(model, test_data_path):
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