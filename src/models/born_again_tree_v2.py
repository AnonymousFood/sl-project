import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import os
import sys
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.data_preprocessing as dp
from sklearn.model_selection import train_test_split

class Node:
    def __init__(self, attribute=None, threshold=None, label=None):
        self.attribute = attribute  # splitting attribute
        self.threshold = threshold  # threshold for numerical attributes
        self.label = label  # leaf node label
        self.branches = {"left": None, "right": None}  # left/right branches

class BornAgainTree:
    def __init__(self, max_depth=None):
        self.tree = None
        self.max_depth = max_depth
        self.feature_names = None
        self.hyperplanes = {}  # Hyperplane levels for each feature
        self.memo = {}  # Memoization for dynamic programming
        
    def get_hyperplane_levels(self, forest):
        hyperplanes = defaultdict(set)
        
        for tree_obj, _, feature_names in forest.trees:
            self.extract_thresholds(tree_obj.root, feature_names, hyperplanes)
        
        # Sort thresholds for each feature
        for feature in hyperplanes:
            hyperplanes[feature] = sorted(hyperplanes[feature])
            
        return hyperplanes
    
    def extract_thresholds(self, node, feature_names, hyperplanes):
        if node is None or node.label is not None:
            return
            
        if node.attribute is not None and node.threshold is not None:
            feature = feature_names[node.attribute] if isinstance(node.attribute, int) else node.attribute
            hyperplanes[feature].add(node.threshold)
            
        # Traverse left and right branches
        for branch_name, branch_node in node.branches.items():
            if branch_node:
                self.extract_thresholds(branch_node, feature_names, hyperplanes)
    
    def get_prediction(self, x, forest):
        """Get prediction for a single point"""
        return forest.predict(x)[0]
    
    def is_region_constant(self, region, X, forest):
        """Check if a region has constant prediction"""
        # Sample corners
        corner_preds = []
        for corner in [region[0], region[1]]:  # Just check min/max corners
            x = pd.DataFrame(columns=X.columns)
            x.loc[0] = X.iloc[0].copy()
            
            for idx, feature in enumerate(self.feature_names):
                z = corner[idx]
                if len(self.hyperplanes.get(feature, [])) == 0:
                    continue
                    
                if z <= 1:  # Min
                    val = self.hyperplanes[feature][0] - 1 
                elif z >= len(self.hyperplanes[feature]):  # Max
                    val = self.hyperplanes[feature][-1] + 1
                else:  # Middle
                    val = (self.hyperplanes[feature][z-2] + self.hyperplanes[feature][z-1]) / 2
                
                x.loc[0, feature] = val
            
            try:
                corner_preds.append(forest.predict(x)[0])
            except:
                pass
        
        return len(set(corner_preds)) <= 1  # All same prediction
    
    def min_depth_tree(self, zL, zR, X, forest):
        # Create region key for memoization
        region_key = (tuple(zL), tuple(zR))
        if region_key in self.memo:
            return self.memo[region_key]
        
        # Check if single cell
        if all(zL[i] == zR[i] for i in range(len(zL))):
            x = pd.DataFrame(columns=X.columns)
            x.loc[0] = X.iloc[0].copy()
            pred = forest.predict(x)[0]
            node = Node(label=pred)
            self.memo[region_key] = (0, node)
            return self.memo[region_key]
        
        # Check if region has constant prediction
        if self.is_region_constant((zL, zR), X, forest):
            x = pd.DataFrame(columns=X.columns)
            x.loc[0] = X.iloc[0].copy()
            pred = forest.predict(x)[0]
            node = Node(label=pred)
            self.memo[region_key] = (0, node)
            return self.memo[region_key]
        
        # Try splitting on each feature and find best split
        best_depth = float('inf')
        best_node = None
        
        for j, feature in enumerate(self.feature_names):
            # Skip if no thresholds for this feature
            if not self.hyperplanes.get(feature):
                continue
                
            # Test each threshold
            for threshold_idx in range(zL[j], zR[j]):
                if threshold_idx <= 0 or threshold_idx > len(self.hyperplanes[feature]):
                    continue
                    
                # Split region
                zR_left = zR.copy()
                zR_left[j] = threshold_idx
                
                zL_right = zL.copy()
                zL_right[j] = threshold_idx
                
                # Recursively find best trees for subregions
                depth1, node1 = self.min_depth_tree(zL, zR_left, X, forest)
                depth2, node2 = self.min_depth_tree(zL_right, zR, X, forest)
                
                # Calculate depth
                current_depth = 1 + max(depth1, depth2)
                
                # Update best if improved
                if current_depth < best_depth:
                    best_depth = current_depth
                    threshold = self.hyperplanes[feature][threshold_idx-1]
                    best_node = Node(attribute=feature, threshold=threshold)
                    best_node.branches["left"] = node1
                    best_node.branches["right"] = node2
        
        # If no valid split found, make a leaf
        if best_node is None:
            x = pd.DataFrame(columns=X.columns)
            x.loc[0] = X.iloc[0].copy()
            pred = forest.predict(x)[0]
            best_node = Node(label=pred)
            best_depth = 0
        
        self.memo[region_key] = (best_depth, best_node)
        return self.memo[region_key]
    
    def select_features(self, X, forest, max_features=10):
        importances = {}
        rf_preds = forest.predict(X)
        
        mi_scores = mutual_info_classif(X, rf_preds, random_state=42)
        for i, feature in enumerate(self.feature_names):
            importances[feature] = mi_scores[i]
        
        # Sort and select top features with thresholds
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        top_features = []
        for feature, _ in sorted_features:
            if len(self.hyperplanes.get(feature, [])) > 0:
                top_features.append(feature)
                if len(top_features) >= max_features:
                    break
                    
        print(f"Selected {len(top_features)} features: {', '.join(top_features[:5])}")
        
        # Create pruned hyperplanes
        pruned_hyperplanes = {f: self.hyperplanes[f] for f in top_features if f in self.hyperplanes}
        return pruned_hyperplanes, top_features
    
    def fit(self, forest, X, max_features=10):
        self.feature_names = X.columns
        print("Extracting hyperplane levels from forest...")
        self.hyperplanes = self.get_hyperplane_levels(forest)
        
        print("Selecting important features...")
        self.hyperplanes, self.feature_names = self.select_features(X, forest, max_features)
        
        print("Building born-again tree...")
        zL = [1] * len(self.feature_names)
        zR = [len(self.hyperplanes.get(feat, [])) + 1 for feat in self.feature_names]
        
        _, self.tree = self.min_depth_tree(zL, zR, X, forest)
        print("Tree construction complete!")
        
        return self
    
    def predict_one(self, x, node=None):
        if node is None:
            node = self.tree
            
        # Leaf node, return label
        if node.label is not None:
            return node.label
            
        # Internal node, follow appropriate branch
        try:
            feature_val = x[node.attribute]
            if feature_val <= node.threshold:
                return self.predict_one(x, node.branches["left"])
            else:
                return self.predict_one(x, node.branches["right"])
        except:
            # Default if feature not found
            return 1
    
    def predict(self, X):
        X = pd.DataFrame(X) if isinstance(X, np.ndarray) else X
        return np.array([self.predict_one(x) for _, x in X.iterrows()])

def train_born_again_tree(forest_model, train_data_path, max_depth=None, max_features=10, post_prune=False):
    print("\nTraining Born Again Tree from Random Forest...")
    
    # Read and preprocess data
    train_df = pd.read_csv(train_data_path)
    X = dp.preprocess_data(train_df)
    y = train_df['Transported'].astype(int)
    
    # Use a sample for efficiency
    X_sample = X.sample(min(500, len(X)), random_state=42)
    
    # Create and fit the tree
    bat = BornAgainTree(max_depth=max_depth)
    bat.fit(forest_model, X_sample, max_features=max_features)
    
    # Evaluate performance
    print("\nEvaluating Born Again Tree performance...")
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    bat_preds = bat.predict(X_test)
    rf_preds = forest_model.predict(X_test)
    
    closeness = np.mean(bat_preds == rf_preds)
    accuracy = np.mean(bat_preds == y_test)
    print(f"Closeness to Random Forest: {closeness:.4f}")
    print(f"Accuracy on test data: {accuracy:.4f}")
    
    print("\nClassification Report (Born Again Tree):")
    print(classification_report(y_test, bat_preds, 
                              target_names=["Not Transported", "Transported"]))
    
    return bat

def predict(model, test_data_path):
    test_df = pd.read_csv(test_data_path)
    passenger_ids = test_df['PassengerId']
    
    X_test = dp.preprocess_data(test_df)
    predictions = model.predict(X_test)
    
    prediction_df = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Transported': pd.Series(predictions).astype(bool)
    })
    
    return prediction_df