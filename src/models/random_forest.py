import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.decision_tree_c45 import C45DecisionTree
import utils.data_preprocessing as dp

class RandomForest:
    def __init__(self, n_trees=100, max_depth=None, min_samples_split=2, class_weight=None, max_features='sqrt'):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []  # list to store trained trees
        self.class_weight = class_weight
        self.max_features = max_features
        self.oob_score_ = None  # Store OOB score after fitting
        self.oob_decision_function_ = None  # Store OOB predictions

    def fit(self, X, y):
        X = pd.DataFrame(X) if isinstance(X, np.ndarray) else X
        y = pd.Series(y) if isinstance(y, np.ndarray) else y
        
        # Get class distribution for reporting
        class_counts = Counter(y)
        weight_type = "balanced" if self.class_weight == 'balanced' else \
                     "custom" if isinstance(self.class_weight, dict) else "none"
        print(f"\nTraining Random Forest with {self.n_trees} trees (class weight: {weight_type})...")
        
        self.feature_names = X.columns.tolist()
        
        n_samples = len(X)
        oob_samples = [[] for _ in range(self.n_trees)] # Stores OOB samples for each tree
        self.oob_predictions = np.zeros((n_samples, 2))
        
        # Function to train a single tree
        def train_tree(tree_idx):
            # Bagging with tracking of OOB samples
            indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
            X_sample = X.iloc[indices]
            y_sample = y.iloc[indices]
            
            # Track OOB samples
            unique_indices = set(indices)
            oob_idx = [i for i in range(n_samples) if i not in unique_indices]
            
            # Feature subsampling based on selected strategy
            total_features = X.shape[1]
            
            # Determine number of features to use
            if self.max_features == 'sqrt':
                n_features = int(np.sqrt(total_features))
            elif self.max_features == 'log2':
                n_features = int(np.log2(total_features))
            elif isinstance(self.max_features, int):
                n_features = min(self.max_features, total_features)  # Max of total features
            elif isinstance(self.max_features, float) and 0.0 < self.max_features <= 1.0:
                n_features = max(1, int(self.max_features * total_features))  # Min of 1 feature
            else:
                n_features = total_features  # Use all features
                
            # Select features
            feature_indices = np.random.choice(total_features, size=n_features, replace=False)
            feature_names = [X.columns[i] for i in feature_indices]
            X_sample_features = X_sample.iloc[:, feature_indices]
            
            # Create and fit tree with class weights
            tree = C45DecisionTree(
                max_depth=self.max_depth, 
                min_samples_split=self.min_samples_split,
                class_weight=self.class_weight
            )
            tree.fit(X_sample_features, y_sample)
            
            return (tree, feature_indices, feature_names, oob_idx)
        
        # Multiprocessing!
        n_jobs = max(1, multiprocessing.cpu_count() - 1)
        print(f"Training trees in parallel using {n_jobs} CPU cores...")
        
        # Do the Multiprocessing!
        self.trees_with_oob = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(train_tree)(i) for i in range(self.n_trees)
        )
        
        self.trees = [(t[0], t[1], t[2]) for t in self.trees_with_oob]
        oob_samples = [t[3] for t in self.trees_with_oob]
        
        # OOB Error Estimation
        print("Computing out-of-bag error estimation...")
        # Extract tree objects
        trees = [tree_info[0] for tree_info in self.trees]
        feature_names_by_tree = [tree_info[2] for tree_info in self.trees]
        
        # For each tree, predict on its OOB samples
        for tree_idx, tree in enumerate(trees):
            # Get OOB samples for this tree
            oob_idx = oob_samples[tree_idx]
            if not oob_idx:
                continue  # Skip if no OOB samples
                
            # Get OOB data
            X_oob = X.iloc[oob_idx]
            
            # Get feature names for this tree
            tree_feature_names = feature_names_by_tree[tree_idx]
            
            # Predict on OOB samples
            for i, sample_idx in enumerate(oob_idx):
                # Extract features for this sample
                sample_data = X_oob.iloc[i]
                sample_features = pd.DataFrame([sample_data[tree_feature_names].values], 
                                            columns=tree_feature_names)
                
                # Get prediction
                pred = tree.predict(sample_features)[0]
                
                # Update OOB predictions
                self.oob_predictions[sample_idx, 0] += 1  # Increment vote count
                self.oob_predictions[sample_idx, 1] += pred  # Add prediction
        
        # Calculate final OOB predictions and score
        oob_final_predictions = np.zeros(n_samples)
        for i in range(n_samples):
            if self.oob_predictions[i, 0] > 0:  # If sample has OOB predictions
                oob_final_predictions[i] = round(self.oob_predictions[i, 1] / self.oob_predictions[i, 0])
            else:
                oob_final_predictions[i] = round(y.mean()) # Else, use majority class
                
        valid_oob_indices = self.oob_predictions[:, 0] > 0
        if np.any(valid_oob_indices): # Calculate OOB score
            self.oob_score_ = np.mean(oob_final_predictions[valid_oob_indices] == y[valid_oob_indices])
            print(f"Out-of-bag score: {self.oob_score_:.4f}")
        else:
            self.oob_score_ = None
            print("No out-of-bag samples found, all variables used")
            
        # Store OOB decision function for later use
        self.oob_decision_function_ = oob_final_predictions
        
        # Check accuracy on a sample after training
        final_accuracy = self.score(X, y, sampling_fraction=0.2)
        print(f"\nRandom Forest training complete. Final accuracy: {final_accuracy:.4f}")
        return self

    def predict(self, X, show_progress=True):
        X = pd.DataFrame(X) if isinstance(X, np.ndarray) else X
        
        if len(self.trees) == 0:
            return np.array([])
            
        if show_progress and len(X) > 1:  # Only show for larger batches
            print(f"Making predictions on {len(X)} samples with {len(self.trees)} trees...")
        
        all_predictions = []
        
        # Separate tree objects and feature information
        trees = [tree_info[0] for tree_info in self.trees]
        feature_indices_by_tree = [tree_info[1] for tree_info in self.trees]
        feature_names_by_tree = [tree_info[2] for tree_info in self.trees]
        
        # Process each sample directly
        for _, row_data in X.iterrows():  # Process all rows at once
            votes = []
            
            # Process all trees for this sample
            for tree_idx, tree in enumerate(trees):
                # Get the feature names for this tree
                tree_feature_names = feature_names_by_tree[tree_idx]
                
                # Extract relevant features for this sample and tree
                sample_features = pd.DataFrame([row_data[tree_feature_names].values], 
                                              columns=tree_feature_names)
                
                votes.append(tree.predict(sample_features)[0])
            
            # Use majority voting
            prediction = Counter(votes).most_common(1)[0][0]
            all_predictions.append(prediction)
            
        return np.array(all_predictions)

    def score(self, X, y, sampling_fraction=0.1):
        if len(self.trees) == 0:
            return 0.0
            
        # Use sampling for speed during training
        sample_size = max(int(len(X) * sampling_fraction), 100)
        sample_indices = np.random.choice(len(X), size=sample_size, replace=False)
        X_sample = X.iloc[sample_indices]
        y_sample = y.iloc[sample_indices]
        
        predictions = self.predict(X_sample, show_progress=False)
        return np.mean(predictions == y_sample.values)

# this passes all the parameters to the RandomForest class above ^^^
def train_decision_trees(train_data_path, n_trees=100, max_depth=None, min_samples_split=5, 
                         class_weight=None, max_features='sqrt'):
    # Read training data
    print("Reading training data...")
    train_df = pd.read_csv(train_data_path)
    
    # Preprocess data
    print("Preprocessing data...")
    X = dp.preprocess_data(train_df)
    y = train_df['Transported'].astype(int)
    
    # Split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Calculate class distribution
    class_counts = Counter(y_train)
    if class_weight == 'balanced':
        print(f"Using balanced class weights: {dict(class_counts)}")
    elif isinstance(class_weight, dict):
        print(f"Using custom class weights: {class_weight}")
    else:
        print(f"Using equal class weights")
    
    # Display feature selection strategy
    if max_features == 'sqrt':
        feature_strategy = f"sqrt({X.shape[1]}) = {int(np.sqrt(X.shape[1]))}"
    elif max_features == 'log2':
        feature_strategy = f"log2({X.shape[1]}) = {int(np.log2(X.shape[1]))}"
    elif isinstance(max_features, int):
        feature_strategy = f"{max_features}"
    elif isinstance(max_features, float):
        feature_strategy = f"{int(max_features * X.shape[1])} ({max_features*100}%)"
    else:
        feature_strategy = f"all ({X.shape[1]})"
        
    print(f"Feature selection strategy: {feature_strategy} features per tree")
    
    print("Training Random Forest model...")
    rf_classifier = RandomForest(n_trees=n_trees, max_depth=max_depth, 
                               min_samples_split=min_samples_split, 
                               class_weight=class_weight,
                               max_features=max_features)
    rf_classifier.fit(X_train, y_train)
    
    # Make predictions and evaluate
    print("\nEvaluating model on validation set...")
    rf_results = evaluate_model(rf_classifier, X_val, y_val, title="Random Forest Validation")
    
    # Print OOB vs validation comparison
    if hasattr(rf_classifier, 'oob_score_') and rf_classifier.oob_score_ is not None:
        print(f"\nPerformance summary:")
        print(f"- Out-of-bag score: {rf_classifier.oob_score_:.4f}")
        print(f"- Validation score: {rf_results['accuracy']:.4f}")
        
    return rf_classifier

def predict(model, test_data_path):
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

def evaluate_model(model, X, y, title="Random Forest Model"):
    print(f"\n{title} Evaluation")
    print("=" * 50)
    
    # Generate predictions
    y_pred = model.predict(X)
    
    acc = accuracy_score(y, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Not Transported', 'Transported'], digits=4))

    cm = confusion_matrix(y, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Transported', 'Transported'],
                yticklabels=['Not Transported', 'Transported'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix - {title}')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f"{title.lower().replace(' ', '_')}_confusion_matrix.png")
    print(f"Confusion matrix saved as '{title.lower().replace(' ', '_')}_confusion_matrix.png'")
    
    return {
        'accuracy': acc,
        'confusion_matrix': cm,
        'y_pred': y_pred
    }