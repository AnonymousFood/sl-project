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
    def __init__(self, n_trees=100, max_depth=10, min_samples_split=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []  # list to store trained trees

    def fit(self, X, y):
        X = pd.DataFrame(X) if isinstance(X, np.ndarray) else X
        y = pd.Series(y) if isinstance(y, np.ndarray) else y
        
        print(f"\nTraining Random Forest with {self.n_trees} trees...")
        self.feature_names = X.columns.tolist()
        
        # Function to train a single tree
        def train_tree(tree_idx):
            # Bagging
            indices = np.random.choice(range(len(X)), size=len(X), replace=True)
            X_sample = X.iloc[indices]
            y_sample = y.iloc[indices]
            
            # Feature subsampling (sqrt n)
            n_features = int(np.sqrt(X.shape[1]))
            feature_indices = np.random.choice(X.shape[1], size=n_features, replace=False)
            feature_names = [X.columns[i] for i in feature_indices]
            X_sample_features = X_sample.iloc[:, feature_indices]
            
            # Create and fit tree
            tree = C45DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample_features, y_sample)
            
            return (tree, feature_indices, feature_names)
        
        # Multiprocessing!
        n_jobs = max(1, multiprocessing.cpu_count() - 1)
        print(f"Training trees in parallel using {n_jobs} CPU cores...")
        
        # Train trees in parallel
        self.trees = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(train_tree)(i) for i in range(self.n_trees)
        )
        
        # Check accuracy on a sample after training
        final_accuracy = self.score(X, y, sampling_fraction=0.2)
        print(f"\nRandom Forest training complete. Final accuracy: {final_accuracy:.4f}")
        return self

    def predict(self, X, show_progress=True):
        X = pd.DataFrame(X) if isinstance(X, np.ndarray) else X
        
        if len(self.trees) == 0:
            return np.array([])
            
        if show_progress:
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

def train_decision_trees(train_data_path, n_trees=100):
    # Read training data
    print("Reading training data...")
    train_df = pd.read_csv(train_data_path)
    
    # Preprocess data
    print("Preprocessing data...")
    X = dp.preprocess_data(train_df)
    y = train_df['Transported'].astype(int)
    
    # Split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest model...")
    rf_classifier = RandomForest(n_trees=n_trees)
    rf_classifier.fit(X_train, y_train)
    
    # Make predictions and evaluate
    print("\nEvaluating model on validation set...")
    rf_results = evaluate_model(rf_classifier, X_val, y_val, title="Random Forest Validation")
    
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