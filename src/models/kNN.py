
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import utils.data_preprocessing as dp
import utils.visuals as vis

class KNearestNeighbors:
    def __init__(self, train_data_path, test_data_path, k: int, graph: bool):
        self.K = k
        labels =  {}
        attributes = {}
        y_probs = {} # Pr(y)
        y_attribute_probs = {} # P(Ai = ai | y)
        
        # Read training data
        train_df = pd.read_csv(train_data_path)
        # Preprocess data
        X = dp.preprocess_data(train_df)
        # was transported?
        Y = train_df['Transported'].astype(int)
        
        # Naive Bayes doesnt have hyperparameters to validate BUT our test data does not have expected results. 
        # Split training data into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
        attributes = x_train.columns
        labels =  y_train.unique()
        
        # Predict x_test using x_val
        y_val_predict = self.predict(x_val, x_train, y_train)
        
        # Print performance
        print(f'\n{self.K} Nearest Neighbors Performance:')
        print("Accuracy:", accuracy_score(y_val, y_val_predict))
        print("\nClassification Report:")
        print(classification_report(y_val, y_val_predict))
        
        
    # L2-Norm distance between rwo points
    # Sqrt of sum (xi1 - xi2)^2
    def euclidian_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def predict(self, test: pd.DataFrame, X_train: pd.DataFrame, Y_train: pd.DataFrame):
        # Ensure that X_train and test only contain numeric data
        X_train_numeric = X_train.select_dtypes(include=[np.number])
        test_numeric = test.select_dtypes(include=[np.number])
        
        distances = []
        for x in test_numeric.itertuples(index=False):
            distances.append([self.euclidian_distance(np.array(x), np.array(x_t)) for x_t in X_train_numeric.values])
        
        predictions = []
        for d in distances:
            # Indices of k-NN
            k_indices = np.argsort(d)[:self.K]
            # Get labels associated to indices
            k_labels = [Y_train.iloc[i] for i in k_indices]
            # Predict most common label
            predictions.append(np.bincount(k_labels).argmax())
            
        return predictions