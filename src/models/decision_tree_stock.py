import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.data_preprocessing as dp

def train_decision_tree(train_data_path): # Training part
    # Read training data
    train_df = pd.read_csv(train_data_path)
    
    # Preprocess data
    X = dp.preprocess_data(train_df)
    
    # was transported?
    y = train_df['Transported'].astype(int)
    
    # Split training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train model
    dt_classifier = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=5,
        random_state=42
    )
    
    dt_classifier.fit(X_train, y_train)
    
    # Make predictions on validation set
    y_pred = dt_classifier.predict(X_val)
    
    # Print performance
    print("\nStock/CART Decision Tree Model Performance:")
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, digits=4))
    
    return dt_classifier

def predict(model, test_data_path): # Testing part
    # Read test data
    test_df = pd.read_csv(test_data_path)
    passenger_ids = test_df['PassengerId']
    
    # Preprocess test data
    X_test = dp.preprocess_data(test_df)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Create submission dataframe for export to csv
    submission_df = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Transported': predictions.astype(bool)
    })
    
    return submission_df