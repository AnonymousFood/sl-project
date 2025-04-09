import models.decision_tree as dtm

# File paths
train_path = "train.csv"  
test_path = "test.csv"

# Train model
print("Training Decision Tree model...")
model = dtm.train_decision_tree(train_path)

# Make predictions on test set
print("\nMaking predictions on test set...")
submission_df = dtm.predict(model, test_path)

# Save predictions
submission_df.to_csv("decision_tree_predictions.csv", index=False)
print("\nPredictions saved to 'decision_tree_predictions.csv'")