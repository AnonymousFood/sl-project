import models.decision_tree_stock as dtm
import models.decision_tree_id3 as id3

# File paths
train_path = "train.csv"  
test_path = "test.csv"

# Train model
print("Training Stock and ID3 Decision Tree model...")
stock_model = dtm.train_decision_tree(train_path)
id3_model = id3.train_decision_tree(train_path)

# Make predictions on test set
print("\nMaking predictions on test set...")
submission_df = dtm.predict(stock_model, test_path)
submission_df2 = id3.predict(id3_model, test_path)

# Save predictions
submission_df.to_csv("decision_tree_predictions.csv", index=False)
print("\nPredictions saved to 'decision_tree_predictions.csv'")