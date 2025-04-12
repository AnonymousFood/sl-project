import models.decision_tree_stock as dtm
import models.decision_tree_id3 as id3
import models.decision_tree_c45 as c45
# from utils.evaluation import compare_models

# File paths
train_path = "train.csv"  
test_path = "test.csv"

# Train model
print("Training Stock and ID3 Decision Tree model...")
stock_model = dtm.train_decision_tree(train_path)
id3_model = id3.train_decision_tree(train_path)
c45_model = c45.train_decision_tree(train_path)

# Make predictions on test set
print("\nMaking predictions on test set...")
submission_df = dtm.predict(stock_model, test_path)
submission_df2 = id3.predict(id3_model, test_path)
submission_df3 = c45.predict(c45_model, test_path)

# # Compare model performances
# metrics_df = compare_models(test_path, 
#                           submission_df['Transported'],
#                           submission_df2['Transported'],
#                           submission_df3['Transported'])

# Save predictions
submission_df.to_csv("decision_tree_predictions.csv", index=False)
print("\nPredictions saved to 'decision_tree_predictions.csv'")