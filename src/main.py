import models.decision_tree_stock as dtm
import models.decision_tree_id3 as id3
import models.decision_tree_c45 as c45
from utils.evaluation import save_tree_visualization
from graphviz import Digraph
from sklearn import tree
import os
import models.naive_bayes as nb

# File paths
train_path = "train.csv"  
test_path = "test.csv"

# Train models
print("Training Stock, ID3 and C4.5 Decision Tree models...")
stock_model = dtm.train_decision_tree(train_path)
id3_model = id3.train_decision_tree(train_path)
c45_model = c45.train_decision_tree(train_path)

# Visualize trees
save_tree_visualization(stock_model, "stock_decision_tree")
save_tree_visualization(id3_model, "id3_decision_tree")
save_tree_visualization(c45_model, "c45_decision_tree")

# Make predictions on test set
print("\nMaking predictions on test set...")
submission_df = dtm.predict(stock_model, test_path)
submission_df2 = id3.predict(id3_model, test_path)
submission_df3 = c45.predict(c45_model, test_path)

# Save predictions
submission_df.to_csv("decision_tree_predictions.csv", index=False)
print("\nPredictions saved to 'decision_tree_predictions.csv'")

# Run Naive Bayes Predictions - False means no graphing
NB = nb.NaiveBayes(train_path, test_path, True)