import models.decision_tree_stock as dtm
import models.decision_tree_id3 as id3
import models.decision_tree_c45 as c45
import models.random_forest as rf
from utils.evaluation import save_tree_visualization
from graphviz import Digraph
from sklearn import tree
import os
import models.naive_bayes as nb
from collections import Counter

# File paths
train_path = "train.csv"  
test_path = "test.csv"

# Train models
print("Training Stock, ID3 and C4.5 Decision Tree models...")
stock_model = dtm.train_decision_tree(train_path)
# id3_model = id3.train_decision_tree(train_path)
# c45_model = c45.train_decision_tree(train_path)

print("Training Random Forest models...")
rf_model = rf.train_decision_trees(train_path, n_trees=100)

# # Visualize trees
# save_tree_visualization(stock_model, "stock_decision_tree")
# save_tree_visualization(id3_model, "id3_decision_tree")
# save_tree_visualization(c45_model, "c45_decision_tree")

# Make predictions on test set
# print("\nMaking predictions on test set...")
submission_df = dtm.predict(stock_model, test_path)
# submission_df2 = id3.predict(id3_model, test_path)
# submission_df3 = c45.predict(c45_model, test_path)
submission_df4 = rf.predict(rf_model, test_path)

# Save predictions
# submission_df.to_csv("decision_tree_predictions.csv", index=False)
submission_df4.to_csv("random_forest_predictions.csv", index=False)
print("\nPredictions saved to 'decision_tree_predictions.csv' and 'random_forest_predictions.csv'")

# Run Naive Bayes Predictions - False means no graphing
NB = nb.NaiveBayes(train_path, test_path, True)

# Run K-Nearest Neighbors Predictions - False means no graphing - last number 0: no PCA, > 0: 2 and 3 PCA
## Find what K is best (less validation loss)
# knn_find_K = kn.KNearestNeighbors(train_path, test_path, 0, False, 0)
    # For 3 step intervals got 22, for 5 step intervals got 21
## Get specific data on K = 22 for N = 0,2,3
# knn_22_pca = kn.KNearestNeighbors(train_path, test_path, 22, True, 1)