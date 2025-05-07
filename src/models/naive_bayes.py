import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import utils.data_preprocessing as dp
import utils.visuals as vis

class NaiveBayes:
    def __init__(self, train_data_path, test_data_path, graph: bool):
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
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        attributes = x_train.columns
        labels =  y_train.unique()
        
        # y_probs : Pr(y)
        # y_attribute_probs : P(Ai = ai | y) directly
        y_probs, y_attribute_probs, y_attribute_means, y_attribute_variances = self.get_probs_direct(x_train, y_train, attributes, labels)
        
        # Make predictions with natural log Naive Bayes
        y_pred_ln = self.predict_natural_log(x_test, y_probs, y_attribute_probs, attributes, labels)
        # Print performance
        print("\nNaive Bayes Log Performance:")
        print("Accuracy:", accuracy_score(y_test, y_pred_ln))
        print("Classification Report:")
        print(classification_report(y_test, y_pred_ln))
        print("AUC:", roc_auc_score(y_test, y_pred_ln))
        
        # Make predictions with product Naive Bayes
        y_pred_product = self.predict_product(x_test, y_probs, y_attribute_probs, attributes, labels)
        # Print performance
        print("\nNaive Bayes Product Performance:")
        print("Accuracy:", accuracy_score(y_test, y_pred_product))
        print("Classification Report:")
        print(classification_report(y_test, y_pred_product))
        print("AUC:", roc_auc_score(y_test, y_pred_product))
        
        # Make predictions with Natural Log - Gaussian Naive Bayes
        y_pred_gaussian_ln = self.predict_Gaussian_ln(x_test, y_probs, y_attribute_means, y_attribute_variances, attributes, labels)
        # Print performance
        print("\nNaive Bayes Gaussian Natural Log Performance:")
        print("Accuracy:", accuracy_score(y_test, y_pred_gaussian_ln))
        print("Classification Report:")
        print(classification_report(y_test, y_pred_gaussian_ln))
        print("AUC:", roc_auc_score(y_test, y_pred_gaussian_ln))
        
        # Make predictions with Product - Gaussian Naive Bayes
        y_pred_gaussian_prod = self.predict_Gaussian_product(x_test, y_probs, y_attribute_means, y_attribute_variances, attributes, labels)
        # Print performance
        print("\nNaive Bayes Gaussian Product Performance:")
        print("Accuracy:", accuracy_score(y_test, y_pred_gaussian_prod))
        print("Classification Report:")
        print(classification_report(y_test, y_pred_gaussian_prod))
        print("AUC:", roc_auc_score(y_test, y_pred_gaussian_prod))
        
        if (graph):
            # Visualize - ROC
            vis.roc(y_test, y_pred_ln, " Naive Bayes - ln - Multinomial")
            # vis.roc(y_test, y_pred_product, " Naive Bayes - Product")
            vis.roc(y_test, y_pred_gaussian_ln, " Naive Bayes -ln - Gaussian")
            # vis.roc(y_test, y_pred_gaussian_prod, " Naive Bayes - Gaussian Product")
            
            # Visualize - Confusion Matrix
            vis.cm(y_test, y_pred_ln, " Naive Bayes - ln - Multinomial")
            # vis.cm(y_test, y_pred_product, " Naive Bayes - Product")
            vis.cm(y_test, y_pred_gaussian_ln, " Naive Bayes - ln - Gaussian")
            # vis.cm(y_test, y_pred_gaussian_prod, " Naive Bayes - Gaussian - Product")
            
            # Gaussian Means of attributes
            means_arr = np.array([list(m.values()) for m in y_attribute_means.values()])
            vis.attribute_bar_compare(attributes, means_arr, "Mean")
            var_arr = np.array([list(m.values()) for m in y_attribute_variances.values()])
            vis.attribute_bar_compare(attributes, var_arr, "Variance")
            # probs_arr = np.array([list(m.values()) for m in y_attribute_probs.values()])
            # vis.attribute_bar_compare(attributes, probs_arr, "Probability")
            
            # Heatmap
            vis.visualize_heatmap(y_attribute_means, "Mean")
            vis.visualize_heatmap(y_attribute_variances, "Variance")
            
            vis.roc_compare(y_test, [y_pred_ln, y_pred_gaussian_ln], ['Multinomial', 'Gaussian'], " Naive Bayes")
        # vis.visualize_feature_influence(y_attribute_probs, attributes, labels)
        # vis.visualize_stacked_feature_contributions(y_attribute_probs, attributes, labels)
        
    def get_probs_direct(self, x: pd.DataFrame, y: pd.DataFrame, attributes, labels):
        y_probs = {} # Pr(y)
        y_attribute_probs = {} # P(Ai = ai | y)
        y_attribute_means = {} # mean_ij
        y_attribute_variances = {} # variance_ij
        
        for l in labels:
            # Get P(y) for all y's
            y_probs[l] = np.sum(l == y) / len(y)
            y_attribute_probs[l] = {}
            y_attribute_means[l] = {}
            y_attribute_variances[l] = {}
            
            # Get P(Ai=ai | y) for all attributes and y combos
            x_y = x[y == l]
            for a in attributes:
                y_attribute_probs[l][a] = x_y[a].value_counts(normalize=True).to_dict()
                x_y_values = x_y[a].values
                y_attribute_means[l][a] = np.mean(x_y_values)
                y_attribute_variances[l][a] = np.var(x_y_values, ddof=1)
        return y_probs, y_attribute_probs, y_attribute_means, y_attribute_variances
        
    # We have a high number of attributes -> more likely that product of their probs will varry
    # yNB = argmaxlnPr(y) + sum(ln(Pr(Ai = ai | y))
    def predict_natural_log(self, xs: pd.DataFrame, y_probs:dict, y_attribute_probs:dict, attributes, labels):
        predictions = []
        for _, x in xs.iterrows():
            probs = {}
            for l in labels:
                prob = np.log(y_probs[l]) # lnPr(y)
                for a in attributes: # sum(ln(Pr(Ai = ai | y))
                    x_a = x[a]
                    # If attribute doesn't exist use 1e-6
                    prob += np.log(y_attribute_probs[l][a].get(x_a,1e-6))
                probs[l] = prob
            predictions.append(max(probs, key=probs.get))
        return predictions
    
    # yNB = argmaxPr(y) * prod(Pr(Ai = ai | y))
    def predict_product(self, xs: pd.DataFrame, y_probs:dict, y_attribute_probs:dict, attributes, labels):
        predictions = []
        for _, x in xs.iterrows():
            probs = {}
            for l in labels:
                prob = y_probs[l] # Pr(y)
                for a in attributes: # prod(Pr(Ai = ai | y))
                    x_a = x[a]
                    # If attribute doesn't exist use 1e-6
                    prob *= y_attribute_probs[l][a].get(x_a,1e-6)
                probs[l] = prob
            predictions.append(max(probs, key=probs.get))
        return predictions

    # Pr(Ai = ai | yj) = 1/sqrt(2*pi*variance_ij) exp(-(1i-mean_ij)^2 / (2 * variance))
    # mean_ij = 1/m_j * sum(a_ik)
    # variance_ij = i/(m_j - 1) * sum((a_ik - mean_ij)^2)
    def predict_Gaussian_product(self, xs: pd.DataFrame, y_probs:dict, y_attribute_means:dict, y_attribute_variances:dict, attributes, labels):
        predictions = []
        for _, x in xs.iterrows():
            probs = {}
            for l in labels:
                prob = y_probs[l] # Pr(y)
                for a in attributes: # prod(Pr(Ai = ai | y))
                    x_a = x[a]
                    var = y_attribute_variances[l][a]
                    mean = y_attribute_means[l][a]
                    # If attribute doesn't exist use 1e-6
                    prob *= (1/ np.sqrt(2 * np.pi * var)) * np.exp(-(x_a - mean)**2 / (2 * var))
                probs[l] = prob
            predictions.append(max(probs, key=probs.get))
        return predictions
    
    # Pr(Ai = ai | yj) = 1/sqrt(2*pi*variance_ij) exp(-(1i-mean_ij)^2 / (2 * variance))
    # mean_ij = 1/m_j * sum(a_ik)
    # variance_ij = i/(m_j - 1) * sum((a_ik - mean_ij)^2)
    def predict_Gaussian_ln(self, xs: pd.DataFrame, y_probs:dict, y_attribute_means:dict, y_attribute_variances:dict, attributes, labels):
        predictions = []
        for _, x in xs.iterrows():
            probs = {}
            for l in labels:
                prob = np.log(y_probs[l]) # lnPr(y)
                for a in attributes: # sum(ln(Pr(Ai = ai | y))
                    x_a = x[a]
                    var = y_attribute_variances[l][a]
                    mean = y_attribute_means[l][a]
                    temp = (1/ np.sqrt(2 * np.pi * var)) * np.exp(-(x_a - mean)**2 / (2 * var))
                    prob += np.log(temp)
                probs[l] = prob
            predictions.append(max(probs, key=probs.get))
        return predictions


    