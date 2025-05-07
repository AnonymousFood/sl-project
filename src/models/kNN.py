
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
import pandas as pd
import utils.data_preprocessing as dp
import utils.visuals as vis
import math


class KNearestNeighbors:
    # n_components just checks if > or = to 0. If >0 it will run both N=2 and N=3
    def __init__(self, train_data_path, test_data_path, k: int, graph: bool, n_components: int = 2):
        self.n_components = n_components  # Number of PCA components to keep
        
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
        
        # KNN doesnt have hyperparameters to validate BUT our test data does not have expected results. 
        # Split training data into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        if k > 0:
            # Predict x_test using x_val
            y_val_predict = self.predict(x_val, x_train, y_train, False, k)
            # Print performance
            print(f'\n{k} Nearest Neighbors Performance:')
            # print("Accuracy:", accuracy_score(y_val, y_val_predict))
            print("Classification Report:")
            print(classification_report(y_val, y_val_predict))
        
        if n_components > 0:
            # Apply PCA for dimensionality reduction
            
            ## N = 2
            pca_2 = PCA(n_components=2)
            X_pca_2 = pca_2.fit_transform(X)  # Reduced dimension data
        
            # Print explained variance to show how much information is retained
            print(f"Explained variance by 2 components: {pca_2.explained_variance_ratio_}")  
            print("\nPCA Component Features Mapping:")
            for i, component in enumerate(pca_2.components_):
                print(f"\nComponent {i+1}:")
                for j, feature in enumerate(X.columns):  # X.columns contains original feature names
                    print(f"  {feature}: {component[j]:.4f}")     

            x_pca_train, x_pca_val, y_pca_train, y_pca_val = train_test_split(X_pca_2, Y, test_size=0.2, random_state=42)
            # PCA Predict x_test using x_val
            y_pca_val_predict_2 = self.predict(x_pca_val, x_pca_train, y_pca_train, True, k)
            # Print performance
            print(f'\n{k}: N= 2 Nearest Neighbors Performance:')
            # print("Accuracy:", accuracy_score(y_pca_val, y_pca_val_predict))
            print("Classification Report:")
            print(classification_report(y_pca_val, y_pca_val_predict_2))
            
            ## N = 3
            pca_3 = PCA(n_components=3)
            X_pca_3 = pca_3.fit_transform(X)  # Reduced dimension data
        
            # Print explained variance to show how much information is retained
            print(f"Explained variance by 3 components: {pca_3.explained_variance_ratio_}")  
            print("\nPCA Component Features Mapping:")
            for i, component in enumerate(pca_3.components_):
                print(f"\nComponent {i+1}:")
                for j, feature in enumerate(X.columns):  # X.columns contains original feature names
                    print(f"  {feature}: {component[j]:.4f}")     

            x_pca_3_train, x_pca_3_val, y_pca_3_train, y_pca_3_val = train_test_split(X_pca_3, Y, test_size=0.2, random_state=42)
            # PCA Predict x_test using x_val
            y_pca_val_predict_3 = self.predict(x_pca_3_val, x_pca_3_train, y_pca_3_train, True, k)
            # Print performance
            print(f'\n{k}: N= 3 Nearest Neighbors Performance:')
            # print("Accuracy:", accuracy_score(y_pca_val, y_pca_val_predict))
            print("Classification Report:")
            print(classification_report(y_pca_3_val, y_pca_val_predict_3))
            
        if graph and k > 0:
            # Visualize - ROC
            vis.roc(y_val, y_val_predict, f' {k}-NN')
            # # Visualize - Confusion Matrix
            vis.cm(y_val, y_val_predict, f' {k}-NN')
            
            if n_components > 0:
                ## N = 2
                # Visualize - ROC
                vis.roc(y_val, y_pca_val_predict_2, f' {k}-NN 2 Component PCA')
                # Visualize - Confusion Matrix
                vis.cm(y_val, y_pca_val_predict_2, f' {k}-NN 2 Component PCA')
                
                # # Component Heatmap
                self.vis_heatmap_components(f' 2', pca_2.components_, X.columns)
                # Component Variance Composition
                self.vis_pca_variance_composition(pca_2.explained_variance_ratio_)
                
                self.vis_2D_ScatterPlot(f' {k}-NN 2 Component PCA', X_pca_2, Y)
                
                
                ## N = 3
                # Visualize - ROC
                vis.roc(y_val, y_pca_val_predict_3, f' {k}-NN 3 Component PCA')
                # Visualize - Confusion Matrix
                vis.cm(y_val, y_pca_val_predict_3, f' {k}-NN 3 Component PCA')
                
                # # Component Heatmap
                self.vis_heatmap_components(f' 3', pca_3.components_, X.columns)
                # Component Variance Composition
                self.vis_pca_variance_composition(pca_3.explained_variance_ratio_)
                
                self.vis_3D_ScatterPlot(f'{k}-NN 3 Component PCA', X_pca_3, Y)
                
                vis.roc_compare(y_val, [y_val_predict, y_pca_val_predict_2, y_pca_val_predict_3], ['N=0', 'N=2', 'N=3'], f'{k}-NN')
                    
        if k == 0:
            err_train = []
            err_val = []
            k_range = range(1, int(math.sqrt(len(x_train))) + 1, 5)  # Step size of 3
            print(k_range)
            for k in k_range:
                print(k)
                # predict
                predict_train = self.predict(x_train, x_train, y_train, False, k)
                predict_val = self.predict(x_val, x_train, y_train, False, k)
                # Err
                err_train.append(1 - accuracy_score(y_train, predict_train))
                err_val.append(1 - accuracy_score(y_val, predict_val))
                
            optimal_k = k_range[np.argmin(err_val)]
            print(f"The optimal value of K is: {optimal_k}")
            self.vis_errs(err_train, err_val, k_range)
            
        
    # L2-Norm distance between rwo points
    # Sqrt of sum (xi1 - xi2)^2
    def euclidian_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def predict(self, test: pd.DataFrame, X_train: pd.DataFrame, Y_train: pd.DataFrame, isPca: bool, k: int):
        # Ensure that X_train and test only contain numeric data
        if not isPca:
            X_train_numeric = X_train.select_dtypes(include=[np.number]).to_numpy()
            test_numeric = test.select_dtypes(include=[np.number]).to_numpy()
        else: 
            X_train_numeric = X_train
            test_numeric = test
        
        distances = []
        for x in test_numeric:
            distances.append([self.euclidian_distance(np.array(x), np.array(x_t)) for x_t in X_train_numeric])
        
        predictions = []
        for d in distances:
            # Indices of k-NN
            k_indices = np.argsort(d)[:k]
            # Get labels associated to indices
            k_labels = [Y_train.iloc[i] for i in k_indices]
            # Predict most common label
            predictions.append(np.bincount(k_labels).argmax())
            
        return predictions
    
    def vis_2D_ScatterPlot(self, label, X_pca_2d, y):
        # 2D PCA visualization
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
        plt.colorbar()  # Show the color scale
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'2D PCA Visualization of {label}')
        plt.show()
        
    def vis_3D_ScatterPlot(self, label, X_pca_3d, y):
        # 3D PCA visualization
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Scatter plot
        sc = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=y, cmap='viridis', edgecolor='k', s=50)
        # Add labels and title
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        ax.set_title(f'3D PCA Visualization of {label}')
        # Show the color bar
        plt.colorbar(sc)
        plt.show()

    def vis_heatmap_components(self, label, components, attributes):
        
        # Create a heatmap to show the loadings of each attribute for each component
        plt.figure(figsize=(10, 6))
        sns.heatmap(components, annot=True, xticklabels=attributes, cmap="coolwarm", cbar=True)
        plt.xlabel('Attributes')
        plt.ylabel('Principal Components')
        plt.title(f'PCA Component Loadings for {label}')
        plt.show()
        
    def vis_pca_variance_composition(self, explained_variance):
        # Plot the explained variance of each principal component
        plt.figure(figsize=(8, 6))
        plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, color='b')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance by Principal Component')
        plt.xticks(range(1, len(explained_variance) + 1))
        plt.show()

    def vis_errs(self, err_train, err_val, k_range):
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, err_train, label='Training Error', color='blue')
        plt.plot(k_range, err_val, label='Validation Error', color='red')
        plt.xlabel('K (Number of Neighbors)')
        plt.ylabel('Error Rate')
        plt.title('Training and Validation Errors vs. K')
        plt.legend()
        plt.grid(True)
        plt.show()
