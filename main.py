from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM 
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

class MyDataset:
    def __init__(self, file_path):
        print("Initializing dataset...")
        self.data = pd.read_csv(file_path)
        

        drop_columns = [
            'udp.checksum', 'icmp.type', 'icmp.code', 'icmp.checksum', 'http.location',
            'http.authorization', 'dns.qry.name', 'dns.qry.class', 'smtp.req.command',
            'smtp.data.fragment', 'pop.request.command', 'pop.response', 'imap.request.command',
            'imap.response', 'ftp.request.command', 'ftp.request.arg', 'ftp.response.code',
            'ipv6.src', 'ipv6.dst', 'http.cookie', 'http.referer', 'ftp.response.arg',
            'dns.flags.rcode', 'dns.resp.ttl', 'dns.resp.len', 'ipv6.plen'
        ]
        self.data = self.data.drop(columns=drop_columns, errors='ignore')
        
        self.numeric_features = [
            'frame.len', 'frame.time_epoch', 'ip.len', 'ip.ttl', 'tcp.len', 'tcp.seq',
            'tcp.ack', 'tcp.window_size', 'udp.length'
        ]
        self.categorical_features = [
            'frame.protocols', 'eth.src', 'eth.dst', 'ip.src', 'ip.dst', 'ip.flags',
            'ip.proto', 'tcp.flags', 'tcp.flags.syn', 'tcp.flags.ack', 'tcp.flags.fin',
            'tcp.flags.reset', 'http.request.method', 'http.request.uri', 'http.request.version',
            'http.request.full_uri', 'http.response.code', 'http.user_agent',
            'http.content_length_header', 'http.content_type', 'http.host', 'dns.qry.name',
            'dns.qry.type', 'dns.qry.class', 'dns.flags.response', 'dns.flags.recdesired'
        ]
        

        self.data_preprocessed, self.numeric_features, self.categorical_features = self.preprocess_data(self.data)
        self.svm_model = None
        self.data['placeholder_target'] = 0
    
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])
    
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])
    
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)])
    

        self.data_preprocessed = preprocessor.fit_transform(self.data)
        self.data_preprocessed = pd.DataFrame(self.data_preprocessed, index=self.data.index)
        print(f"Data preprocessing completed. Shape: {self.data_preprocessed.shape}")

        self.data_reduced = self.data_preprocessed
    
        
        self.model = None
        self.svm_model = None
        self.cluster_labels = None
        self.pca_result = None
        self.tsne_result = None
        self.pca_cluster_labels = None
        self.tsne_cluster_labels = None
    
        
        self.add_svm_anomaly_column()
        self.split_data()
        self.train_anomaly_detection_model()
        self.train_autoencoder()
        self.calculate_reconstruction_errors()
        self.evaluate_autoencoder()
        self.trainoneclasssvm()
        self.perform_clustering(self.data_preprocessed, 'preprocessed')  
        self.pca_result = self.perform_pca()  
        self.tsne_result = self.perform_tsne()  

    def load_data(self):
        print("Loading data...")
        chunks = []
        chunk_size = 10000
        columns_to_drop = [
            'udp.checksum', 'icmp.type', 'icmp.code', 'icmp.checksum', 'http.location',
            'http.authorization', 'dns.qry.name', 'dns.qry.class', 'smtp.req.command',
            'smtp.data.fragment', 'pop.request.command', 'pop.response', 'imap.request.command',
            'imap.response', 'ftp.request.command', 'ftp.request.arg', 'ftp.response.code',
            'ipv6.src', 'ipv6.dst','http.cookie','http.referer','ftp.response.arg'
        ]
        
        for chunk in pd.read_csv(self.file_path, chunksize=chunk_size, low_memory=False):
            print(f"Processing chunk with shape: {chunk.shape}")
            
            chunk = chunk.drop(columns=columns_to_drop, errors='ignore')
            chunks.append(chunk)
        
        data = pd.concat(chunks, ignore_index=True)
        print(f"Data loaded with shape: {data.shape}")
        return data
    
    def add_svm_anomaly_column(self):
        print("Adding 'svm_anomaly' column...")
        
        
        if not isinstance(self.data_preprocessed, pd.DataFrame):
            self.data_preprocessed = pd.DataFrame(self.data_preprocessed)
        
        
        self.data_preprocessed.columns = self.data_preprocessed.columns.astype(str)
        
        
        svm_model = OneClassSVM(gamma='auto').fit(self.data_reduced)
        svm_predictions = svm_model.predict(self.data_reduced)
        
        
        print(f"Length of data_reduced: {len(self.data_reduced)}")
        print(f"Length of svm_predictions: {len(svm_predictions)}")
        
        if len(svm_predictions) != len(self.data):
            raise ValueError(f"Length of svm_predictions ({len(svm_predictions)}) does not match length of data ({len(self.data)})")
        
        
        self.data['svm_anomaly'] = svm_predictions
        self.data['svm_anomaly'] = self.data['svm_anomaly'].map({1: 0, -1: 1})  
        print("'svm_anomaly' column added to data.")
    
    def split_data(self):
        X = self.data_preprocessed

        self.data['label'] = self.data['alert'].map({'benign': 0, 'suspicious': 1})
        
        
        if 'label' in self.data.columns:
            print("The 'label' column has been successfully created.")
        else:
            raise ValueError("The 'label' column was not created.")
        y = self.data['label']  
        if y.nunique() <= 1:
            raise ValueError("The target variable must contain more than one class.")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Testing set shape: {self.X_test.shape}")


    def preprocess_data(self, data):
        print("Preprocessing data...")
        
        data = data.dropna(axis=1, how='all')
        
        
        data = data.loc[:, data.notna().any(axis=0)]
        
        
        categorical_features = data.select_dtypes(include=['object']).columns
        data.loc[:, categorical_features] = data[categorical_features].astype(str)
        
        
        for col in data.columns:
            if data[col].dtype == 'object':
                data[col] = data[col].astype(str)
            elif data[col].dtype in ['int64', 'float64']:
                data[col] = data[col].astype(float)
        
        
        numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = data.select_dtypes(include=['object']).columns

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])

        
        data_preprocessed = preprocessor.fit_transform(data)
        data_preprocessed = pd.DataFrame(data_preprocessed, index=data.index)
        print(f"Data preprocessing completed. Shape: {data_preprocessed.shape}")


        return data_preprocessed, numeric_features, categorical_features

    def cross_validate_model(self, data, target):
        print("Cross-validating model...")
        
        
        numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = data.select_dtypes(include=['object']).columns
    
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])
    
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])
    
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])
    
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('variance_threshold', VarianceThreshold(threshold=0.01)),
            ('classifier', LogisticRegression(max_iter=20000, random_state=100))  
        ])
    
        
        initial_scores = cross_val_score(pipeline, data, target, cv=5)
        print(f"Initial cross-validation scores: {initial_scores}")
        print(f"Mean initial cross-validation score: {np.mean(initial_scores)}")
    
        
        param_grid = {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__solver': ['liblinear', 'lbfgs']
        }
    
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(data, target)
    
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_}")
    
        
        best_params = grid_search.best_params_
        best_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('variance_threshold', VarianceThreshold(threshold=0.01)),
            ('classifier', LogisticRegression(max_iter=20000, random_state=100, 
                                              C=best_params['classifier__C'], 
                                              solver=best_params['classifier__solver']))
        ])
    
        return best_pipeline

    def create_label_column(self):
        
        self.data['label'] = self.data['alert'].map({'benign': 0, 'suspicious': 1})

    def train_anomaly_detection_model(self):
        print("Training anomaly detection model...")
        
        
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.model.fit(self.X_train)
        predictions = self.model.predict(self.X_train)
        
        
        print(f"Length of X_train: {len(self.X_train)}")
        print(f"Length of predictions: {len(predictions)}")
        
        if len(predictions) != len(self.X_train):
            raise ValueError(f"Length of predictions ({len(predictions)}) does not match length of X_train ({len(self.X_train)})")
        
        
        self.data.loc[self.X_train.index, 'anomaly'] = predictions
        self.data['anomaly'] = self.data['anomaly'].map({1: 0, -1: 1})  
        print("Anomaly detection model trained,")

    def trainoneclasssvm(self):
        print("Training One-Class SVM model...")
        
        
        if not isinstance(self.data_preprocessed, pd.DataFrame):
            self.data_preprocessed = pd.DataFrame(self.data_preprocessed)
        
        
        self.data_preprocessed.columns = self.data_preprocessed.columns.astype(str)
        
        
        self.svm_model = OneClassSVM(gamma='auto').fit(self.X_train)
        self.svm_predictions_train = self.svm_model.predict(self.X_train)
        self.svm_predictions_test = self.svm_model.predict(self.X_test)
        
        
        print(f"Length of X_train: {len(self.X_train)}")
        print(f"Length of svm_predictions_train: {len(self.svm_predictions_train)}")
        
        if len(self.svm_predictions_train) != len(self.X_train):
            raise ValueError(f"Length of svm_predictions_train ({len(self.svm_predictions_train)}) does not match length of X_train ({len(self.X_train)})")
        
        
        self.data.loc[self.X_train.index, 'svm_anomaly'] = self.svm_predictions_train
        self.data['svm_anomaly'] = self.data['svm_anomaly'].map({1: 0, -1: 1})  
        print("One-Class SVM model trained")

    def perform_clustering(self, data, data_type='preprocessed'):
        print(f"Performing clustering on {data_type} data...")
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        self.cluster_labels = kmeans.fit_predict(data)
        print(f"Clustering completed. Number of clusters: {len(set(self.cluster_labels))}")

    def perform_pca(self):
        print("Performing PCA...")
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.data_preprocessed)
        print("PCA completed.")
        return pca_result

    def perform_tsne(self):
        print("Performing t-SNE...")
        tsne = TSNE(n_components=2, random_state=99)
        tsne_result = tsne.fit_transform(self.data_preprocessed)
        print("t-SNE completed.")
        return tsne_result

    def save_plots(self):
        print("Saving plots...")
        
        numeric_features_list = list(self.numeric_features)
        
        
        print(f"Length of data_reduced: {len(self.data_reduced)}")
        
        if self.cluster_labels is not None:
            print(f"Length of cluster_labels: {len(self.cluster_labels)}")
        else:
            print("cluster_labels is None")
            return  
        
        if self.pca_result is not None:
            print(f"Length of pca_result: {len(self.pca_result)}")
        else:
            print("pca_result is None")
        
        print(f"Length of data: {len(self.data)}")
        
        
        if len(self.data_reduced) != len(self.cluster_labels):
            raise ValueError(f"Length of data_reduced ({len(self.data_reduced)}) does not match length of cluster_labels ({len(self.cluster_labels)})")
        if self.pca_result is not None and len(self.pca_result) != len(self.data):
            raise ValueError(f"Length of pca_result ({len(self.pca_result)}) does not match length of data ({len(self.data)})")
        
        
        if 'svm_anomaly' not in self.data.columns:
            raise KeyError("'svm_anomaly' column not found in data")
        
        
        feature_indices_1 = [numeric_features_list.index('ip.len'), numeric_features_list.index('tcp.srcport')]
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.data_reduced.iloc[:, feature_indices_1[0]], y=self.data_reduced.iloc[:, feature_indices_1[1]], hue=self.cluster_labels, palette='viridis', s=100, alpha=0.6, edgecolor='w')
        plt.title('Clusters of Attack Simulation Alerts (IP Length vs TCP Source Port)')
        plt.xlabel('IP Length (Standardized)')
        plt.ylabel('TCP Source Port (Standardized)')
        plt.legend(title='Cluster')
        plt.grid(True)
        plt.savefig('clusters_ip_len_vs_tcp_srcport.png')
        plt.close()
    
        
        feature_indices_2 = [numeric_features_list.index('ip.len'), numeric_features_list.index('tcp.dstport')]
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.data_reduced.iloc[:, feature_indices_2[0]], y=self.data_reduced.iloc[:, feature_indices_2[1]], hue=self.cluster_labels, palette='viridis', s=100, alpha=0.6, edgecolor='w')
        plt.title('Clusters of Attack Simulation Alerts (IP Length vs TCP Destination Port)')
        plt.xlabel('IP Length (Standardized)')
        plt.ylabel('TCP Destination Port (Standardized)')
        plt.legend(title='Cluster')
        plt.grid(True)
        plt.savefig('clusters_ip_len_vs_tcp_dstport.png')
        plt.close()
    
        
        feature_indices_3 = [numeric_features_list.index('ip.ttl'), numeric_features_list.index('tcp.window_size')]
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.data_reduced.iloc[:, feature_indices_3[0]], y=self.data_reduced.iloc[:, feature_indices_3[1]], hue=self.cluster_labels, palette='viridis', s=100, alpha=0.6, edgecolor='w')
        plt.title('Clusters of Attack Simulation Alerts (IP TTL vs TCP Window Size)')
        plt.xlabel('IP TTL (Standardized)')
        plt.ylabel('TCP Window Size (Standardized)')
        plt.legend(title='Cluster')
        plt.grid(True)
        plt.savefig('clusters_ip_ttl_vs_tcp_window_size.png')
        plt.close()
    
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.pca_result[:, 0], y=self.pca_result[:, 1], hue=self.pca_cluster_labels, palette='viridis', s=100, alpha=0.6, edgecolor='w')
        plt.title('PCA of Attack Simulation Alerts (Derived from Original Features)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend(title='Cluster')
        plt.grid(True)
        plt.savefig('pca_plot.png')
        plt.close()
    
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.tsne_result[:, 0], y=self.tsne_result[:, 1], hue=self.tsne_cluster_labels, palette='viridis', s=100, alpha=0.6, edgecolor='w')
        plt.title('t-SNE of Attack Simulation Alerts (Derived from Original Features)')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend(title='Cluster')
        plt.grid(True)
        plt.savefig('tsne_plot.png')
        plt.close()
    
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.pca_result[:, 0], y=self.pca_result[:, 1], hue=self.data['anomaly'], palette='coolwarm', s=100, alpha=0.6, edgecolor='w')
        plt.title('PCA of Attack Simulation Alerts (Normal vs Anomaly)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend(title='Anomaly')
        plt.grid(True)
        plt.savefig('pca_normal_vs_anomaly.png')
        plt.close()
    
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.tsne_result[:, 0], y=self.tsne_result[:, 1], hue=self.data['anomaly'], palette='coolwarm', s=100, alpha=0.6, edgecolor='w')
        plt.title('t-SNE of Attack Simulation Alerts (Normal vs Anomaly)')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend(title='Anomaly')
        plt.grid(True)
        plt.savefig('tsne_normal_vs_anomaly.png')
        plt.close()
    
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.pca_result[:, 0], y=self.pca_result[:, 1], hue=self.data['svm_anomaly'], palette='coolwarm', s=100, alpha=0.6, edgecolor='w')
        plt.title('PCA of Attack Simulation Alerts (Normal vs Anomaly - One-Class SVM)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend(title='Anomaly')
        plt.grid(True)
        plt.savefig('pca_normal_vs_anomaly_one_class_svm.png')
        plt.close()
        
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.tsne_result[:, 0], y=self.tsne_result[:, 1], hue=self.data['svm_anomaly'], palette='coolwarm', s=100, alpha=0.6, edgecolor='w')
        plt.title('t-SNE of Attack Simulation Alerts (Normal vs Anomaly - One-Class SVM)')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend(title='Anomaly')
        plt.grid(True)
        plt.savefig('tsne_normal_vs_anomaly_one_class_svm.png')
        plt.close()
    
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.pca_result[:, 0], y=self.pca_result[:, 1], hue=self.data['autoencoder_anomaly'], palette='coolwarm', s=100, alpha=0.6, edgecolor='w')
        plt.title('PCA of Attack Simulation Alerts (Normal vs Anomaly - Autoencoder)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend(title='Anomaly')
        plt.grid(True)
        plt.savefig('pca_normal_vs_anomaly_autoencoder.png')
        plt.close()
        
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.tsne_result[:, 0], y=self.tsne_result[:, 1], hue=self.data['autoencoder_anomaly'], palette='coolwarm', s=100, alpha=0.6, edgecolor='w')
        plt.title('t-SNE of Attack Simulation Alerts (Normal vs Anomaly - Autoencoder)')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend(title='Anomaly')
        plt.grid(True)
        plt.savefig('tsne_normal_vs_anomaly_autoencoder.png')
        plt.close()
    
        
        # plt.figure(figsize=(10, 6))
        # sns.scatterplot(x=self.pca_result[:, 0], y=self.pca_result[:, 1], hue=self.data['ensemble'], palette='coolwarm', s=100, alpha=0.6, edgecolor='w')
        # plt.title('PCA of Attack Simulation Alerts (Normal vs Anomaly - Ensemble)')
        # plt.xlabel('PCA Component 1')
        # plt.ylabel('PCA Component 2')
        # plt.legend(title='Anomaly')
        # plt.grid(True)
        # plt.savefig('pca_normal_vs_anomaly_ensemble.png')
        # plt.close()
        
        
        # plt.figure(figsize=(10, 6))
        # sns.scatterplot(x=self.tsne_result[:, 0], y=self.tsne_result[:, 1], hue=self.data['ensemble'], palette='coolwarm', s=100, alpha=0.6, edgecolor='w')
        # plt.title('t-SNE of Attack Simulation Alerts (Normal vs Anomaly - Ensemble)')
        # plt.xlabel('t-SNE Component 1')
        # plt.ylabel('t-SNE Component 2')
        # plt.legend(title='Anomaly')
        # plt.grid(True)
        # plt.savefig('tsne_normal_vs_anomaly_ensemble.png')
        # plt.close()

    def plot_dendrogram(self):
        print("Saving dendrogram...")
        linked = linkage(self.data_preprocessed, 'ward')
        plt.figure(figsize=(10, 6))
        dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
        plt.title('Dendrogram')
        plt.xlabel('Sample index')
        plt.ylabel('Distance')
        plt.savefig('dendrogram.png')
        plt.close()

    def calculate_reconstruction_errors(self):
        
        self.data_reconstructed = self.autoencoder.predict(self.data_preprocessed)
        
        
        self.reconstruction_errors = np.mean(np.power(self.data_preprocessed - self.data_reconstructed, 2), axis=1)
        
        
        self.threshold = np.percentile(self.reconstruction_errors, 95)
        print(f"Reconstruction error threshold: {self.threshold}")
        
        
        self.data['autoencoder_anomaly'] = (self.reconstruction_errors > self.threshold).astype(int)

    def evaluate_model(self):
        print("Evaluating model...")
        
        
        if not isinstance(self.data_preprocessed, pd.DataFrame):
            self.data_preprocessed = pd.DataFrame(self.data_preprocessed)
        
        
        self.data_preprocessed.columns = self.data_preprocessed.columns.astype(str)
        
        
        data_for_evaluation = self.data_preprocessed
        
        
        print(f"Shape of data_preprocessed: {self.data_preprocessed.shape}")
        print(f"Shape of data_for_evaluation: {data_for_evaluation.shape}")
        
        
        self.data_preprocessed['anomaly'] = self.model.predict(data_for_evaluation)
        self.data_preprocessed['anomaly'] = self.data_preprocessed['anomaly'].map({1: 0, -1: 1})  
        
        

    def trainisoforest(self):
        print("Training Isolation Forest...")
        param_grid = {
            'n_estimators': [50],
            'max_samples': ['auto'],
            'contamination': [0.1],
            'max_features': [1.0],
            'bootstrap': [True]
        }
        grid_search = GridSearchCV(IsolationForest(random_state=42), param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)
        self.isolation_forest = grid_search.best_estimator_
        self.isolation_forest_predictions_train = self.isolation_forest.predict(self.X_train)
        self.isolation_forest_predictions_test = self.isolation_forest.predict(self.X_test)
        print("Isolation Forest trained with best parameters:", grid_search.best_params_)

    def trainkmeans(self):
        print("Training KMeans...")
        param_grid = {
            'n_clusters': [2],
            'init': ['random'],
            'n_init': [10],
            'max_iter': [600]
        }
        grid_search = GridSearchCV(KMeans(random_state=42), param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)
        self.kmeans = grid_search.best_estimator_
        self.kmeans_predictions_train = self.kmeans.predict(self.X_train)
        self.kmeans_predictions_test = self.kmeans.predict(self.X_test)
        print("KMeans trained with best parameters:", grid_search.best_params_)

    def combine_predictions(self):
        print("Combining predictions from Isolation Forest and KMeans...")
        self.X_train_combined = np.column_stack((self.isolation_forest_predictions_train, self.kmeans_predictions_train))
        self.X_test_combined = np.column_stack((self.isolation_forest_predictions_test, self.kmeans_predictions_test))
        print("Predictions combined.")
        
        print("Combined Predictions (Train):", self.X_train_combined)

    def combine_predictions2(self):
        print("Combining predictions from Isolation Forest and One-Class SVM...")
        self.X_train_combined = np.column_stack((self.isolation_forest_predictions_train, self.svm_predictions_train))
        self.X_test_combined = np.column_stack((self.isolation_forest_predictions_test, self.svm_predictions_test))
        print("Predictions combined.")

    def combine_predictions3(self):
        print("Combining predictions from different models...")
        self.X_train_combined = np.column_stack((self.svm_predictions_train, self.kmeans_predictions_train))
        self.X_test_combined = np.column_stack((self.svm_predictions_test, self.kmeans_predictions_test))
        print("Predictions combined.")

    def train_meta_model3(self):
        print("Training meta-model...")
        self.meta_model = LogisticRegression(random_state=42)
        self.meta_model.fit(self.X_train_combined, self.y_train)
        print("Meta-model trained.")

    def evaluate_meta_model3(self):
        print("Evaluating meta-model...")
        y_pred = self.meta_model.predict(self.X_test_combined)
        print("Classification Report (Meta-Model):")
        print(classification_report(self.y_test, y_pred, zero_division=0))
        print("Confusion Matrix (Meta-Model):")
        print(confusion_matrix(self.y_test, y_pred))

    def train_meta_model2(self):
        print("Training meta-model...")
        self.meta_model = LogisticRegression(random_state=42)
        self.meta_model.fit(self.X_train_combined, self.y_train)
        print("Meta-model trained.")

    def evaluate_meta_model2(self):
        print("Evaluating meta-model...")
        y_pred = self.meta_model.predict(self.X_test_combined)
        print("Classification Report (Meta-Model):")
        print(classification_report(self.y_test, y_pred, zero_division=0))
        print("Confusion Matrix (Meta-Model):")
        print(confusion_matrix(self.y_test, y_pred))

    def run_ensemble_model2(self):
        self.trainisoforest()
        self.train_one_class_svm2()  
        self.combine_predictions2()
        self.train_meta_model2()
        self.evaluate_meta_model2()

    def train_one_class_svm2(self):
        print("Training One-Class SVM model...")
        self.svm_model = OneClassSVM(gamma='auto').fit(self.X_train)
        self.svm_predictions_train = self.svm_model.predict(self.X_train)
        self.svm_predictions_test = self.svm_model.predict(self.X_test)
        print("One-Class SVM model trained.")

    def train_one_class_svm3(self):
        print("Training One-Class SVM model...")
        self.svm_model = OneClassSVM(gamma='auto').fit(self.X_train)
        self.svm_predictions_train = self.svm_model.predict(self.X_train)
        self.svm_predictions_test = self.svm_model.predict(self.X_test)
        print("One-Class SVM model trained.")

    def train_meta_model(self):
        print("Training meta-model...")
        self.meta_model = LogisticRegression(random_state=42)
        self.meta_model.fit(self.X_train_combined, self.y_train)
        print("Meta-model trained.")

    def evaluate_meta_model(self):
        print("Evaluating meta-model...")
        y_pred = self.meta_model.predict(self.X_test_combined)
        print("Classification Report (Meta-Model):")
        print(classification_report(self.y_test, y_pred, zero_division=0))
        print("Confusion Matrix (Meta-Model):")
        print(confusion_matrix(self.y_test, y_pred))

    def evaluate_predictions(self, prediction_column):
        print(f"Evaluating predictions in column '{prediction_column}'...")
        y_pred = self.data[prediction_column]
        y_true = self.data['label']
        print(f"Classification Report ({prediction_column}):")
        print(classification_report(y_true, y_pred, zero_division=0))
        print(f"Confusion Matrix ({prediction_column}):")
        print(confusion_matrix(y_true, y_pred))

    def run_ensemble_model(self):
        self.trainisoforest()
        self.trainkmeans()
        self.combine_predictions()
        self.train_meta_model()
        self.evaluate_meta_model()


    def evaluate_anomaly_detection_model(self):
        print("Evaluating anomaly detection model...")
        
        
        data_for_evaluation = self.X_test
        
        
        predictions = self.model.predict(data_for_evaluation)
        self.data.loc[self.X_test.index, 'anomaly'] = predictions
        self.data['anomaly'] = self.data['anomaly'].map({1: 0, -1: 1})  
        print("Anomaly detection model evaluation completed.")
        print("Classification Report (Isolation Forest):")
        print(classification_report(self.y_test, self.data.loc[self.X_test.index, 'anomaly']))
        print("Confusion Matrix (Isolation Forest):")
        print(confusion_matrix(self.y_test, self.data.loc[self.X_test.index, 'anomaly']))
    
    def evaluate_svm_model(self):
        print("Evaluating One-Class SVM model...")
        if self.svm_model is None:
            raise ValueError("SVM model is not trained yet.")
        svm_predictions = self.svm_model.predict(self.X_test)
        
        
        print(f"Length of X_test: {len(self.X_test)}")
        print(f"Length of svm_predictions: {len(svm_predictions)}")
        
        if len(svm_predictions) != len(self.X_test):
            raise ValueError(f"Length of svm_predictions ({len(svm_predictions)}) does not match length of X_test ({len(self.X_test)})")
        
        
        self.data.loc[self.X_test.index, 'svm_anomaly'] = svm_predictions
        self.data['svm_anomaly'] = self.data['svm_anomaly'].map({1: 0, -1: 1})  
        print("One-Class SVM model evaluated and predictions added to testing data.")
        print("Classification Report")
        print(classification_report(self.y_test, self.data.loc[self.X_test.index, 'svm_anomaly']))
        print("Confusion Matrix (Test Set):")
        print(confusion_matrix(self.y_test, self.data.loc[self.X_test.index, 'svm_anomaly']))
        
    def build_autoencoder(self, input_dim):
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(32, activation='relu')(input_layer)
        encoded = Dense(16, activation='relu')(encoded)
        encoded = Dense(8, activation='relu')(encoded)
        decoded = Dense(16, activation='relu')(encoded)
        decoded = Dense(32, activation='relu')(decoded)
        output_layer = Dense(input_dim, activation='sigmoid')(decoded)
        autoencoder = Model(inputs=input_layer, outputs=output_layer)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder
    
    def combine_predictionsnew(self):
        print("Combining predictions from Isolation Forest, One-Class SVM, and KMeans...")
        self.X_train_combined = np.column_stack((self.svm_predictions_train, self.kmeans_predictions_train))
        self.X_test_combined = np.column_stack((self.svm_predictions_test, self.kmeans_predictions_test))
        print("Predictions combined.")

    def train_meta_modelnew(self):
        print("Training meta-model...")
        self.meta_model = LogisticRegression(random_state=42)
        self.meta_model.fit(self.X_train_combined, self.y_train)
        print("Meta-model trained.")

    def evaluate_meta_modelnew(self):
        print("Evaluating meta-model...")
        y_pred = self.meta_model.predict(self.X_test_combined)
        print("Classification Report (Meta-Model):")
        print(classification_report(self.y_test, y_pred, zero_division=0))
        print("Confusion Matrix (Meta-Model):")
        print(confusion_matrix(self.y_test, y_pred))

    def run_ensemble_modelnew(self):

        self.trainoneclasssvm()
        self.trainkmeans()
        self.combine_predictionsnew()
        self.train_meta_modelnew()
        self.evaluate_meta_modelnew()
    
    def train_autoencoder(self):
        print("Training autoencoder...")
        input_dim = self.data_preprocessed.shape[1]  
        self.autoencoder = self.build_autoencoder(input_dim)
        self.autoencoder.fit(self.data_preprocessed, self.data_preprocessed, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
        print("Autoencoder trained.")

    def evaluate_autoencoder(self):
        print("Evaluating autoencoder...")
        
        X_test_reconstructed = self.autoencoder.predict(self.X_test)
        
        
        reconstruction_error = np.mean(np.power(self.X_test - X_test_reconstructed, 2), axis=1)
        
        
        threshold = np.percentile(reconstruction_error, 95)
        print(f"Reconstruction error threshold: {threshold}")
        
        
        y_pred = (reconstruction_error > threshold).astype(int)
        
        
        print("Classification Report (Autoencoder):")
        print(classification_report(self.y_test, y_pred))
        print("Confusion Matrix (Autoencoder):")
        print(confusion_matrix(self.y_test, y_pred))
    

dataset = MyDataset(file_path='archive/attack-simulation-alert.csv')

print("\nCluster labels:")
dataset.perform_clustering(dataset.data_preprocessed, 'preprocessed')  
print(dataset.cluster_labels)

dataset.trainisoforest()
dataset.trainkmeans()
dataset.trainoneclasssvm()
dataset.save_plots()
dataset.plot_dendrogram()
dataset.evaluate_model()
dataset.train_anomaly_detection_model()
dataset.evaluate_anomaly_detection_model()
dataset.train_autoencoder()
dataset.calculate_reconstruction_errors()
dataset.trainoneclasssvm()
dataset.evaluate_svm_model()
dataset.run_ensemble_model()
dataset.run_ensemble_model2()
dataset.run_ensemble_modelnew()
best_model = dataset.cross_validate_model(dataset.X_train, dataset.y_train)
