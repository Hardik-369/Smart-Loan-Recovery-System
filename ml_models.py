import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class BorrowerSegmentation:
    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.pca = PCA(n_components=2, random_state=random_state)
        self.cluster_names = {}
        self.is_fitted = False
    
    def prepare_clustering_features(self, df):
        """Prepare features specifically for clustering"""
        # Select relevant features for segmentation
        clustering_features = [
            'Monthly_Income', 'Loan_Amount', 'Loan_Tenure', 'Collateral_Value',
            'Outstanding_Loan_Amount', 'Num_Missed_Payments', 'Days_Past_Due',
            'Debt_to_Income_Ratio', 'Payment_Punctuality_Score', 'Financial_Health_Score'
        ]
        
        # Filter features that exist in the dataframe
        available_features = [col for col in clustering_features if col in df.columns]
        
        return df[available_features].fillna(0)
    
    def find_optimal_clusters(self, X, max_clusters=10):
        """Find optimal number of clusters using elbow method"""
        inertias = []
        silhouette_scores = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            
            # Calculate silhouette score
            from sklearn.metrics import silhouette_score
            score = silhouette_score(X, kmeans.labels_)
            silhouette_scores.append(score)
        
        # Find elbow point
        optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
        
        return optimal_k, inertias, silhouette_scores
    
    def fit_kmeans(self, X):
        """Fit K-Means clustering"""
        self.kmeans.fit(X)
        labels = self.kmeans.labels_
        
        # Create cluster names based on characteristics
        self.cluster_names = self._create_cluster_names(X, labels)
        self.is_fitted = True
        
        return labels
    
    def _create_cluster_names(self, X, labels):
        """Create meaningful names for clusters based on their characteristics"""
        cluster_names = {}
        
        for cluster in np.unique(labels):
            cluster_data = X[labels == cluster]
            
            # Calculate cluster characteristics
            avg_income = cluster_data.iloc[:, 0].mean()  # Assuming first column is income
            avg_loan = cluster_data.iloc[:, 1].mean()    # Assuming second column is loan amount
            avg_missed = cluster_data.iloc[:, 5].mean()  # Assuming missed payments column
            avg_health = cluster_data.iloc[:, -1].mean() # Assuming last column is financial health
            
            # Create name based on characteristics
            if avg_health > 0.5:  # High financial health
                if avg_income > 0.5:
                    name = "High Income, Low Risk"
                else:
                    name = "Moderate Income, Low Risk"
            elif avg_health > 0:  # Moderate financial health
                if avg_missed > 0.3:
                    name = "Moderate Income, High Payment Issues"
                else:
                    name = "Moderate Income, Moderate Risk"
            else:  # Low financial health
                if avg_loan > 0.5:
                    name = "High Loan Burden, High Risk"
                else:
                    name = "Low Income, High Risk"
            
            cluster_names[cluster] = name
        
        return cluster_names
    
    def create_cluster_visualization(self, X, labels, df_original):
        """Create interactive cluster visualization"""
        # Apply PCA for 2D visualization
        X_pca = self.pca.fit_transform(X)
        
        # Create visualization dataframe
        risk_scores = df_original.get('Risk_Score', pd.Series(np.random.random(len(labels))))
        if hasattr(risk_scores, 'values'):
            risk_scores = risk_scores.values
        
        viz_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Cluster': [self.cluster_names.get(label, f"Cluster {label}") for label in labels],
            'Cluster_ID': labels,
            'Monthly_Income': df_original['Monthly_Income'].values,
            'Loan_Amount': df_original['Loan_Amount'].values,
            'Payment_History': df_original['Payment_History'].values,
            'Risk_Score': risk_scores
        })
        
        # Create interactive scatter plot
        fig = px.scatter(viz_df, x='PC1', y='PC2', color='Cluster', 
                        hover_data=['Monthly_Income', 'Loan_Amount', 'Payment_History', 'Risk_Score'],
                        title='Borrower Segmentation - K-Means Clustering',
                        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'})
        
        fig.update_layout(height=600, width=800)
        
        return fig, viz_df

class RiskPredictionModel:
    def __init__(self, model_type='random_forest', random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.feature_importance = {}
        self.model_metrics = {}
        self.is_trained = False
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=random_state
            )
    
    def train_model(self, X, y, test_size=0.2):
        """Train the risk prediction model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        self.model_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(X.columns, self.model.feature_importances_))
        
        self.is_trained = True
        
        return X_train, X_test, y_train, y_test, y_pred, y_pred_proba
    
    def predict_risk(self, X):
        """Predict risk for new data"""
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        return predictions, probabilities
    
    def create_feature_importance_plot(self):
        """Create feature importance visualization"""
        if not self.feature_importance:
            return None
        
        # Sort features by importance
        sorted_features = sorted(self.feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)[:15]  # Top 15 features
        
        features, importance = zip(*sorted_features)
        
        # Create interactive bar plot
        fig = go.Figure(data=[go.Bar(
            x=list(importance),
            y=list(features),
            orientation='h',
            marker_color='lightblue'
        )])
        
        fig.update_layout(
            title='Feature Importance in Risk Prediction',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=600,
            width=800
        )
        
        return fig
    
    def create_model_evaluation_plots(self, y_true, y_pred, y_pred_proba):
        """Create model evaluation visualizations"""
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Low Risk', 'High Risk'],
            y=['Low Risk', 'High Risk'],
            text=cm,
            texttemplate="%{text}",
            colorscale='Blues'
        ))
        
        fig_cm.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual'
        )
        
        # ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {auc_score:.3f})',
            line=dict(color='blue', width=2)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig_roc.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=500
        )
        
        return fig_cm, fig_roc

class EarlyWarningSystem:
    def __init__(self, risk_threshold=0.7, warning_days=90):
        self.risk_threshold = risk_threshold
        self.warning_days = warning_days
    
    def identify_at_risk_borrowers(self, df, risk_probabilities):
        """Identify borrowers who need early intervention"""
        df_risk = df.copy()
        df_risk['Risk_Probability'] = risk_probabilities
        
        # Calculate debt-to-income ratio if not present
        if 'Debt_to_Income_Ratio' not in df_risk.columns:
            df_risk['Debt_to_Income_Ratio'] = df_risk['Monthly_EMI'] / df_risk['Monthly_Income']
        
        # Criteria for early warning
        early_warning_conditions = (
            (df_risk['Risk_Probability'] > self.risk_threshold) |
            (df_risk['Days_Past_Due'] > 30) |
            (df_risk['Num_Missed_Payments'] > 2) |
            (df_risk['Debt_to_Income_Ratio'] > 0.5)
        )
        
        at_risk_borrowers = df_risk[early_warning_conditions].copy()
        
        # Priority scoring
        at_risk_borrowers['Priority_Score'] = (
            at_risk_borrowers['Risk_Probability'] * 0.4 +
            (at_risk_borrowers['Outstanding_Loan_Amount'] / at_risk_borrowers['Loan_Amount']) * 0.3 +
            (at_risk_borrowers['Days_Past_Due'] / 365) * 0.2 +
            (at_risk_borrowers['Num_Missed_Payments'] / at_risk_borrowers['Loan_Tenure']) * 0.1
        )
        
        # Sort by priority
        at_risk_borrowers = at_risk_borrowers.sort_values('Priority_Score', ascending=False)
        
        return at_risk_borrowers
    
    def create_early_warning_dashboard(self, at_risk_borrowers):
        """Create early warning visualization"""
        # Risk distribution
        fig_risk = px.histogram(at_risk_borrowers, x='Risk_Probability', 
                               title='Risk Probability Distribution - At-Risk Borrowers',
                               nbins=20)
        
        # Priority vs Outstanding Amount
        fig_priority = px.scatter(at_risk_borrowers, x='Priority_Score', y='Outstanding_Loan_Amount',
                                color='Days_Past_Due', size='Loan_Amount',
                                title='Priority Score vs Outstanding Loan Amount',
                                hover_data=['Borrower_ID', 'Risk_Probability'])
        
        return fig_risk, fig_priority

class AdvancedAnalytics:
    @staticmethod
    def calculate_expected_recovery(df, risk_probabilities):
        """Calculate expected recovery amount for each loan"""
        recovery_rates = {
            'Low Risk': 0.9,
            'Moderate Risk': 0.6,
            'High Risk': 0.3
        }
        
        expected_recovery = []
        for i, prob in enumerate(risk_probabilities):
            if prob < 0.3:
                rate = recovery_rates['Low Risk']
            elif prob < 0.7:
                rate = recovery_rates['Moderate Risk']
            else:
                rate = recovery_rates['High Risk']
            
            expected_amount = df.iloc[i]['Outstanding_Loan_Amount'] * rate
            expected_recovery.append(expected_amount)
        
        return expected_recovery
    
    @staticmethod
    def optimize_collection_strategy(df, risk_probabilities):
        """Recommend optimal collection strategy based on risk and loan characteristics"""
        strategies = []
        
        for i, (_, row) in enumerate(df.iterrows()):
            risk_prob = risk_probabilities[i]
            outstanding = row['Outstanding_Loan_Amount']
            days_overdue = row['Days_Past_Due']
            
            if risk_prob > 0.8 and outstanding > 100000:
                strategy = "Legal Action + Asset Recovery"
            elif risk_prob > 0.6 and days_overdue > 60:
                strategy = "Debt Collection Agency"
            elif risk_prob > 0.4:
                strategy = "Settlement Offer"
            elif days_overdue > 30:
                strategy = "Payment Plan Negotiation"
            else:
                strategy = "Automated Reminders"
            
            strategies.append(strategy)
        
        return strategies

# Testing and utility functions
if __name__ == "__main__":
    # Import required modules
    from data_generator import LoanDataGenerator
    from data_preprocessor import DataPreprocessor
    
    # Generate and preprocess data
    print("Generating test data...")
    generator = LoanDataGenerator(n_borrowers=2000)
    raw_data = generator.generate_complete_dataset()
    
    print("Preprocessing data...")
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.process_complete_pipeline(raw_data)
    
    # Prepare features for ML
    X, y, feature_cols = preprocessor.prepare_features_for_ml(processed_data)
    
    print(f"Training data shape: X={X.shape}, y={y.shape}")
    
    # Test segmentation
    print("\nTesting borrower segmentation...")
    segmentation = BorrowerSegmentation(n_clusters=5)
    clustering_features = segmentation.prepare_clustering_features(processed_data)
    
    # Find optimal clusters
    optimal_k, inertias, silhouette_scores = segmentation.find_optimal_clusters(clustering_features)
    print(f"Optimal number of clusters: {optimal_k}")
    
    # Fit clustering
    cluster_labels = segmentation.fit_kmeans(clustering_features)
    print("Cluster distribution:", np.bincount(cluster_labels))
    print("Cluster names:", segmentation.cluster_names)
    
    # Test risk prediction
    print("\nTesting risk prediction model...")
    risk_model = RiskPredictionModel(model_type='random_forest')
    X_train, X_test, y_train, y_test, y_pred, y_pred_proba = risk_model.train_model(X, y)
    
    print("Model Metrics:")
    for metric, value in risk_model.model_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Test early warning system
    print("\nTesting early warning system...")
    early_warning = EarlyWarningSystem()
    at_risk_borrowers = early_warning.identify_at_risk_borrowers(processed_data, risk_model.predict_risk(X)[1])
    print(f"Number of at-risk borrowers: {len(at_risk_borrowers)}")
    
    print("\nML Models testing completed!")