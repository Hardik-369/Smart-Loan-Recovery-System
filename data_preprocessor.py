import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.is_fitted = False
    
    def create_derived_features(self, df):
        """Create derived features for better model performance"""
        df_processed = df.copy()
        
        # Debt-to-Income ratio
        df_processed['Debt_to_Income_Ratio'] = df_processed['Monthly_EMI'] / df_processed['Monthly_Income']
        
        # Payment Punctuality Score (0-100)
        max_payments = df_processed.groupby('Borrower_ID')['Loan_Tenure'].first().reset_index()
        df_processed = df_processed.merge(max_payments.rename(columns={'Loan_Tenure': 'Max_Loan_Tenure'}), on='Borrower_ID')
        
        # Calculate punctuality based on missed payments relative to loan tenure
        df_processed['Payment_Punctuality_Score'] = np.clip(
            100 - (df_processed['Num_Missed_Payments'] / df_processed['Loan_Tenure'] * 100), 0, 100
        )
        
        # Total Recovery Amount (for loans that are recovered)
        def calculate_recovery_amount(row):
            status = row['Recovery_Status']
            loan_amount = row['Loan_Amount']
            outstanding = row['Outstanding_Loan_Amount']
            
            if status == 'Fully Recovered':
                return loan_amount
            elif status == 'Partially Recovered':
                return loan_amount * np.random.uniform(0.3, 0.8)
            elif status == 'In Progress':
                return max(0, loan_amount - outstanding)
            elif status == 'Write-off':
                return 0
            elif status == 'Under Legal Action':
                return loan_amount * np.random.uniform(0.1, 0.4)
            else:
                return 0
        
        df_processed['Total_Recovery_Amount'] = df_processed.apply(calculate_recovery_amount, axis=1)
        
        # Loan-to-Collateral ratio
        df_processed['Loan_to_Collateral_Ratio'] = np.where(
            df_processed['Collateral_Value'] > 0,
            df_processed['Loan_Amount'] / df_processed['Collateral_Value'],
            np.inf
        )
        df_processed['Loan_to_Collateral_Ratio'] = np.clip(df_processed['Loan_to_Collateral_Ratio'], 0, 10)
        
        # Age group categorization
        df_processed['Age_Group'] = pd.cut(df_processed['Age'], 
                                         bins=[20, 30, 40, 50, 70], 
                                         labels=['Young', 'Adult', 'Middle-aged', 'Senior'])
        
        # Income bracket
        df_processed['Income_Bracket'] = pd.cut(df_processed['Monthly_Income'], 
                                              bins=[0, 25000, 50000, 100000, np.inf],
                                              labels=['Low', 'Medium', 'High', 'Very High'])
        
        # EMI burden category
        df_processed['EMI_Burden'] = pd.cut(df_processed['Debt_to_Income_Ratio'],
                                          bins=[0, 0.2, 0.4, 0.6, np.inf],
                                          labels=['Low', 'Moderate', 'High', 'Very High'])
        
        # Days overdue category
        df_processed['Overdue_Category'] = pd.cut(df_processed['Days_Past_Due'],
                                                bins=[-1, 0, 30, 90, np.inf],
                                                labels=['On-Time', 'Recent', 'Moderate', 'Severe'])
        
        # Financial health score (composite metric)
        # Normalize components to 0-100 scale
        income_score = np.clip((df_processed['Monthly_Income'] - 15000) / (500000 - 15000) * 100, 0, 100)
        punctuality_score = df_processed['Payment_Punctuality_Score']
        burden_score = np.clip((1 - df_processed['Debt_to_Income_Ratio']) * 100, 0, 100)
        collateral_score = np.clip((df_processed['Collateral_Value'] / df_processed['Loan_Amount']) * 50, 0, 100)
        
        df_processed['Financial_Health_Score'] = (
            income_score * 0.3 + 
            punctuality_score * 0.4 + 
            burden_score * 0.2 + 
            collateral_score * 0.1
        )
        
        # Risk indicators
        df_processed['High_Risk_Indicators'] = (
            (df_processed['Employment_Type'] == 'Self-Employed').astype(int) +
            (df_processed['Debt_to_Income_Ratio'] > 0.4).astype(int) +
            (df_processed['Collateral_Value'] == 0).astype(int) +
            (df_processed['Days_Past_Due'] > 60).astype(int) +
            (df_processed['Num_Missed_Payments'] > 3).astype(int)
        )
        
        return df_processed
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        df_cleaned = df.copy()
        
        # Numerical columns
        numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_cleaned[col].isnull().sum() > 0:
                # Use median for financial data
                df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
        
        # Categorical columns
        categorical_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df_cleaned[col].isnull().sum() > 0:
                # Use mode for categorical data
                df_cleaned[col].fillna(df_cleaned[col].mode().iloc[0], inplace=True)
        
        return df_cleaned
    
    def encode_categorical_features(self, df):
        """Encode categorical features for ML models"""
        df_encoded = df.copy()
        
        # Binary categorical features (use label encoding)
        binary_categorical = ['Gender', 'Employment_Type', 'Legal_Action_Taken']
        
        for col in binary_categorical:
            if col in df_encoded.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
        
        # Multi-category features (use one-hot encoding)
        multi_categorical = ['Payment_History', 'Collection_Method', 'Recovery_Status', 
                           'Age_Group', 'Income_Bracket', 'EMI_Burden', 'Overdue_Category']
        
        for col in multi_categorical:
            if col in df_encoded.columns:
                # Create dummy variables
                dummies = pd.get_dummies(df_encoded[col], prefix=col)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded.drop(col, axis=1, inplace=True)
        
        return df_encoded
    
    def normalize_features(self, df, fit=True):
        """Normalize numerical features"""
        df_normalized = df.copy()
        
        # Select numerical columns for normalization
        numerical_cols = [
            'Age', 'Monthly_Income', 'Num_Dependents', 'Loan_Amount', 'Loan_Tenure',
            'Interest_Rate', 'Collateral_Value', 'Outstanding_Loan_Amount', 'Monthly_EMI',
            'Num_Missed_Payments', 'Days_Past_Due', 'Collection_Attempts',
            'Debt_to_Income_Ratio', 'Payment_Punctuality_Score', 'Total_Recovery_Amount',
            'Loan_to_Collateral_Ratio', 'Financial_Health_Score', 'High_Risk_Indicators'
        ]
        
        # Only normalize columns that exist in the dataframe
        cols_to_normalize = [col for col in numerical_cols if col in df_normalized.columns]
        
        if fit:
            df_normalized[cols_to_normalize] = self.scaler.fit_transform(df_normalized[cols_to_normalize])
            self.feature_names = df_normalized.columns.tolist()
        else:
            if self.is_fitted:
                df_normalized[cols_to_normalize] = self.scaler.transform(df_normalized[cols_to_normalize])
            else:
                raise ValueError("Scaler not fitted yet. Call with fit=True first.")
        
        return df_normalized
    
    def create_risk_labels(self, df):
        """Create risk labels for classification"""
        df_with_labels = df.copy()
        
        # Create risk score based on multiple factors
        risk_score = 0
        
        # Payment history risk
        payment_risk = pd.Series(0, index=df_with_labels.index)
        payment_risk[df_with_labels['Payment_History'] == 'On-Time'] = 0
        payment_risk[df_with_labels['Payment_History'] == 'Delayed'] = 0.3
        payment_risk[df_with_labels['Payment_History'] == 'Missed'] = 0.6
        
        # EMI burden risk
        emi_risk = np.clip(df_with_labels['Debt_to_Income_Ratio'] / 0.6, 0, 1) * 0.4
        
        # Missed payments risk
        max_missed = df_with_labels['Num_Missed_Payments'].max()
        missed_risk = (df_with_labels['Num_Missed_Payments'] / max_missed) * 0.3 if max_missed > 0 else 0
        
        # Days past due risk
        max_days = df_with_labels['Days_Past_Due'].max()
        overdue_risk = (df_with_labels['Days_Past_Due'] / max_days) * 0.2 if max_days > 0 else 0
        
        # Employment type risk
        employment_risk = (df_with_labels['Employment_Type'] == 'Self-Employed').astype(float) * 0.1
        
        # Collateral risk
        collateral_risk = (df_with_labels['Collateral_Value'] == 0).astype(float) * 0.1
        
        # Combine all risk factors
        total_risk = payment_risk + emi_risk + missed_risk + overdue_risk + employment_risk + collateral_risk
        
        # Create binary risk labels
        df_with_labels['Risk_Score'] = total_risk
        df_with_labels['Risk_Label'] = (total_risk > 0.5).astype(int)  # 1 for High Risk, 0 for Low Risk
        df_with_labels['Risk_Category'] = pd.cut(total_risk, 
                                               bins=[0, 0.3, 0.7, 1.0],
                                               labels=['Low Risk', 'Moderate Risk', 'High Risk'])
        
        return df_with_labels
    
    def prepare_features_for_ml(self, df, target_column='Risk_Label'):
        """Prepare features for machine learning models"""
        df_ml = df.copy()
        
        # Remove non-feature columns
        non_feature_cols = [
            'Borrower_ID', 'Loan_ID', target_column, 'Risk_Score', 'Risk_Category',
            'Max_Loan_Tenure'  # This was added temporarily for calculations
        ]
        
        feature_cols = [col for col in df_ml.columns if col not in non_feature_cols]
        
        X = df_ml[feature_cols]
        y = df_ml[target_column] if target_column in df_ml.columns else None
        
        return X, y, feature_cols
    
    def process_complete_pipeline(self, df, fit=True):
        """Complete preprocessing pipeline"""
        print("Starting data preprocessing pipeline...")
        
        # Step 1: Handle missing values
        print("1. Handling missing values...")
        df_processed = self.handle_missing_values(df)
        
        # Step 2: Create derived features
        print("2. Creating derived features...")
        df_processed = self.create_derived_features(df_processed)
        
        # Step 3: Create risk labels
        print("3. Creating risk labels...")
        df_processed = self.create_risk_labels(df_processed)
        
        # Step 4: Encode categorical features
        print("4. Encoding categorical features...")
        df_processed = self.encode_categorical_features(df_processed)
        
        # Step 5: Normalize features
        print("5. Normalizing numerical features...")
        df_processed = self.normalize_features(df_processed, fit=fit)
        
        if fit:
            self.is_fitted = True
        
        print("Data preprocessing completed!")
        return df_processed
    
    def get_feature_importance_data(self, df):
        """Get data for feature importance analysis"""
        feature_descriptions = {
            'Age': 'Borrower Age',
            'Monthly_Income': 'Monthly Income',
            'Loan_Amount': 'Original Loan Amount',
            'Interest_Rate': 'Loan Interest Rate',
            'Outstanding_Loan_Amount': 'Current Outstanding Amount',
            'Num_Missed_Payments': 'Number of Missed Payments',
            'Days_Past_Due': 'Days Past Due',
            'Debt_to_Income_Ratio': 'EMI to Income Ratio',
            'Payment_Punctuality_Score': 'Payment Punctuality (0-100)',
            'Financial_Health_Score': 'Overall Financial Health',
            'High_Risk_Indicators': 'Number of Risk Factors',
            'Loan_to_Collateral_Ratio': 'Loan to Collateral Ratio'
        }
        
        return feature_descriptions

if __name__ == "__main__":
    # Test the preprocessor
    from data_generator import LoanDataGenerator
    
    # Generate test data
    generator = LoanDataGenerator(n_borrowers=1000)
    df = generator.generate_complete_dataset()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Process data
    processed_df = preprocessor.process_complete_pipeline(df)
    
    print("\nProcessed dataset shape:", processed_df.shape)
    print("\nFeature columns:", len([col for col in processed_df.columns if col not in ['Borrower_ID', 'Loan_ID', 'Risk_Label', 'Risk_Score', 'Risk_Category']]))
    print("\nRisk label distribution:")
    print(processed_df['Risk_Label'].value_counts())
    
    # Prepare for ML
    X, y, feature_cols = preprocessor.prepare_features_for_ml(processed_df)
    print(f"\nML features shape: {X.shape}")
    print(f"Target shape: {y.shape}")