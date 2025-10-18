import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class LoanDataGenerator:
    def __init__(self, n_borrowers=10000):
        self.n_borrowers = n_borrowers
        np.random.seed(42)
        random.seed(42)
        
    def generate_borrower_profiles(self):
        """Generate borrower profiles with realistic distributions"""
        borrowers = []
        
        for i in range(self.n_borrowers):
            # Age distribution (normal distribution centered at 35)
            age = int(np.clip(np.random.normal(35, 12), 21, 70))
            
            # Gender
            gender = np.random.choice(['Male', 'Female'], p=[0.6, 0.4])
            
            # Employment type (more salaried employees)
            employment_type = np.random.choice(['Salaried', 'Self-Employed'], p=[0.7, 0.3])
            
            # Monthly income (log-normal distribution)
            if employment_type == 'Salaried':
                income = np.random.lognormal(mean=10.5, sigma=0.6) * 100  # Higher for salaried
            else:
                income = np.random.lognormal(mean=10.2, sigma=0.8) * 100  # More variance for self-employed
            
            income = max(15000, min(500000, income))  # Cap between 15k and 500k
            
            # Number of dependents
            num_dependents = np.random.poisson(1.5)  # Poisson distribution
            num_dependents = min(5, num_dependents)
            
            borrowers.append({
                'Borrower_ID': f'B{i+1:06d}',
                'Age': age,
                'Gender': gender,
                'Employment_Type': employment_type,
                'Monthly_Income': round(income, 2),
                'Num_Dependents': num_dependents
            })
            
        return pd.DataFrame(borrowers)
    
    def generate_loan_details(self, borrowers_df):
        """Generate loan details based on borrower profiles"""
        loans = []
        
        for _, borrower in borrowers_df.iterrows():
            # Loan amount based on income (3-10x monthly income)
            income_multiplier = np.random.uniform(3, 10)
            loan_amount = borrower['Monthly_Income'] * income_multiplier
            
            # Adjust loan amount based on employment type
            if borrower['Employment_Type'] == 'Self-Employed':
                loan_amount *= np.random.uniform(0.7, 1.0)  # Lower for self-employed
            
            loan_amount = max(50000, min(5000000, loan_amount))  # Cap between 50k and 5M
            
            # Loan tenure (12-84 months, more common around 24-60)
            tenure_ranges = [(12, 24), (25, 36), (37, 60), (61, 84)]
            tenure_weights = [0.1, 0.3, 0.4, 0.2]  # 12-24, 25-36, 37-60, 61-84
            tenure_idx = np.random.choice(len(tenure_ranges), p=tenure_weights)
            tenure_range = tenure_ranges[tenure_idx]
            loan_tenure = np.random.randint(tenure_range[0], tenure_range[1] + 1)
            
            # Interest rate (5-20%, based on risk profile)
            base_rate = np.random.uniform(8, 15)
            if borrower['Employment_Type'] == 'Self-Employed':
                base_rate += np.random.uniform(1, 3)  # Higher for self-employed
            if borrower['Monthly_Income'] < 30000:
                base_rate += np.random.uniform(2, 4)  # Higher for low income
            
            interest_rate = min(20, max(5, base_rate))
            
            # Collateral value (0-3x loan amount)
            collateral_prob = np.random.random()
            if collateral_prob < 0.3:  # 30% have no collateral
                collateral_value = 0
            else:
                collateral_multiplier = np.random.uniform(0.5, 3.0)
                collateral_value = loan_amount * collateral_multiplier
            
            loans.append({
                'Borrower_ID': borrower['Borrower_ID'],
                'Loan_ID': f'L{len(loans)+1:06d}',
                'Loan_Amount': round(loan_amount, 2),
                'Loan_Tenure': loan_tenure,
                'Interest_Rate': round(interest_rate, 2),
                'Collateral_Value': round(collateral_value, 2)
            })
            
        return pd.DataFrame(loans)
    
    def generate_repayment_history(self, borrowers_df, loans_df):
        """Generate repayment history and current status"""
        repayment_data = []
        
        for _, loan in loans_df.iterrows():
            borrower = borrowers_df[borrowers_df['Borrower_ID'] == loan['Borrower_ID']].iloc[0]
            
            # Calculate EMI
            loan_amount = loan['Loan_Amount']
            rate = loan['Interest_Rate'] / 100 / 12  # Monthly rate
            tenure = loan['Loan_Tenure']
            
            if rate == 0:
                monthly_emi = loan_amount / tenure
            else:
                monthly_emi = (loan_amount * rate * (1 + rate)**tenure) / ((1 + rate)**tenure - 1)
            
            # Determine payment behavior based on borrower profile
            payment_risk_score = 0
            
            # Risk factors
            if borrower['Employment_Type'] == 'Self-Employed':
                payment_risk_score += 0.2
            if borrower['Monthly_Income'] < 30000:
                payment_risk_score += 0.3
            if monthly_emi / borrower['Monthly_Income'] > 0.4:  # High EMI to income ratio
                payment_risk_score += 0.4
            if loan['Collateral_Value'] == 0:
                payment_risk_score += 0.2
            if borrower['Num_Dependents'] > 2:
                payment_risk_score += 0.1
            
            # Generate payment history
            payment_behavior = np.random.random()
            if payment_behavior < payment_risk_score:
                payment_history = np.random.choice(['Delayed', 'Missed'], p=[0.6, 0.4])
                num_missed_payments = np.random.poisson(min(tenure * 0.3, 12))
                days_past_due = np.random.poisson(45) if payment_history == 'Delayed' else np.random.poisson(90)
            else:
                payment_history = 'On-Time'
                num_missed_payments = np.random.poisson(0.5)  # Very few missed payments
                days_past_due = 0
            
            num_missed_payments = max(0, min(tenure, num_missed_payments))
            
            # Calculate outstanding loan amount
            months_elapsed = min(tenure, np.random.randint(6, tenure + 1))
            payments_made = max(0, months_elapsed - num_missed_payments)
            
            if rate == 0:
                outstanding_amount = loan_amount - (payments_made * monthly_emi)
            else:
                # Calculate remaining principal using amortization formula
                if payments_made == 0:
                    outstanding_amount = loan_amount
                else:
                    remaining_payments = tenure - payments_made
                    if remaining_payments <= 0:
                        outstanding_amount = 0
                    else:
                        outstanding_amount = monthly_emi * ((1 + rate)**remaining_payments - 1) / (rate * (1 + rate)**remaining_payments)
            
            outstanding_amount = max(0, outstanding_amount)
            
            repayment_data.append({
                'Loan_ID': loan['Loan_ID'],
                'Outstanding_Loan_Amount': round(outstanding_amount, 2),
                'Monthly_EMI': round(monthly_emi, 2),
                'Payment_History': payment_history,
                'Num_Missed_Payments': num_missed_payments,
                'Days_Past_Due': max(0, days_past_due)
            })
            
        return pd.DataFrame(repayment_data)
    
    def generate_collection_efforts(self, loans_df, repayment_df):
        """Generate collection efforts based on payment history"""
        collection_data = []
        
        for _, loan in loans_df.iterrows():
            repayment = repayment_df[repayment_df['Loan_ID'] == loan['Loan_ID']].iloc[0]
            
            # Collection efforts based on payment history
            if repayment['Payment_History'] == 'On-Time':
                collection_attempts = np.random.poisson(0.5)
                collection_method = 'Automated Reminders'
                legal_action = 'No'
            elif repayment['Payment_History'] == 'Delayed':
                collection_attempts = np.random.poisson(3)
                collection_method = np.random.choice(['Calls', 'Settlement Offer'], p=[0.7, 0.3])
                legal_action = 'No' if np.random.random() > 0.1 else 'Yes'
            else:  # Missed
                collection_attempts = np.random.poisson(8)
                collection_method = np.random.choice(['Legal Notice', 'Debt Collectors', 'Settlement Offer'], p=[0.4, 0.4, 0.2])
                legal_action = 'Yes' if np.random.random() > 0.3 else 'No'
            
            collection_attempts = max(0, collection_attempts)
            
            collection_data.append({
                'Loan_ID': loan['Loan_ID'],
                'Collection_Method': collection_method,
                'Collection_Attempts': collection_attempts,
                'Legal_Action_Taken': legal_action
            })
            
        return pd.DataFrame(collection_data)
    
    def generate_recovery_status(self, merged_df):
        """Generate loan recovery status based on all factors"""
        recovery_statuses = []
        
        for _, row in merged_df.iterrows():
            # Calculate recovery probability based on multiple factors
            recovery_prob = 0.5  # Base probability
            
            # Payment history impact
            if row['Payment_History'] == 'On-Time':
                recovery_prob += 0.3
            elif row['Payment_History'] == 'Delayed':
                recovery_prob += 0.1
            else:  # Missed
                recovery_prob -= 0.2
            
            # Income impact
            if row['Monthly_Income'] > 75000:
                recovery_prob += 0.2
            elif row['Monthly_Income'] < 25000:
                recovery_prob -= 0.2
            
            # Collateral impact
            if row['Collateral_Value'] > row['Outstanding_Loan_Amount']:
                recovery_prob += 0.3
            elif row['Collateral_Value'] == 0:
                recovery_prob -= 0.1
            
            # Collection efforts impact
            if row['Legal_Action_Taken'] == 'Yes':
                recovery_prob += 0.1
            
            # EMI to income ratio impact
            emi_ratio = row['Monthly_EMI'] / row['Monthly_Income']
            if emi_ratio > 0.5:
                recovery_prob -= 0.3
            elif emi_ratio < 0.2:
                recovery_prob += 0.1
            
            # Final recovery status
            recovery_prob = max(0, min(1, recovery_prob))
            
            if np.random.random() < recovery_prob:
                if row['Outstanding_Loan_Amount'] == 0:
                    recovery_status = 'Fully Recovered'
                else:
                    recovery_status = np.random.choice(['Partially Recovered', 'In Progress'], p=[0.4, 0.6])
            else:
                recovery_status = np.random.choice(['Write-off', 'Under Legal Action'], p=[0.6, 0.4])
            
            recovery_statuses.append(recovery_status)
            
        return recovery_statuses
    
    def generate_complete_dataset(self):
        """Generate complete synthetic dataset"""
        print("Generating borrower profiles...")
        borrowers_df = self.generate_borrower_profiles()
        
        print("Generating loan details...")
        loans_df = self.generate_loan_details(borrowers_df)
        
        print("Generating repayment history...")
        repayment_df = self.generate_repayment_history(borrowers_df, loans_df)
        
        print("Generating collection efforts...")
        collection_df = self.generate_collection_efforts(loans_df, repayment_df)
        
        print("Merging all data...")
        # Merge all dataframes
        merged_df = borrowers_df.merge(loans_df, on='Borrower_ID')
        merged_df = merged_df.merge(repayment_df, on='Loan_ID')
        merged_df = merged_df.merge(collection_df, on='Loan_ID')
        
        print("Generating recovery status...")
        merged_df['Recovery_Status'] = self.generate_recovery_status(merged_df)
        
        return merged_df

if __name__ == "__main__":
    generator = LoanDataGenerator(n_borrowers=5000)
    dataset = generator.generate_complete_dataset()
    
    print("\nDataset shape:", dataset.shape)
    print("\nDataset info:")
    print(dataset.info())
    print("\nFirst few rows:")
    print(dataset.head())
    
    # Save dataset
    dataset.to_csv('loan_recovery_dataset.csv', index=False)
    print("\nDataset saved as 'loan_recovery_dataset.csv'")