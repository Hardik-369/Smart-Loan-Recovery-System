import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class RecoveryStrategyEngine:
    def __init__(self):
        self.strategy_definitions = {
            'Automated Reminders': {
                'cost': 5,
                'success_rate': 0.15,
                'time_to_recovery': 30,
                'description': 'Automated SMS/Email reminders',
                'suitable_for': 'Low risk borrowers with minor delays'
            },
            'Phone Call Campaign': {
                'cost': 25,
                'success_rate': 0.35,
                'time_to_recovery': 15,
                'description': 'Personal phone calls from collection team',
                'suitable_for': 'Moderate risk borrowers, early intervention'
            },
            'Payment Plan Negotiation': {
                'cost': 100,
                'success_rate': 0.55,
                'time_to_recovery': 45,
                'description': 'Restructured payment plans',
                'suitable_for': 'Borrowers with temporary financial difficulties'
            },
            'Settlement Offer': {
                'cost': 200,
                'success_rate': 0.45,
                'time_to_recovery': 60,
                'description': 'Discounted settlement offers',
                'suitable_for': 'High risk borrowers, partial recovery preferred'
            },
            'Debt Collection Agency': {
                'cost': 500,
                'success_rate': 0.40,
                'time_to_recovery': 90,
                'description': 'Third-party collection agencies',
                'suitable_for': 'Persistent defaulters, moderate to high amounts'
            },
            'Legal Action': {
                'cost': 2000,
                'success_rate': 0.65,
                'time_to_recovery': 180,
                'description': 'Legal proceedings and asset seizure',
                'suitable_for': 'High-value loans with collateral'
            },
            'Write-off': {
                'cost': 50,
                'success_rate': 0.05,
                'time_to_recovery': 365,
                'description': 'Debt write-off with tax benefits',
                'suitable_for': 'Uncollectable debts'
            }
        }
        
        self.collection_history = []
        self.strategy_effectiveness = {}
    
    def assign_recovery_strategy(self, borrower_data, risk_probability):
        """Assign optimal recovery strategy based on borrower profile and risk"""
        
        # Extract key characteristics
        outstanding_amount = borrower_data.get('Outstanding_Loan_Amount', 0)
        days_past_due = borrower_data.get('Days_Past_Due', 0)
        collateral_value = borrower_data.get('Collateral_Value', 0)
        monthly_income = borrower_data.get('Monthly_Income', 0)
        num_missed_payments = borrower_data.get('Num_Missed_Payments', 0)
        legal_action_taken = borrower_data.get('Legal_Action_Taken', 'No')
        
        # Decision tree logic for strategy assignment
        if risk_probability <= 0.3:
            if days_past_due <= 30:
                return 'Automated Reminders'
            elif days_past_due <= 60:
                return 'Phone Call Campaign'
            else:
                return 'Payment Plan Negotiation'
        
        elif risk_probability <= 0.6:
            if outstanding_amount < 50000:
                if days_past_due <= 60:
                    return 'Phone Call Campaign'
                else:
                    return 'Settlement Offer'
            else:
                if collateral_value > outstanding_amount * 0.5:
                    return 'Payment Plan Negotiation'
                else:
                    return 'Debt Collection Agency'
        
        else:  # High risk (> 0.6)
            if outstanding_amount >= 200000:
                if collateral_value > outstanding_amount * 0.3:
                    return 'Legal Action'
                else:
                    return 'Debt Collection Agency'
            elif outstanding_amount >= 50000:
                if legal_action_taken == 'Yes':
                    return 'Legal Action'
                else:
                    return 'Debt Collection Agency'
            else:
                if days_past_due >= 180:
                    return 'Write-off'
                else:
                    return 'Settlement Offer'
    
    def calculate_strategy_roi(self, strategy, outstanding_amount, risk_probability):
        """Calculate expected ROI for a given strategy"""
        strategy_info = self.strategy_definitions[strategy]
        
        # Calculate expected recovery amount
        success_rate = strategy_info['success_rate']
        
        # Adjust success rate based on risk probability
        adjusted_success_rate = success_rate * (1 - risk_probability * 0.5)
        
        expected_recovery = outstanding_amount * adjusted_success_rate
        cost = strategy_info['cost']
        
        # Calculate ROI
        roi = (expected_recovery - cost) / cost if cost > 0 else expected_recovery
        
        return {
            'expected_recovery': expected_recovery,
            'cost': cost,
            'roi': roi,
            'adjusted_success_rate': adjusted_success_rate,
            'time_to_recovery': strategy_info['time_to_recovery']
        }
    
    def recommend_optimal_strategy(self, borrower_data, risk_probability):
        """Recommend the optimal strategy based on ROI and other factors"""
        outstanding_amount = borrower_data.get('Outstanding_Loan_Amount', 0)
        
        # Calculate ROI for all applicable strategies
        strategy_analysis = {}
        
        for strategy in self.strategy_definitions.keys():
            roi_data = self.calculate_strategy_roi(strategy, outstanding_amount, risk_probability)
            
            # Add strategy suitability score
            suitability_score = self._calculate_suitability_score(
                strategy, borrower_data, risk_probability
            )
            
            strategy_analysis[strategy] = {
                **roi_data,
                'suitability_score': suitability_score,
                'total_score': roi_data['roi'] * 0.6 + suitability_score * 0.4
            }
        
        # Rank strategies by total score
        ranked_strategies = sorted(
            strategy_analysis.items(),
            key=lambda x: x[1]['total_score'],
            reverse=True
        )
        
        # Return top 3 recommendations
        return ranked_strategies[:3]
    
    def _calculate_suitability_score(self, strategy, borrower_data, risk_probability):
        """Calculate how suitable a strategy is for the borrower profile"""
        score = 0.5  # Base score
        
        outstanding_amount = borrower_data.get('Outstanding_Loan_Amount', 0)
        days_past_due = borrower_data.get('Days_Past_Due', 0)
        collateral_value = borrower_data.get('Collateral_Value', 0)
        
        if strategy == 'Automated Reminders':
            if risk_probability < 0.3 and days_past_due < 30:
                score += 0.4
        
        elif strategy == 'Phone Call Campaign':
            if 0.2 < risk_probability < 0.5 and days_past_due < 60:
                score += 0.3
        
        elif strategy == 'Payment Plan Negotiation':
            if outstanding_amount < 100000 and risk_probability < 0.6:
                score += 0.3
        
        elif strategy == 'Settlement Offer':
            if 0.4 < risk_probability < 0.8:
                score += 0.2
        
        elif strategy == 'Debt Collection Agency':
            if risk_probability > 0.5 and outstanding_amount > 30000:
                score += 0.3
        
        elif strategy == 'Legal Action':
            if collateral_value > outstanding_amount * 0.2 and outstanding_amount > 100000:
                score += 0.4
        
        elif strategy == 'Write-off':
            if risk_probability > 0.8 or days_past_due > 365:
                score += 0.3
        
        return min(1.0, score)
    
    def create_recovery_priority_list(self, df, risk_probabilities, strategies):
        """Create prioritized list of borrowers for recovery efforts"""
        
        priority_data = []
        
        for i, (_, borrower) in enumerate(df.iterrows()):
            outstanding = borrower['Outstanding_Loan_Amount']
            risk_prob = risk_probabilities[i]
            strategy = strategies[i]
            
            # Calculate priority score
            amount_weight = min(outstanding / 500000, 1) * 0.4  # Normalize by max amount
            risk_weight = risk_prob * 0.3
            urgency_weight = min(borrower['Days_Past_Due'] / 365, 1) * 0.3
            
            priority_score = amount_weight + risk_weight + urgency_weight
            
            # Calculate expected recovery value
            strategy_info = self.strategy_definitions[strategy]
            expected_recovery = outstanding * strategy_info['success_rate'] * (1 - risk_prob * 0.3)
            
            priority_data.append({
                'Borrower_ID': borrower['Borrower_ID'],
                'Priority_Score': priority_score,
                'Risk_Probability': risk_prob,
                'Outstanding_Amount': outstanding,
                'Expected_Recovery': expected_recovery,
                'Strategy': strategy,
                'Days_Past_Due': borrower['Days_Past_Due'],
                'Monthly_Income': borrower['Monthly_Income'],
                'ROI': (expected_recovery - strategy_info['cost']) / strategy_info['cost']
            })
        
        priority_df = pd.DataFrame(priority_data)
        priority_df = priority_df.sort_values('Priority_Score', ascending=False)
        
        return priority_df
    
    def create_strategy_dashboard(self, df, risk_probabilities):
        """Create comprehensive strategy analysis dashboard"""
        
        # Assign strategies to all borrowers
        strategies = []
        for i, (_, borrower) in enumerate(df.iterrows()):
            strategy = self.assign_recovery_strategy(borrower.to_dict(), risk_probabilities[i])
            strategies.append(strategy)
        
        # Strategy distribution
        strategy_counts = pd.Series(strategies).value_counts()
        fig_dist = px.pie(values=strategy_counts.values, names=strategy_counts.index,
                         title='Distribution of Recovery Strategies')
        
        # Expected recovery by strategy
        recovery_by_strategy = {}
        cost_by_strategy = {}
        
        for strategy in strategy_counts.index:
            strategy_mask = np.array(strategies) == strategy
            strategy_borrowers = df[strategy_mask]
            strategy_risks = np.array(risk_probabilities)[strategy_mask]
            
            total_recovery = 0
            total_cost = 0
            
            for j, (_, borrower) in enumerate(strategy_borrowers.iterrows()):
                roi_data = self.calculate_strategy_roi(
                    strategy, 
                    borrower['Outstanding_Loan_Amount'], 
                    strategy_risks[j]
                )
                total_recovery += roi_data['expected_recovery']
                total_cost += roi_data['cost']
            
            recovery_by_strategy[strategy] = total_recovery
            cost_by_strategy[strategy] = total_cost
        
        # ROI by strategy
        fig_roi = go.Figure()
        strategies_list = list(recovery_by_strategy.keys())
        recovery_values = list(recovery_by_strategy.values())
        cost_values = list(cost_by_strategy.values())
        roi_values = [(r-c)/c if c > 0 else r for r, c in zip(recovery_values, cost_values)]
        
        fig_roi.add_trace(go.Bar(
            name='Expected Recovery',
            x=strategies_list,
            y=recovery_values,
            yaxis='y'
        ))
        
        fig_roi.add_trace(go.Scatter(
            name='ROI',
            x=strategies_list,
            y=roi_values,
            yaxis='y2',
            mode='lines+markers',
            line=dict(color='red', width=3)
        ))
        
        fig_roi.update_layout(
            title='Recovery Strategy Analysis',
            xaxis_title='Strategy',
            yaxis=dict(title='Expected Recovery (₹)', side='left'),
            yaxis2=dict(title='ROI', side='right', overlaying='y'),
            height=500
        )
        
        # Priority heatmap
        priority_df = self.create_recovery_priority_list(df, risk_probabilities, strategies)
        
        fig_priority = px.scatter(priority_df.head(100), 
                                x='Priority_Score', 
                                y='Expected_Recovery',
                                color='Strategy',
                                size='Outstanding_Amount',
                                hover_data=['Borrower_ID', 'Risk_Probability'],
                                title='Top 100 Priority Borrowers')
        
        return fig_dist, fig_roi, fig_priority, priority_df
    
    def generate_collection_plan(self, priority_df, team_capacity=50, time_horizon_days=90):
        """Generate actionable collection plan for the team"""
        
        collection_plan = {
            'immediate_action': [],
            'weekly_targets': [],
            'monthly_goals': {},
            'resource_allocation': {}
        }
        
        # Immediate action (top priority borrowers)
        immediate_borrowers = priority_df.head(team_capacity).to_dict('records')
        collection_plan['immediate_action'] = immediate_borrowers
        
        # Weekly targets
        weeks = time_horizon_days // 7
        borrowers_per_week = len(priority_df) // weeks
        
        for week in range(weeks):
            start_idx = week * borrowers_per_week
            end_idx = min((week + 1) * borrowers_per_week, len(priority_df))
            week_borrowers = priority_df.iloc[start_idx:end_idx]
            
            collection_plan['weekly_targets'].append({
                'week': week + 1,
                'borrower_count': len(week_borrowers),
                'total_outstanding': week_borrowers['Outstanding_Amount'].sum(),
                'expected_recovery': week_borrowers['Expected_Recovery'].sum(),
                'strategies': week_borrowers['Strategy'].value_counts().to_dict()
            })
        
        # Monthly goals by strategy
        for strategy in priority_df['Strategy'].unique():
            strategy_data = priority_df[priority_df['Strategy'] == strategy]
            collection_plan['monthly_goals'][strategy] = {
                'borrower_count': len(strategy_data),
                'total_outstanding': strategy_data['Outstanding_Amount'].sum(),
                'expected_recovery': strategy_data['Expected_Recovery'].sum(),
                'avg_roi': strategy_data['ROI'].mean()
            }
        
        # Resource allocation recommendations
        strategy_workload = priority_df['Strategy'].value_counts()
        total_workload = len(priority_df)
        
        for strategy, count in strategy_workload.items():
            percentage = (count / total_workload) * 100
            collection_plan['resource_allocation'][strategy] = {
                'percentage': percentage,
                'recommended_agents': max(1, int(team_capacity * percentage / 100)),
                'avg_time_per_case': self.strategy_definitions[strategy]['time_to_recovery']
            }
        
        return collection_plan
    
    def export_recovery_report(self, priority_df, collection_plan):
        """Generate exportable recovery report"""
        
        report_data = {
            'summary': {
                'total_borrowers': len(priority_df),
                'total_outstanding': priority_df['Outstanding_Amount'].sum(),
                'total_expected_recovery': priority_df['Expected_Recovery'].sum(),
                'average_roi': priority_df['ROI'].mean(),
                'high_priority_borrowers': len(priority_df[priority_df['Priority_Score'] > 0.7])
            },
            'strategy_breakdown': collection_plan['monthly_goals'],
            'resource_allocation': collection_plan['resource_allocation'],
            'top_priority_borrowers': priority_df.head(50).to_dict('records'),
            'recommendations': self._generate_recommendations(priority_df, collection_plan)
        }
        
        return report_data
    
    def _generate_recommendations(self, priority_df, collection_plan):
        """Generate actionable recommendations for the collection team"""
        
        recommendations = []
        
        # High-value, low-risk borrowers
        high_value_low_risk = priority_df[
            (priority_df['Outstanding_Amount'] > 100000) & 
            (priority_df['Risk_Probability'] < 0.4)
        ]
        
        if len(high_value_low_risk) > 0:
            recommendations.append({
                'category': 'Quick Wins',
                'description': f'{len(high_value_low_risk)} high-value, low-risk borrowers should be contacted immediately',
                'expected_recovery': high_value_low_risk['Expected_Recovery'].sum(),
                'action': 'Prioritize phone calls and payment plan negotiations'
            })
        
        # Legal action candidates
        legal_candidates = priority_df[priority_df['Strategy'] == 'Legal Action']
        if len(legal_candidates) > 0:
            recommendations.append({
                'category': 'Legal Action',
                'description': f'{len(legal_candidates)} borrowers identified for legal proceedings',
                'expected_recovery': legal_candidates['Expected_Recovery'].sum(),
                'action': 'Prepare legal documentation and asset verification'
            })
        
        # Write-off candidates
        writeoff_candidates = priority_df[priority_df['Strategy'] == 'Write-off']
        if len(writeoff_candidates) > 0:
            recommendations.append({
                'category': 'Write-off Review',
                'description': f'{len(writeoff_candidates)} loans recommended for write-off',
                'expected_recovery': writeoff_candidates['Expected_Recovery'].sum(),
                'action': 'Review for tax benefits and final settlement attempts'
            })
        
        return recommendations

if __name__ == "__main__":
    # Test the recovery strategy engine
    from data_generator import LoanDataGenerator
    from data_preprocessor import DataPreprocessor
    from ml_models import RiskPredictionModel
    
    print("Testing Recovery Strategy Engine...")
    
    # Generate test data
    generator = LoanDataGenerator(n_borrowers=1000)
    raw_data = generator.generate_complete_dataset()
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.process_complete_pipeline(raw_data)
    
    # Train risk model
    X, y, _ = preprocessor.prepare_features_for_ml(processed_data)
    risk_model = RiskPredictionModel()
    risk_model.train_model(X, y)
    
    # Get risk probabilities
    _, risk_probabilities = risk_model.predict_risk(X)
    
    # Initialize recovery strategy engine
    recovery_engine = RecoveryStrategyEngine()
    
    # Test strategy assignment
    sample_borrower = raw_data.iloc[0].to_dict()
    sample_risk = risk_probabilities[0]
    
    strategy = recovery_engine.assign_recovery_strategy(sample_borrower, sample_risk)
    print(f"Assigned strategy for sample borrower: {strategy}")
    
    # Test strategy recommendations
    recommendations = recovery_engine.recommend_optimal_strategy(sample_borrower, sample_risk)
    print(f"Top 3 strategy recommendations:")
    for i, (strat, data) in enumerate(recommendations):
        print(f"  {i+1}. {strat} (Score: {data['total_score']:.3f}, ROI: {data['roi']:.2f})")
    
    # Create priority list
    strategies = [recovery_engine.assign_recovery_strategy(row.to_dict(), risk_probabilities[i]) 
                 for i, (_, row) in enumerate(raw_data.iterrows())]
    
    priority_df = recovery_engine.create_recovery_priority_list(raw_data, risk_probabilities, strategies)
    print(f"\nCreated priority list with {len(priority_df)} borrowers")
    print(f"Top priority borrower: {priority_df.iloc[0]['Borrower_ID']} (Score: {priority_df.iloc[0]['Priority_Score']:.3f})")
    
    # Generate collection plan
    collection_plan = recovery_engine.generate_collection_plan(priority_df)
    print(f"\nCollection plan generated:")
    print(f"  Immediate action borrowers: {len(collection_plan['immediate_action'])}")
    print(f"  Weekly targets: {len(collection_plan['weekly_targets'])} weeks")
    print(f"  Strategies to deploy: {len(collection_plan['monthly_goals'])}")
    
    print("\nRecovery Strategy Engine testing completed!")