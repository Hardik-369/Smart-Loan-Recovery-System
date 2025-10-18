import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_generator import LoanDataGenerator
from data_preprocessor import DataPreprocessor
from ml_models import BorrowerSegmentation, RiskPredictionModel, EarlyWarningSystem, AdvancedAnalytics
from recovery_strategies import RecoveryStrategyEngine

# Page configuration
st.set_page_config(
    page_title="Smart Loan Recovery System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
    }
    .risk-high {
        color: #ff4757;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffa726;
        font-weight: bold;
    }
    .risk-low {
        color: #2ed573;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'risk_model' not in st.session_state:
    st.session_state.risk_model = None
if 'segmentation_model' not in st.session_state:
    st.session_state.segmentation_model = None

# Utility functions
@st.cache_data
def load_and_process_data(n_borrowers=5000):
    """Load and process the loan data"""
    with st.spinner("Generating synthetic dataset..."):
        generator = LoanDataGenerator(n_borrowers=n_borrowers)
        raw_data = generator.generate_complete_dataset()
    
    with st.spinner("Preprocessing data..."):
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.process_complete_pipeline(raw_data)
    
    return raw_data, processed_data, preprocessor

@st.cache_resource
def train_models(processed_data, _preprocessor):
    """Train ML models"""
    with st.spinner("Training risk prediction model..."):
        X, y, feature_cols = _preprocessor.prepare_features_for_ml(processed_data)
        risk_model = RiskPredictionModel(model_type='random_forest')
        risk_model.train_model(X, y)
    
    with st.spinner("Training segmentation model..."):
        segmentation = BorrowerSegmentation(n_clusters=5)
        clustering_features = segmentation.prepare_clustering_features(processed_data)
        cluster_labels = segmentation.fit_kmeans(clustering_features)
    
    return risk_model, segmentation, X, y

def format_currency(amount):
    """Format currency in Indian format"""
    if amount >= 10000000:  # 1 crore
        return f"₹{amount/10000000:.2f}Cr"
    elif amount >= 100000:  # 1 lakh
        return f"₹{amount/100000:.2f}L"
    else:
        return f"₹{amount:,.0f}"

def get_risk_color(risk_prob):
    """Get color based on risk probability"""
    if risk_prob > 0.7:
        return "🔴"
    elif risk_prob > 0.4:
        return "🟡"
    else:
        return "🟢"

# Main application
def main():
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🏦 Smart Loan Recovery System</h1>
            <p>Advanced ML-powered loan recovery with risk prediction and strategy optimization</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data generation settings
        st.subheader("Data Settings")
        n_borrowers = st.slider("Number of Borrowers", 1000, 10000, 5000, 500)
        
        # Load/Generate data button
        if st.button("🔄 Generate New Dataset", type="primary"):
            st.session_state.data_loaded = False
            st.cache_data.clear()
            st.cache_resource.clear()
        
        # Risk threshold settings
        st.subheader("Risk Thresholds")
        high_risk_threshold = st.slider("High Risk Threshold", 0.5, 0.9, 0.7, 0.05)
        moderate_risk_threshold = st.slider("Moderate Risk Threshold", 0.2, 0.6, 0.4, 0.05)
        
        # Filter settings
        st.subheader("Filters")
        
    # Load data if not already loaded
    if not st.session_state.data_loaded:
        with st.spinner("Loading and processing data..."):
            raw_data, processed_data, preprocessor = load_and_process_data(n_borrowers)
            st.session_state.raw_data = raw_data
            st.session_state.processed_data = processed_data
            st.session_state.preprocessor = preprocessor
            
            # Train models
            risk_model, segmentation, X, y = train_models(processed_data, preprocessor)
            st.session_state.risk_model = risk_model
            st.session_state.segmentation_model = segmentation
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.data_loaded = True
    
    # Get data from session state
    raw_data = st.session_state.raw_data
    processed_data = st.session_state.processed_data
    risk_model = st.session_state.risk_model
    segmentation = st.session_state.segmentation_model
    X = st.session_state.X
    y = st.session_state.y
    
    # Get risk predictions
    risk_predictions, risk_probabilities = risk_model.predict_risk(X)
    
    # Apply filters in sidebar
    with st.sidebar:
        employment_filter = st.multiselect(
            "Employment Type",
            options=raw_data['Employment_Type'].unique(),
            default=raw_data['Employment_Type'].unique()
        )
        
        income_range = st.slider(
            "Monthly Income Range (₹)",
            int(raw_data['Monthly_Income'].min()),
            int(raw_data['Monthly_Income'].max()),
            (int(raw_data['Monthly_Income'].min()), int(raw_data['Monthly_Income'].max()))
        )
        
        loan_amount_range = st.slider(
            "Loan Amount Range (₹)",
            int(raw_data['Loan_Amount'].min()),
            int(raw_data['Loan_Amount'].max()),
            (int(raw_data['Loan_Amount'].min()), int(raw_data['Loan_Amount'].max()))
        )
        
        recovery_status_filter = st.multiselect(
            "Recovery Status",
            options=raw_data['Recovery_Status'].unique(),
            default=raw_data['Recovery_Status'].unique()
        )
    
    # Apply filters
    filtered_mask = (
        (raw_data['Employment_Type'].isin(employment_filter)) &
        (raw_data['Monthly_Income'].between(income_range[0], income_range[1])) &
        (raw_data['Loan_Amount'].between(loan_amount_range[0], loan_amount_range[1])) &
        (raw_data['Recovery_Status'].isin(recovery_status_filter))
    )
    
    filtered_raw_data = raw_data[filtered_mask].reset_index(drop=True)
    filtered_processed_data = processed_data[filtered_mask].reset_index(drop=True)
    filtered_risk_probabilities = risk_probabilities[filtered_mask]
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "🎯 Risk Analysis", "👥 Borrower Segmentation", 
        "⚠️ Early Warning", "📋 Recovery Strategies", "📈 Analytics"
    ])
    
    with tab1:
        st.header("📊 Portfolio Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_borrowers = len(filtered_raw_data)
            st.metric("Total Borrowers", f"{total_borrowers:,}")
        
        with col2:
            total_outstanding = filtered_raw_data['Outstanding_Loan_Amount'].sum()
            st.metric("Total Outstanding", format_currency(total_outstanding))
        
        with col3:
            avg_risk = filtered_risk_probabilities.mean()
            st.metric("Average Risk Score", f"{avg_risk:.3f}")
        
        with col4:
            high_risk_count = sum(filtered_risk_probabilities > high_risk_threshold)
            st.metric("High Risk Borrowers", f"{high_risk_count:,}")
        
        # Risk distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk Distribution")
            risk_categories = []
            for prob in filtered_risk_probabilities:
                if prob > high_risk_threshold:
                    risk_categories.append("High Risk")
                elif prob > moderate_risk_threshold:
                    risk_categories.append("Moderate Risk")
                else:
                    risk_categories.append("Low Risk")
            
            risk_counts = pd.Series(risk_categories).value_counts()
            fig_risk_dist = px.pie(values=risk_counts.values, names=risk_counts.index,
                                  color_discrete_map={
                                      'High Risk': '#ff4757',
                                      'Moderate Risk': '#ffa726',
                                      'Low Risk': '#2ed573'
                                  })
            st.plotly_chart(fig_risk_dist, use_container_width=True)
        
        with col2:
            st.subheader("Recovery Status Distribution")
            recovery_counts = filtered_raw_data['Recovery_Status'].value_counts()
            fig_recovery = px.bar(x=recovery_counts.index, y=recovery_counts.values,
                                title="Recovery Status Distribution")
            st.plotly_chart(fig_recovery, use_container_width=True)
        
        # Loan amount vs Risk scatter plot
        st.subheader("Loan Amount vs Risk Analysis")
        fig_scatter = px.scatter(
            filtered_raw_data, 
            x='Loan_Amount', 
            y=filtered_risk_probabilities,
            color='Employment_Type',
            size='Outstanding_Loan_Amount',
            hover_data=['Borrower_ID', 'Monthly_Income'],
            title="Loan Amount vs Risk Probability",
            labels={'y': 'Risk Probability'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Monthly income distribution
        st.subheader("Monthly Income Distribution")
        fig_income = px.histogram(filtered_raw_data, x='Monthly_Income', 
                                nbins=50, title="Monthly Income Distribution")
        st.plotly_chart(fig_income, use_container_width=True)
    
    with tab2:
        st.header("🎯 Risk Analysis & Prediction")
        
        # Model performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Accuracy", f"{risk_model.model_metrics['accuracy']:.3f}")
        with col2:
            st.metric("Precision", f"{risk_model.model_metrics['precision']:.3f}")
        with col3:
            st.metric("Recall", f"{risk_model.model_metrics['recall']:.3f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Feature importance plot
            st.subheader("Feature Importance")
            fig_importance = risk_model.create_feature_importance_plot()
            if fig_importance:
                st.plotly_chart(fig_importance, use_container_width=True)
        
        with col2:
            # Risk probability histogram
            st.subheader("Risk Probability Distribution")
            fig_risk_hist = px.histogram(x=filtered_risk_probabilities, 
                                       nbins=30, title="Risk Probability Distribution")
            st.plotly_chart(fig_risk_hist, use_container_width=True)
        
        # High-risk borrowers table
        st.subheader("High-Risk Borrowers")
        high_risk_mask = filtered_risk_probabilities > high_risk_threshold
        high_risk_data = filtered_raw_data[high_risk_mask].copy()
        high_risk_data['Risk_Probability'] = filtered_risk_probabilities[high_risk_mask]
        
        if len(high_risk_data) > 0:
            # Add risk emojis
            high_risk_data['Risk_Level'] = [get_risk_color(prob) for prob in high_risk_data['Risk_Probability']]
            
            display_cols = ['Borrower_ID', 'Risk_Level', 'Risk_Probability', 'Outstanding_Loan_Amount', 
                          'Days_Past_Due', 'Monthly_Income', 'Employment_Type']
            
            st.dataframe(
                high_risk_data[display_cols].head(20),
                column_config={
                    "Outstanding_Loan_Amount": st.column_config.NumberColumn(
                        "Outstanding Amount",
                        format="₹%d"
                    ),
                    "Monthly_Income": st.column_config.NumberColumn(
                        "Monthly Income",
                        format="₹%d"
                    ),
                    "Risk_Probability": st.column_config.ProgressColumn(
                        "Risk Probability",
                        help="Risk probability from 0 to 1",
                        min_value=0,
                        max_value=1,
                    ),
                }
            )
        else:
            st.info("No high-risk borrowers found with current filters.")
    
    with tab3:
        st.header("👥 Borrower Segmentation")
        
        # Get clustering results
        clustering_features = segmentation.prepare_clustering_features(filtered_processed_data)
        if len(clustering_features) > 0:
            cluster_labels = segmentation.kmeans.predict(clustering_features)
            
            # Cluster visualization
            fig_clusters, viz_df = segmentation.create_cluster_visualization(
                clustering_features, cluster_labels, filtered_raw_data
            )
            st.plotly_chart(fig_clusters, use_container_width=True)
            
            # Cluster characteristics
            st.subheader("Cluster Characteristics")
            
            cluster_summary = []
            for cluster_id, cluster_name in segmentation.cluster_names.items():
                cluster_mask = cluster_labels == cluster_id
                cluster_data = filtered_raw_data[cluster_mask]
                
                if len(cluster_data) > 0:
                    cluster_summary.append({
                        'Cluster': cluster_name,
                        'Count': len(cluster_data),
                        'Avg_Income': cluster_data['Monthly_Income'].mean(),
                        'Avg_Loan': cluster_data['Loan_Amount'].mean(),
                        'Avg_Outstanding': cluster_data['Outstanding_Loan_Amount'].mean(),
                        'Avg_Risk': filtered_risk_probabilities[cluster_mask].mean()
                    })
            
            if cluster_summary:
                cluster_df = pd.DataFrame(cluster_summary)
                st.dataframe(
                    cluster_df,
                    column_config={
                        "Avg_Income": st.column_config.NumberColumn(
                            "Avg Monthly Income",
                            format="₹%.0f"
                        ),
                        "Avg_Loan": st.column_config.NumberColumn(
                            "Avg Loan Amount",
                            format="₹%.0f"
                        ),
                        "Avg_Outstanding": st.column_config.NumberColumn(
                            "Avg Outstanding",
                            format="₹%.0f"
                        ),
                    }
                )
    
    with tab4:
        st.header("⚠️ Early Warning System")
        
        # Initialize early warning system
        early_warning = EarlyWarningSystem(risk_threshold=high_risk_threshold)
        
        # Identify at-risk borrowers
        at_risk_borrowers = early_warning.identify_at_risk_borrowers(
            filtered_raw_data, filtered_risk_probabilities
        )
        
        if len(at_risk_borrowers) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("At-Risk Borrowers", len(at_risk_borrowers))
                st.metric("Total At-Risk Amount", format_currency(at_risk_borrowers['Outstanding_Loan_Amount'].sum()))
            
            with col2:
                st.metric("Avg Priority Score", f"{at_risk_borrowers['Priority_Score'].mean():.3f}")
                st.metric("Immediate Action Required", len(at_risk_borrowers[at_risk_borrowers['Priority_Score'] > 0.8]))
            
            # Early warning visualizations
            fig_risk_dist, fig_priority = early_warning.create_early_warning_dashboard(at_risk_borrowers)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_risk_dist, use_container_width=True)
            with col2:
                st.plotly_chart(fig_priority, use_container_width=True)
            
            # Top priority borrowers
            st.subheader("Top Priority Borrowers for Immediate Action")
            priority_display = at_risk_borrowers.head(20)[
                ['Borrower_ID', 'Priority_Score', 'Risk_Probability', 'Outstanding_Loan_Amount', 
                 'Days_Past_Due', 'Monthly_Income']
            ].copy()
            
            priority_display['Action_Required'] = priority_display['Priority_Score'].apply(
                lambda x: "🚨 Urgent" if x > 0.8 else "⚠️ High" if x > 0.6 else "📌 Monitor"
            )
            
            st.dataframe(
                priority_display,
                column_config={
                    "Priority_Score": st.column_config.ProgressColumn(
                        "Priority Score",
                        min_value=0,
                        max_value=1,
                    ),
                    "Risk_Probability": st.column_config.ProgressColumn(
                        "Risk Probability",
                        min_value=0,
                        max_value=1,
                    ),
                    "Outstanding_Loan_Amount": st.column_config.NumberColumn(
                        "Outstanding Amount",
                        format="₹%d"
                    ),
                    "Monthly_Income": st.column_config.NumberColumn(
                        "Monthly Income",
                        format="₹%d"
                    ),
                }
            )
        else:
            st.info("No borrowers currently require early warning alerts.")
    
    with tab5:
        st.header("📋 Recovery Strategies & Recommendations")
        
        # Initialize recovery strategy engine
        recovery_engine = RecoveryStrategyEngine()
        
        # Generate strategy dashboard
        fig_dist, fig_roi, fig_priority, priority_df = recovery_engine.create_strategy_dashboard(
            filtered_raw_data, filtered_risk_probabilities
        )
        
        # Strategy distribution and ROI analysis
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_dist, use_container_width=True)
        with col2:
            st.plotly_chart(fig_roi, use_container_width=True)
        
        # Priority borrowers visualization
        st.plotly_chart(fig_priority, use_container_width=True)
        
        # Collection plan
        st.subheader("Recommended Collection Plan")
        collection_plan = recovery_engine.generate_collection_plan(priority_df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Immediate Action Required", len(collection_plan['immediate_action']))
        with col2:
            total_expected = sum([week['expected_recovery'] for week in collection_plan['weekly_targets']])
            st.metric("Total Expected Recovery", format_currency(total_expected))
        with col3:
            strategies_count = len(collection_plan['monthly_goals'])
            st.metric("Strategies to Deploy", strategies_count)
        
        # Strategy breakdown
        st.subheader("Strategy Breakdown")
        strategy_data = []
        for strategy, data in collection_plan['monthly_goals'].items():
            strategy_data.append({
                'Strategy': strategy,
                'Borrower_Count': data['borrower_count'],
                'Total_Outstanding': data['total_outstanding'],
                'Expected_Recovery': data['expected_recovery'],
                'Avg_ROI': data['avg_roi']
            })
        
        strategy_breakdown_df = pd.DataFrame(strategy_data)
        st.dataframe(
            strategy_breakdown_df,
            column_config={
                "Total_Outstanding": st.column_config.NumberColumn(
                    "Total Outstanding",
                    format="₹%.0f"
                ),
                "Expected_Recovery": st.column_config.NumberColumn(
                    "Expected Recovery",
                    format="₹%.0f"
                ),
                "Avg_ROI": st.column_config.NumberColumn(
                    "Average ROI",
                    format="%.2f"
                ),
            }
        )
        
        # Export recovery report
        if st.button("📊 Generate Recovery Report"):
            report_data = recovery_engine.export_recovery_report(priority_df, collection_plan)
            
            # Convert to JSON for download
            report_json = json.dumps(report_data, indent=2, default=str)
            st.download_button(
                label="📥 Download Recovery Report (JSON)",
                data=report_json,
                file_name=f"recovery_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            # Display summary
            st.subheader("Recovery Report Summary")
            st.json(report_data['summary'])
            
            # Display recommendations
            st.subheader("Recommended Actions")
            for rec in report_data['recommendations']:
                st.info(f"**{rec['category']}**: {rec['description']}\n\n*Action*: {rec['action']}")
    
    with tab6:
        st.header("📈 Advanced Analytics")
        
        # Financial health analysis
        st.subheader("Financial Health Analysis")
        if 'Financial_Health_Score' in processed_data.columns:
            health_scores = processed_data.loc[filtered_mask, 'Financial_Health_Score'].values
            
            col1, col2 = st.columns(2)
            with col1:
                fig_health = px.histogram(x=health_scores, nbins=30, title="Financial Health Score Distribution")
                st.plotly_chart(fig_health, use_container_width=True)
            
            with col2:
                # Health score vs Risk probability
                fig_health_risk = px.scatter(
                    x=health_scores, 
                    y=filtered_risk_probabilities,
                    title="Financial Health vs Risk Probability",
                    labels={'x': 'Financial Health Score', 'y': 'Risk Probability'}
                )
                st.plotly_chart(fig_health_risk, use_container_width=True)
        
        # Expected recovery analysis
        st.subheader("Expected Recovery Analysis")
        expected_recovery = AdvancedAnalytics.calculate_expected_recovery(
            filtered_raw_data, filtered_risk_probabilities
        )
        
        recovery_analysis = pd.DataFrame({
            'Current_Outstanding': filtered_raw_data['Outstanding_Loan_Amount'],
            'Expected_Recovery': expected_recovery,
            'Recovery_Rate': np.array(expected_recovery) / filtered_raw_data['Outstanding_Loan_Amount'],
            'Risk_Probability': filtered_risk_probabilities
        })
        
        fig_recovery_analysis = px.scatter(
            recovery_analysis, 
            x='Risk_Probability', 
            y='Recovery_Rate',
            size='Current_Outstanding',
            title="Risk vs Recovery Rate Analysis",
            labels={'Recovery_Rate': 'Expected Recovery Rate'}
        )
        st.plotly_chart(fig_recovery_analysis, use_container_width=True)
        
        # Summary statistics
        st.subheader("Portfolio Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Expected Recovery", format_currency(sum(expected_recovery)))
        with col2:
            avg_recovery_rate = np.mean(recovery_analysis['Recovery_Rate'])
            st.metric("Average Recovery Rate", f"{avg_recovery_rate:.2%}")
        with col3:
            potential_loss = sum(filtered_raw_data['Outstanding_Loan_Amount']) - sum(expected_recovery)
            st.metric("Potential Loss", format_currency(potential_loss))
        
        # Export data options
        st.subheader("📥 Export Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export Portfolio Data"):
                csv_buffer = io.StringIO()
                filtered_raw_data.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📊 Download Portfolio CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"portfolio_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("Export High-Risk Borrowers"):
                high_risk_mask = filtered_risk_probabilities > high_risk_threshold
                high_risk_export = filtered_raw_data[high_risk_mask].copy()
                high_risk_export['Risk_Probability'] = filtered_risk_probabilities[high_risk_mask]
                
                csv_buffer = io.StringIO()
                high_risk_export.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="🚨 Download High-Risk CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"high_risk_borrowers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("Export Recovery Analysis"):
                csv_buffer = io.StringIO()
                recovery_analysis.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="💰 Download Recovery Analysis CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"recovery_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()