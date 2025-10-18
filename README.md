# 🏦 Smart Loan Recovery System

An advanced Machine Learning-powered loan recovery system with interactive Streamlit dashboard for risk prediction, borrower segmentation, and recovery strategy optimization.

## 🌟 Features

### Core Functionality
- **Synthetic Dataset Generation**: Realistic financial data generation with proper distributions
- **Risk Prediction**: ML-powered risk scoring using Random Forest/Gradient Boosting
- **Borrower Segmentation**: K-Means clustering for borrower profiling
- **Early Warning System**: Proactive identification of high-risk borrowers
- **Recovery Strategies**: Dynamic strategy assignment based on risk profiles
- **Interactive Dashboard**: Comprehensive Streamlit interface with real-time analytics

### Advanced Features
- **Financial Health Scoring**: Composite metrics combining multiple financial indicators
- **Expected Recovery Calculation**: Probabilistic recovery amount estimation
- **Collection Plan Generation**: Actionable plans with team resource allocation
- **Export Capabilities**: CSV/JSON exports for reporting teams
- **Real-time Filtering**: Dynamic data filtering by demographics and loan characteristics
- **Performance Monitoring**: Model evaluation metrics and visualizations

## 🏗️ Architecture

### Data Pipeline
```
Raw Data → Data Preprocessing → Feature Engineering → ML Models → Predictions → Strategies → Dashboard
```

### Components
1. **Data Generator** (`data_generator.py`): Synthetic dataset creation
2. **Data Preprocessor** (`data_preprocessor.py`): Feature engineering and data cleaning
3. **ML Models** (`ml_models.py`): Risk prediction and clustering models
4. **Recovery Strategies** (`recovery_strategies.py`): Strategy recommendation engine
5. **Streamlit App** (`streamlit_app.py`): Interactive dashboard

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
```bash
# Clone or download the project
# Navigate to the project directory

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Start the Streamlit dashboard
streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

### Usage
1. **Generate Dataset**: Click "Generate New Dataset" to create synthetic data
2. **Configure Settings**: Use sidebar controls to adjust risk thresholds and filters
3. **Explore Tabs**: Navigate through different analytics sections
4. **Export Reports**: Download analysis results in CSV/JSON formats

## 📊 Dashboard Sections

### 1. Portfolio Overview
- **Key Metrics**: Total borrowers, outstanding amounts, risk distribution
- **Visualizations**: Risk pie charts, recovery status, loan amount analysis
- **Income Analysis**: Distribution patterns and correlations

### 2. Risk Analysis & Prediction
- **Model Performance**: Accuracy, precision, recall metrics
- **Feature Importance**: ML model explainability
- **High-Risk Identification**: Borrower tables with risk scores
- **Probability Distributions**: Risk score histograms

### 3. Borrower Segmentation
- **Cluster Visualization**: PCA-based 2D scatter plots
- **Segment Characteristics**: Statistical summaries by cluster
- **Interactive Filtering**: Segment-specific analysis
- **Naming Convention**: Meaningful cluster labels

### 4. Early Warning System
- **At-Risk Detection**: Proactive identification algorithms
- **Priority Scoring**: Multi-factor priority calculations
- **Alert Dashboard**: Visual indicators and urgency levels
- **Action Lists**: Ranked borrower tables for immediate attention

### 5. Recovery Strategies
- **Strategy Assignment**: Rule-based recommendation engine
- **ROI Analysis**: Expected returns by strategy type
- **Collection Plans**: Structured recovery roadmaps
- **Resource Allocation**: Team capacity optimization

### 6. Advanced Analytics
- **Financial Health**: Composite scoring metrics
- **Recovery Prediction**: Expected amounts and rates
- **Portfolio Statistics**: Comprehensive performance indicators
- **Export Options**: Data download capabilities

## 🤖 Machine Learning Models

### Risk Prediction Model
- **Algorithm**: Random Forest Classifier
- **Features**: 15+ engineered features including:
  - Borrower demographics
  - Loan characteristics
  - Payment history
  - Financial ratios
- **Performance**: ~85-90% accuracy on synthetic data
- **Output**: Risk probability scores (0-1)

### Borrower Segmentation
- **Algorithm**: K-Means Clustering
- **Features**: Income, loan amount, payment behavior, financial health
- **Segments**: 5 meaningful borrower categories:
  - High Income, Low Risk
  - Moderate Income, Low Risk
  - Moderate Income, High Payment Issues
  - High Loan Burden, High Risk
  - Low Income, High Risk

### Feature Engineering
- **Derived Metrics**: Debt-to-income ratio, payment punctuality score
- **Financial Health**: Composite scoring algorithm
- **Risk Indicators**: Multi-factor risk assessment
- **Categorical Encoding**: One-hot and label encoding for ML compatibility

## 📋 Recovery Strategies

### Strategy Types
1. **Automated Reminders**: Low-cost, automated contact (Success Rate: 15%)
2. **Phone Call Campaign**: Personal outreach (Success Rate: 35%)
3. **Payment Plan Negotiation**: Restructured agreements (Success Rate: 55%)
4. **Settlement Offer**: Discounted payoffs (Success Rate: 45%)
5. **Debt Collection Agency**: Third-party collection (Success Rate: 40%)
6. **Legal Action**: Court proceedings (Success Rate: 65%)
7. **Write-off**: Tax benefit optimization (Success Rate: 5%)

### Assignment Logic
- **Risk-based**: Strategy selection based on probability scores
- **Amount-based**: High-value loans get priority treatment
- **Collateral-aware**: Asset-backed loans receive different strategies
- **ROI Optimized**: Expected return calculations guide decisions

## 🔧 Configuration

### Risk Thresholds
- **High Risk**: Default 0.7 (adjustable 0.5-0.9)
- **Moderate Risk**: Default 0.4 (adjustable 0.2-0.6)
- **Custom Ranges**: Real-time threshold adjustment

### Data Generation Settings
- **Borrower Count**: 1,000 - 10,000 (default 5,000)
- **Distributions**: Log-normal income, realistic age/tenure patterns
- **Risk Factors**: Employment type, income levels, collateral presence

### Filtering Options
- **Employment Type**: Salaried vs Self-employed
- **Income Range**: Dynamic slider controls
- **Loan Amount**: Flexible range selection
- **Recovery Status**: Multi-select filtering

## 📈 Performance Metrics

### Model Evaluation
- **Accuracy**: Overall prediction correctness
- **Precision**: True positive rate for high-risk identification
- **Recall**: Sensitivity to actual high-risk cases
- **F1-Score**: Balanced precision-recall metric
- **ROC-AUC**: Area under receiver operating curve

### Business Metrics
- **Expected Recovery**: Probabilistic amount calculations
- **Collection Efficiency**: Strategy success rate optimization
- **Resource Utilization**: Team allocation recommendations
- **Priority Scoring**: Multi-factor borrower ranking

## 💡 Advanced Features

### Early Warning System
- **Predictive Alerts**: 3-month default probability
- **Multi-factor Analysis**: Payment patterns, financial stress indicators
- **Priority Scoring**: Weighted risk assessment
- **Intervention Recommendations**: Proactive action suggestions

### Financial Health Scoring
- **Composite Metrics**: Income stability, debt burden, payment history
- **Weighted Algorithm**: 
  - Income score (30%)
  - Payment punctuality (40%)
  - Debt burden (20%)
  - Collateral strength (10%)
- **Dynamic Updates**: Real-time recalculation

### Collection Plan Generation
- **Team Capacity**: Resource allocation optimization
- **Time Horizons**: Weekly/monthly target setting
- **Strategy Mix**: Balanced approach recommendations
- **ROI Maximization**: Profit-driven prioritization

## 📁 File Structure

```
smart-loan-recovery-system/
├── data_generator.py          # Synthetic data creation
├── data_preprocessor.py       # Feature engineering & cleaning
├── ml_models.py              # ML algorithms & evaluation
├── recovery_strategies.py     # Strategy engine & recommendations
├── streamlit_app.py          # Main dashboard application
├── requirements.txt          # Python dependencies
└── README.md                 # Documentation
```

## 🔐 Data Privacy & Security

- **Synthetic Data**: No real customer information used
- **Local Processing**: All computations run locally
- **Export Control**: Secure data download options
- **Configuration Privacy**: Sensitive settings stored in session state

## 🤝 Contributing

This is a demonstration project showcasing advanced ML techniques for loan recovery optimization. For production deployment, consider:

- Real data integration pipelines
- Database connectivity
- User authentication systems
- API endpoints for external systems
- Scheduled model retraining
- Performance monitoring dashboards

## 📞 Support

For questions about implementation or customization:
- Review the inline code documentation
- Check the Streamlit interface help tooltips
- Examine the model performance metrics
- Analyze the feature importance visualizations

## 🎯 Future Enhancements

- **Deep Learning Models**: Neural networks for complex pattern recognition
- **Real-time Processing**: Streaming data pipeline integration
- **A/B Testing**: Strategy effectiveness experimentation
- **Mobile Interface**: Responsive design for field teams
- **API Integration**: External system connectivity
- **Automated Reporting**: Scheduled report generation

## 📜 License

This project is for educational and demonstration purposes. Adapt the code structure and algorithms for your specific business requirements while ensuring compliance with financial regulations and data protection laws.

---

Built with ❤️ using Python, Streamlit, and Scikit-learn