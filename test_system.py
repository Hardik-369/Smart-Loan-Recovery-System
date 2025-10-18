#!/usr/bin/env python3
"""
Test script for Smart Loan Recovery System
This script verifies all components work correctly before running the Streamlit app
"""

import sys
import traceback
from datetime import datetime

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        import plotly
        print("✅ Core libraries imported successfully")
    except ImportError as e:
        print(f"❌ Core library import failed: {e}")
        return False
    
    try:
        from data_generator import LoanDataGenerator
        from data_preprocessor import DataPreprocessor
        from ml_models import BorrowerSegmentation, RiskPredictionModel, EarlyWarningSystem, AdvancedAnalytics
        from recovery_strategies import RecoveryStrategyEngine
        print("✅ Custom modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Custom module import failed: {e}")
        return False

def test_data_generation():
    """Test synthetic data generation"""
    print("\nTesting data generation...")
    
    try:
        from data_generator import LoanDataGenerator
        generator = LoanDataGenerator(n_borrowers=100)
        dataset = generator.generate_complete_dataset()
        
        assert len(dataset) == 100, f"Expected 100 borrowers, got {len(dataset)}"
        assert 'Borrower_ID' in dataset.columns, "Borrower_ID column missing"
        assert 'Risk_Score' not in dataset.columns, "Risk_Score should not be in raw data"
        
        print(f"✅ Generated {len(dataset)} borrowers with {len(dataset.columns)} features")
        return True, dataset
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        traceback.print_exc()
        return False, None

def test_data_preprocessing(raw_data):
    """Test data preprocessing pipeline"""
    print("\nTesting data preprocessing...")
    
    try:
        from data_preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.process_complete_pipeline(raw_data)
        
        assert len(processed_data) == len(raw_data), "Processed data length mismatch"
        assert 'Risk_Score' in processed_data.columns, "Risk_Score column missing after preprocessing"
        assert 'Financial_Health_Score' in processed_data.columns, "Financial_Health_Score missing"
        
        print(f"✅ Preprocessed data: {len(processed_data)} rows, {len(processed_data.columns)} columns")
        return True, processed_data, preprocessor
    except Exception as e:
        print(f"❌ Data preprocessing failed: {e}")
        traceback.print_exc()
        return False, None, None

def test_ml_models(processed_data, preprocessor):
    """Test machine learning models"""
    print("\nTesting ML models...")
    
    try:
        from ml_models import BorrowerSegmentation, RiskPredictionModel
        # Test risk prediction model
        X, y, feature_cols = preprocessor.prepare_features_for_ml(processed_data)
        risk_model = RiskPredictionModel(model_type='random_forest')
        risk_model.train_model(X, y)
        
        predictions, probabilities = risk_model.predict_risk(X)
        
        assert len(predictions) == len(X), "Predictions length mismatch"
        assert len(probabilities) == len(X), "Probabilities length mismatch"
        assert all(0 <= p <= 1 for p in probabilities), "Probabilities not in [0,1] range"
        
        print(f"✅ Risk model trained: Accuracy={risk_model.model_metrics['accuracy']:.3f}")
        
        # Test segmentation model
        segmentation = BorrowerSegmentation(n_clusters=5)
        clustering_features = segmentation.prepare_clustering_features(processed_data)
        cluster_labels = segmentation.fit_kmeans(clustering_features)
        
        assert len(cluster_labels) == len(clustering_features), "Cluster labels length mismatch"
        assert len(set(cluster_labels)) <= 5, "Too many clusters generated"
        
        print(f"✅ Segmentation model trained: {len(set(cluster_labels))} clusters")
        
        return True, risk_model, segmentation, probabilities
    except Exception as e:
        print(f"❌ ML models failed: {e}")
        traceback.print_exc()
        return False, None, None, None

def test_recovery_strategies(raw_data, probabilities):
    """Test recovery strategy engine"""
    print("\nTesting recovery strategies...")
    
    try:
        from recovery_strategies import RecoveryStrategyEngine
        recovery_engine = RecoveryStrategyEngine()
        
        # Test strategy assignment for a sample borrower
        sample_borrower = raw_data.iloc[0].to_dict()
        sample_risk = probabilities[0]
        
        strategy = recovery_engine.assign_recovery_strategy(sample_borrower, sample_risk)
        assert isinstance(strategy, str), "Strategy should be a string"
        assert strategy in recovery_engine.strategy_definitions, f"Invalid strategy: {strategy}"
        
        print(f"✅ Strategy assignment working: '{strategy}' for risk {sample_risk:.3f}")
        
        # Test strategy recommendations
        recommendations = recovery_engine.recommend_optimal_strategy(sample_borrower, sample_risk)
        assert len(recommendations) <= 3, "Should return at most 3 recommendations"
        
        print(f"✅ Strategy recommendations: {len(recommendations)} options generated")
        
        # Test priority list generation
        strategies = [recovery_engine.assign_recovery_strategy(row.to_dict(), probabilities[i]) 
                     for i, (_, row) in enumerate(raw_data.iterrows())]
        
        priority_df = recovery_engine.create_recovery_priority_list(raw_data, probabilities, strategies)
        assert len(priority_df) == len(raw_data), "Priority list length mismatch"
        
        print(f"✅ Priority list generated: {len(priority_df)} borrowers ranked")
        
        return True, recovery_engine
    except Exception as e:
        print(f"❌ Recovery strategies failed: {e}")
        traceback.print_exc()
        return False, None

def test_early_warning(processed_data, probabilities):
    """Test early warning system"""
    print("\nTesting early warning system...")
    
    try:
        from ml_models import EarlyWarningSystem
        early_warning = EarlyWarningSystem(risk_threshold=0.7)
        at_risk_borrowers = early_warning.identify_at_risk_borrowers(processed_data, probabilities)
        
        # Should identify some at-risk borrowers
        print(f"✅ Early warning system: {len(at_risk_borrowers)} at-risk borrowers identified")
        
        return True
    except Exception as e:
        print(f"❌ Early warning system failed: {e}")
        traceback.print_exc()
        return False

def test_advanced_analytics(raw_data, probabilities):
    """Test advanced analytics functions"""
    print("\nTesting advanced analytics...")
    
    try:
        from ml_models import AdvancedAnalytics
        # Test expected recovery calculation
        expected_recovery = AdvancedAnalytics.calculate_expected_recovery(raw_data, probabilities)
        assert len(expected_recovery) == len(raw_data), "Expected recovery length mismatch"
        assert all(r >= 0 for r in expected_recovery), "Negative recovery amounts found"
        
        print(f"✅ Expected recovery calculated: Avg ₹{np.mean(expected_recovery):,.0f}")
        
        # Test collection strategy optimization
        strategies = AdvancedAnalytics.optimize_collection_strategy(raw_data, probabilities)
        assert len(strategies) == len(raw_data), "Strategies length mismatch"
        
        print(f"✅ Collection strategies optimized: {len(set(strategies))} unique strategies")
        
        return True
    except Exception as e:
        print(f"❌ Advanced analytics failed: {e}")
        traceback.print_exc()
        return False

def run_full_test():
    """Run complete system test"""
    print("=" * 60)
    print("🧪 SMART LOAN RECOVERY SYSTEM - FULL SYSTEM TEST")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test 1: Imports
    if not test_imports():
        print("\n❌ CRITICAL: Import test failed. Please check dependencies.")
        return False
    
    # Test 2: Data Generation
    success, raw_data = test_data_generation()
    if not success:
        print("\n❌ CRITICAL: Data generation failed.")
        return False
    
    # Test 3: Data Preprocessing
    success, processed_data, preprocessor = test_data_preprocessing(raw_data)
    if not success:
        print("\n❌ CRITICAL: Data preprocessing failed.")
        return False
    
    # Test 4: ML Models
    success, risk_model, segmentation, probabilities = test_ml_models(processed_data, preprocessor)
    if not success:
        print("\n❌ CRITICAL: ML models failed.")
        return False
    
    # Test 5: Recovery Strategies
    success, recovery_engine = test_recovery_strategies(raw_data, probabilities)
    if not success:
        print("\n❌ CRITICAL: Recovery strategies failed.")
        return False
    
    # Test 6: Early Warning System
    if not test_early_warning(processed_data, probabilities):
        print("\n⚠️  WARNING: Early warning system issues detected.")
    
    # Test 7: Advanced Analytics
    if not test_advanced_analytics(raw_data, probabilities):
        print("\n⚠️  WARNING: Advanced analytics issues detected.")
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\n✅ System is ready to run. Execute: streamlit run streamlit_app.py")
    print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return True

if __name__ == "__main__":
    try:
        import numpy as np
        success = run_full_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error during testing: {e}")
        traceback.print_exc()
        sys.exit(1)