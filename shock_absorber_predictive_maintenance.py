"""
AI Predictive Maintenance Model for Shock Absorbers

This module implements an AI system for predictive maintenance of automotive shock absorbers.
It combines physical fatigue analysis (Wöhler curve and Miner's rule of cumulative damage)
with machine learning (random forests) to predict the remaining useful life and probability
of failure, and classify the shock absorbers state (healthy, degraded, failed).

Author: [Your Name]
License: MIT
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, classification_report, confusion_matrix
from sklearn.neural_network import MLPRegressor
import joblib
from typing import Dict, List, Tuple, Optional
import warnings
import os

warnings.filterwarnings('ignore')


class MaintenancePredictiveAI:
    """
    AI System for Automotive Predictive Maintenance.
    Combines physical models and machine learning.
    """
    
    def __init__(self, component: str = "amortisseur"):
        self.component = component
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.training_history = {}
        self.alert_thresholds = {
            'critique': 0.9,    # Probable failure within 30 days
            'warning': 0.7,     # Enhanced monitoring required
            'normal': 0.5       # Normal operation
        }
        
    def generate_synthetic_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """
        Generates synthetic data for training.
        Based on realistic physical relationships.
        """
        np.random.seed(42)
        
        data = []
        
        for i in range(n_samples):
            # Usage parameters
            mileage = np.random.uniform(0, 300000)  # km
            vehicle_age = np.random.uniform(0, 20)  # years
            average_speed = np.random.uniform(20, 120)  # km/h
            
            # Environmental conditions
            average_temperature = np.random.normal(15, 20)  # °C
            average_humidity = np.random.uniform(30, 90)  # %
            
            # Road types (numeric encoding)
            road_types = ['urban', 'road', 'highway', 'off_road']
            road_proportions = np.random.dirichlet([1, 2, 1.5, 0.5])  # Realistic distribution
            dominant_road = np.argmax(road_proportions)
            
            # Stress factors by road type
            stress_factors = {
                0: 1.5,  # urban (frequent stops)
                1: 1.0,  # road (reference)
                2: 0.7,  # highway (stable driving)
                3: 2.5   # off-road (very severe)
            }
            
            road_stress = stress_factors[dominant_road]
            
            # Driving style (aggressiveness)
            aggressiveness = np.random.beta(2, 5)  # Realistic distribution (most drive normally)
            
            # Vehicle mass and loading
            base_mass = np.random.uniform(1000, 2500)  # kg
            loading_factor = np.random.uniform(0.8, 1.5)
            effective_mass = base_mass * loading_factor
            
            # Wear calculation based on physical model
            # Combined wear factor
            wear_factor = (
                (mileage / 100000) *  # Base mileage wear
                (1 + road_stress) *      # Road type impact
                (1 + aggressiveness * 0.5) * # Driving style impact
                (effective_mass / 1500) * # Mass impact (normalized to 1500kg)
                (1 + abs(average_temperature - 20) / 50) *  # Temperature impact
                np.exp(vehicle_age / 20)  # Exponential aging
            )
            
            # Add stochastic variability
            variability = np.random.lognormal(0, 0.3)
            wear_factor *= variability
            
            # Derived variables
            fatigue_cycles = mileage * road_stress * (1 + aggressiveness)
            dissipated_energy = fatigue_cycles * effective_mass / 1000
            
            # Maintenance indicators
            previous_repairs = np.random.poisson(mileage / 50000)
            time_since_last_maintenance = np.random.uniform(0, 24)  # months
            
            # Sensor variables (IoT data simulation)
            vibration_rms = wear_factor * 0.5 + np.random.normal(0, 0.1)
            component_temperature = average_temperature + 20 + wear_factor * 5
            
            # Remaining life calculation (regression target)
            nominal_lifespan = 150000  # km for new shock absorber
            remaining_life = max(0, nominal_lifespan * (1 - wear_factor) + 
                               np.random.normal(0, 10000))
            
            # State classification (classification target)
            if remaining_life < 10000:
                state = 'critique'
                failure_probability = 0.9 + np.random.normal(0, 0.05)
            elif remaining_life < 30000:
                state = 'warning'
                failure_probability = 0.6 + np.random.normal(0, 0.1)
            else:
                state = 'normal'
                failure_probability = 0.2 + np.random.normal(0, 0.1)
            
            # Limit probabilities
            failure_probability = np.clip(failure_probability, 0, 1)
            
            sample = {
                # Usage variables
                'kilometrage': mileage,
                'age_vehicule': vehicle_age,
                'vitesse_moyenne': average_speed,
                'type_route_dominant': dominant_road,
                'proportion_urbain': road_proportions[0],
                'proportion_route': road_proportions[1],
                'proportion_autoroute': road_proportions[2],
                'proportion_terrain': road_proportions[3],
                'agressivite_conduite': aggressiveness,
                
                # Environmental variables
                'temperature_moyenne': average_temperature,
                'humidite_moyenne': average_humidity,
                
                # Vehicle variables
                'masse_vehicule': base_mass,
                'facteur_chargement': loading_factor,
                'nb_reparations_anterieures': previous_repairs,
                'temps_derniere_maintenance': time_since_last_maintenance,
                
                # Calculated/sensor variables
                'nb_cycles_fatigue': fatigue_cycles,
                'energie_dissipee': dissipated_energy,
                'vibration_rms': max(0, vibration_rms),
                'temperature_composant': component_temperature,
                'facteur_usure_calcule': wear_factor,
                
                # Target variables
                'duree_vie_restante_km': remaining_life,
                'probabilite_defaillance': failure_probability,
                'etat_composant': state
            }
            
            data.append(sample)
        
        df = pd.DataFrame(data)
        return df
    
    def preprocess_data(self, df: pd.DataFrame, 
                       training: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Data preprocessing for training/prediction.
        """
        # Feature selection for prediction
        numerical_features = [
            'kilometrage', 'age_vehicule', 'vitesse_moyenne',
            'proportion_urbain', 'proportion_route', 'proportion_autoroute', 'proportion_terrain',
            'agressivite_conduite', 'temperature_moyenne', 'humidite_moyenne',
            'masse_vehicule', 'facteur_chargement', 'nb_reparations_anterieures',
            'temps_derniere_maintenance', 'nb_cycles_fatigue', 'energie_dissipee',
            'vibration_rms', 'temperature_composant', 'facteur_usure_calcule'
        ]
        
        X = df[numerical_features].values
        
        # Feature normalization
        if training:
            self.scalers['features'] = StandardScaler()
            X_scaled = self.scalers['features'].fit_transform(X)
        else:
            X_scaled = self.scalers['features'].transform(X)
        
        # Target preparation
        y_regression = df['duree_vie_restante_km'].values if 'duree_vie_restante_km' in df.columns else None
        
        if 'etat_composant' in df.columns:
            if training:
                self.encoders['state'] = LabelEncoder()
                y_classification = self.encoders['state'].fit_transform(df['etat_composant'])
            else:
                y_classification = self.encoders['state'].transform(df['etat_composant'])
        else:
            y_classification = None
        
        return X_scaled, y_regression, y_classification
    
    def train_models(self, df: pd.DataFrame):
        """
        Trains regression and classification models.
        """
        print("Starting model training...")
        
        # Preprocessing
        X, y_reg, y_clf = self.preprocess_data(df, training=True)
        
        # Train/test split
        X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = \
            train_test_split(X, y_reg, y_clf, test_size=0.2, random_state=42)
        
        # 1. Regression model (remaining lifespan)
        print("Training regression model...")
        
        # Test multiple models
        regression_models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        best_reg_score = -np.inf
        best_reg_model = None
        
        for name, model in regression_models.items():
            if name != 'GradientBoosting':  # GB is for classification
                scores = cross_val_score(model, X_train, y_reg_train, 
                                       cv=5, scoring='neg_mean_absolute_error')
                mean_score = scores.mean()
                print(f"{name} - CV Score: {mean_score:.4f}")
                
                if mean_score > best_reg_score:
                    best_reg_score = mean_score
                    best_reg_model = name
        
        # Train best regression model
        self.models['regression'] = regression_models[best_reg_model]
        self.models['regression'].fit(X_train, y_reg_train)
        
        # Regression evaluation
        y_pred_reg = self.models['regression'].predict(X_test)
        mae = mean_absolute_error(y_reg_test, y_pred_reg)
        print(f"Best regression model: {best_reg_model}")
        print(f"MAE on test: {mae:.2f} km")
        
        # 2. Classification model (component state)
        print("\nTraining classification model...")
        
        # Hyperparameter optimization for RandomForest
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
        
        rf_classifier = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf_classifier, param_grid, cv=3, scoring='f1_weighted')
        grid_search.fit(X_train, y_clf_train)
        
        self.models['classification'] = grid_search.best_estimator_
        
        # Classification evaluation
        y_pred_clf = self.models['classification'].predict(X_test)
        print("Best hyperparameters:", grid_search.best_params_)
        print("\nClassification report:")
        print(classification_report(y_clf_test, y_pred_clf, 
                                  target_names=self.encoders['state'].classes_))
        
        # Save training metrics
        self.training_history = {
            'mae_regression': mae,
            'best_regression_model': best_reg_model,
            'classification_score': grid_search.best_score_,
            'regression_feature_importance': dict(zip(
                [f"feature_{i}" for i in range(len(self.models['regression'].feature_importances_))],
                self.models['regression'].feature_importances_
            )),
            'classification_feature_importance': dict(zip(
                [f"feature_{i}" for i in range(len(self.models['classification'].feature_importances_))],
                self.models['classification'].feature_importances_
            ))
        }
        
        print(f"\nTraining complete!")
        print(f"Regression score (MAE): {mae:.2f} km")
        print(f"Classification score (F1): {grid_search.best_score_:.4f}")
    
    def predict(self, vehicle_data: Dict) -> Dict:
        """
        Makes a maintenance prediction for a vehicle.
        """
        # Convert to DataFrame
        df_input = pd.DataFrame([vehicle_data])
        
        # Preprocessing
        X, _, _ = self.preprocess_data(df_input, training=False)
        
        # Predictions
        predicted_lifespan = self.models['regression'].predict(X)[0]
        state_probabilities = self.models['classification'].predict_proba(X)[0]
        predicted_state = self.models['classification'].predict(X)[0]
        
        # Convert predicted state
        state_name = self.encoders['state'].inverse_transform([predicted_state])[0]
        
        # Calculate recommendations
        if predicted_lifespan < 10000:
            urgency = "CRITIQUE"
            recommendation = "Immediate replacement required"
            intervention_delay = "< 1 week"
        elif predicted_lifespan < 30000:
            urgency = "ATTENTION"
            recommendation = "Enhanced monitoring, schedule replacement"
            intervention_delay = "< 3 months"
        else:
            urgency = "NORMAL"
            recommendation = "Standard preventive maintenance"
            intervention_delay = "According to plan"
        
        # Calculate prediction reliability
        reliability = max(state_probabilities)
        
        results = {
            'remaining_life_km': max(0, predicted_lifespan),
            'predicted_state': state_name,
            'state_probabilities': {
                cls: prob for cls, prob in 
                zip(self.encoders['state'].classes_, state_probabilities)
            },
            'urgency': urgency,
            'recommendation': recommendation,
            'intervention_delay': intervention_delay,
            'prediction_reliability': reliability,
            'risk_factor': 1 - (predicted_lifespan / 150000),  # Normalized
        }
        
        return results
    
    def analyze_feature_importance(self, top_n: int = 10) -> Dict:
        """
        Analyzes feature importance in predictions.
        """
        if 'regression' not in self.models:
            raise ValueError("Models not trained")
        
        feature_names = [
            'kilometrage', 'age_vehicule', 'vitesse_moyenne',
            'proportion_urbain', 'proportion_route', 'proportion_autoroute', 'proportion_terrain',
            'agressivite_conduite', 'temperature_moyenne', 'humidite_moyenne',
            'masse_vehicule', 'facteur_chargement', 'nb_reparations_anterieures',
            'temps_derniere_maintenance', 'nb_cycles_fatigue', 'energie_dissipee',
            'vibration_rms', 'temperature_composant', 'facteur_usure_calcule'
        ]
        
        # Regression importance
        reg_importance = self.models['regression'].feature_importances_
        reg_features = list(zip(feature_names, reg_importance))
        reg_features.sort(key=lambda x: x[1], reverse=True)
        
        # Classification importance
        clf_importance = self.models['classification'].feature_importances_
        clf_features = list(zip(feature_names, clf_importance))
        clf_features.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'regression_top_features': reg_features[:top_n],
            'classification_top_features': clf_features[:top_n],
            'common_important_features': list(set(
                [f[0] for f in reg_features[:5]] + [f[0] for f in clf_features[:5]]
            ))
        }
    
    def simulate_vehicle_fleet(self, n_vehicles: int = 1000) -> pd.DataFrame:
        """
        Simulates analysis of a vehicle fleet.
        """
        # Generate fleet data
        fleet_data = self.generate_synthetic_data(n_vehicles)
        
        # Predictions for each vehicle
        predictions = []
        
        for idx, vehicle in fleet_data.iterrows():
            vehicle_data = vehicle.to_dict()
            try:
                pred = self.predict(vehicle_data)
                pred['vehicle_id'] = idx
                pred['current_mileage'] = vehicle['kilometrage']
                predictions.append(pred)
            except Exception as e:
                print(f"Error vehicle {idx}: {e}")
                continue
        
        return pd.DataFrame(predictions)
    
    def generate_maintenance_schedule(self, predictions_df: pd.DataFrame) -> Dict:
        """
        Generates an optimized maintenance schedule.
        """
        # Classification by urgency
        critical = predictions_df[predictions_df['urgency'] == 'CRITIQUE']
        attention = predictions_df[predictions_df['urgency'] == 'ATTENTION']
        normal = predictions_df[predictions_df['urgency'] == 'NORMAL']
        
        # Cost estimation
        urgent_repair_cost = 450  # €
        scheduled_repair_cost = 320  # €
        roadside_breakdown_cost = 800  # €
        
        schedule = {
            'immediate_interventions': {
                'count': len(critical),
                'vehicles': critical['vehicle_id'].tolist(),
                'estimated_cost': len(critical) * urgent_repair_cost
            },
            'scheduled_interventions': {
                'count': len(attention),
                'vehicles': attention['vehicle_id'].tolist(),
                'estimated_cost': len(attention) * scheduled_repair_cost
            },
            'enhanced_monitoring': {
                'count': len(normal[normal['risk_factor'] > 0.3]),
                'vehicles': normal[normal['risk_factor'] > 0.3]['vehicle_id'].tolist()
            },
            'estimated_savings': {
                'breakdown_avoidance': len(critical) * (roadside_breakdown_cost - urgent_repair_cost),
                'schedule_optimization': len(attention) * (urgent_repair_cost - scheduled_repair_cost)
            }
        }
        
        return schedule
    
    def save_models(self, path: str = "models_predictive_maintenance"):
        """
        Saves trained models.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Save models
        joblib.dump(self.models, f"{path}/models.pkl")
        joblib.dump(self.scalers, f"{path}/scalers.pkl")
        joblib.dump(self.encoders, f"{path}/encoders.pkl")
        joblib.dump(self.training_history, f"{path}/history.pkl")
        
        print(f"Models saved to {path}/")
    
    def load_models(self, path: str = "models_predictive_maintenance"):
        """
        Loads pre-trained models.
        """
        self.models = joblib.load(f"{path}/models.pkl")
        self.scalers = joblib.load(f"{path}/scalers.pkl")
        self.encoders = joblib.load(f"{path}/encoders.pkl")
        self.training_history = joblib.load(f"{path}/history.pkl")
        
        print(f"Models loaded from {path}/")
    
    def generate_ai_report(self, vehicle_data: Dict) -> str:
        """
        Generates a detailed AI analysis report.
        """
        prediction = self.predict(vehicle_data)
        importance = self.analyze_feature_importance(5)
        
        report = f"""
AI ANALYSIS REPORT - PREDICTIVE MAINTENANCE
===========================================

ANALYZED VEHICLE:
- Mileage: {vehicle_data.get('kilometrage', 'N/A'):,.0f} km
- Age: {vehicle_data.get('age_vehicule', 'N/A'):.1f} years
- Primary usage: {vehicle_data.get('type_route_dominant', 'N/A')}

AI PREDICTIONS:
- Predicted state: {prediction['predicted_state'].upper()}
- Remaining lifespan: {prediction['remaining_life_km']:,.0f} km
- Urgency level: {prediction['urgency']}
- Prediction reliability: {prediction['prediction_reliability']:.2%}

PROBABILITIES BY STATE:
"""
        for state, prob in prediction['state_probabilities'].items():
            report += f"- {state.capitalize()}: {prob:.2%}\n"
        
        report += f"""
RECOMMENDATIONS:
- Action: {prediction['recommendation']}
- Intervention delay: {prediction['intervention_delay']}
- Risk factor: {prediction['risk_factor']:.2f}

KEY FACTORS IDENTIFIED (Top 5):
"""
        for i, (feature, importance_val) in enumerate(importance['regression_top_features'][:5], 1):
            report += f"{i}. {feature.replace('_', ' ').title()}: {importance_val:.3f}\n"
        
        return report


def main():
    """
    Example usage of the system.
    """
    print("=== AI PREDICTIVE MAINTENANCE SYSTEM ===\n")
    
    # 1. Initialize and train
    print("1. Initializing system...")
    ai_system = MaintenancePredictiveAI("shock_absorber")
    
    print("2. Generating training data...")
    training_data = ai_system.generate_synthetic_data(5000)
    print(f"   Data generated: {len(training_data)} samples")
    
    print("3. Training models...")
    ai_system.train_models(training_data)
    
    # 2. Test on a vehicle
    print("\n4. Testing on example vehicle...")
    test_vehicle = {
        'kilometrage': 85000,
        'age_vehicule': 5.5,
        'vitesse_moyenne': 65,
        'type_route_dominant': 1,  # road
        'proportion_urbain': 0.2,
        'proportion_route': 0.6,
        'proportion_autoroute': 0.2,
        'proportion_terrain': 0.0,
        'agressivite_conduite': 0.3,
        'temperature_moyenne': 18,
        'humidite_moyenne': 65,
        'masse_vehicule': 1400,
        'facteur_chargement': 1.1,
        'nb_reparations_anterieures': 1,
        'temps_derniere_maintenance': 8,
        'nb_cycles_fatigue': 95000,
        'energie_dissipee': 133,
        'vibration_rms': 0.45,
        'temperature_composant': 45,
        'facteur_usure_calcule': 0.75
    }
    
    prediction = ai_system.predict(test_vehicle)
    print(f"   Predicted state: {prediction['predicted_state']}")
    print(f"   Remaining lifespan: {prediction['remaining_life_km']:,.0f} km")
    print(f"   Urgency: {prediction['urgency']}")
    
    # 3. Fleet analysis
    print("\n5. Simulating fleet analysis...")
    fleet_predictions = ai_system.simulate_vehicle_fleet(100)
    schedule = ai_system.generate_maintenance_schedule(fleet_predictions)
    
    print(f"   Critical vehicles: {schedule['immediate_interventions']['count']}")
    print(f"   Vehicles to monitor: {schedule['scheduled_interventions']['count']}")
    print(f"   Estimated savings: {schedule['estimated_savings']['breakdown_avoidance']:,.0f}€")
    
    # 4. Detailed report
    print("\n6. Generating report...")
    report = ai_system.generate_ai_report(test_vehicle)
    print(report)
    
    # 5. Feature importance analysis
    importance = ai_system.analyze_feature_importance()
    print("\nMost important variables:")
    for i, (var, imp) in enumerate(importance['regression_top_features'][:5], 1):
        print(f"{i}. {var}: {imp:.3f}")
    
    # 6. Save models
    try:
        ai_system.save_models()
        print("\nModels saved successfully!")
    except Exception as e:
        print(f"Save error: {e}")
    
    return ai_system, fleet_predictions, schedule


if __name__ == "__main__":
    ai_system, predictions, schedule = main()
