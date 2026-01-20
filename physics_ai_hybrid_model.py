"""
Physics-Informed AI Hybrid Model for Shock Absorber Predictive Maintenance

This module implements a hybrid Physics-AI system that combines:
- Physical fatigue model (Wöhler curve + Miner's cumulative damage rule)
- Machine Learning (RandomForest) for correction factor learning

The hybrid approach uses physics to compute baseline predictions, then ML learns
to correct for real-world effects not captured by the physical model.

Author: [Your Name]
License: MIT
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, classification_report, r2_score
import joblib
from typing import Dict, List, Tuple, Optional
import warnings
import os

warnings.filterwarnings('ignore')


# =============================================================================
# PART 1: PHYSICAL FATIGUE MODEL (Wöhler + Miner)
# =============================================================================

class FatiguePhysicsModel:
    """
    Physical fatigue model for shock absorber based on:
    - Wöhler S-N curve for cycles to failure
    - Miner's rule for cumulative damage accumulation
    - Correction factors: surface, size, reliability, temperature, corrosion
    """
    
    def __init__(self, 
                 Sut: float = 600,      # Ultimate tensile strength (MPa)
                 d: float = 20,          # Diameter (mm)
                 surface: str = 'usinee',  # Surface finish
                 R: float = 0.95,        # Reliability
                 T: float = 25,          # Temperature (°C)
                 Esalt: float = 0.2,     # Salt exposure factor
                 H: float = 60,          # Humidity (%)
                 Kt: float = 1.5,        # Stress concentration factor
                 q: float = 0.75):       # Notch sensitivity
        
        self.Sut = Sut
        self.d = d
        self.surface = surface
        self.R = R
        self.T = T
        self.Esalt = Esalt
        self.H = H
        self.Kt = Kt
        self.q = q
        self.D_cumule = 0.0  # Cumulative damage (Miner's rule)
        self.rdamage = 1e-3  # Damage rate per unit time
        self.m = 5           # Wöhler exponent
        
    def reset_damage(self):
        """Reset cumulative damage to zero."""
        self.D_cumule = 0.0
    
    # --- Correction Factors ---
    def surface_factor(self) -> float:
        """Surface finish correction factor (Ka)."""
        coeff = {
            'poli': (1.58, -0.085), 
            'usinee': (4.51, -0.265), 
            'laminee': (57.7, -0.718)
        }
        a, b = coeff.get(self.surface, (4.51, -0.265))
        return a * (self.Sut ** b)
    
    def size_factor(self) -> float:
        """Size correction factor (Kb)."""
        if self.d <= 8:
            return 1.0
        elif self.d <= 250:
            return 1.189 * (self.d ** -0.097)
        return 0.6
    
    def reliability_factor(self) -> float:
        """Reliability correction factor (Kc)."""
        z_table = {0.999: 0.753, 0.99: 0.814, 0.95: 0.868, 0.90: 0.897}
        return z_table.get(round(self.R, 3), 0.868)
    
    def dynamic_factor(self) -> float:
        """Dynamic stress concentration factor (Kf)."""
        return 1 / (self.q * (self.Kt - 1) + 1)
    
    def temperature_factor(self) -> float:
        """Temperature correction factor (Kd)."""
        if self.T <= 71:
            return 1.0
        else:
            return 344 / (self.T + 273)
    
    def corrosion_factor(self) -> float:
        """Corrosion/environment correction factor (Ke)."""
        return 1 / (1 + 0.3 * self.Esalt + 0.1 * self.H / 100)
    
    def corrected_fatigue_limit(self) -> float:
        """
        Calculate corrected endurance limit (Se').
        Se' = Se * Ka * Kb * Kc * Kd * Ke * Kf
        """
        Ks = self.surface_factor()
        Kg = self.size_factor()
        KF = self.reliability_factor()
        Kc = self.dynamic_factor()
        KT = self.temperature_factor()
        Kd = self.corrosion_factor()
        return 0.5 * self.Sut * Ks * Kg * KF * KT * Kd * Kc
    
    # --- Stress and Life Calculations ---
    def calculate_stress(self, F: float) -> float:
        """
        Calculate stress in the shock absorber rod.
        σ = (F / A) * Kt
        """
        A = np.pi * (self.d / 2) ** 2  # Cross-sectional area (mm²)
        return (F / A) * self.Kt
    
    def cycles_to_failure(self, sigma: float) -> float:
        """
        Calculate cycles to failure using Wöhler (S-N) curve.
        N = 10^3 * (0.9*Sut / σ)^(3 / log10(0.9*Sut / σD))
        """
        sigma_D = self.corrected_fatigue_limit()
        if sigma <= sigma_D:
            return float('inf')  # Below endurance limit, infinite life
        
        exponent = 3.0 / np.log10((0.9 * self.Sut) / sigma_D)
        N = (10**3) * ((0.9 * self.Sut) / sigma) ** exponent
        return max(N, 1)
    
    def update_damage(self, sigma: float, n_cycles: int = 1) -> float:
        """
        Update cumulative damage using Miner's rule.
        D = Σ(ni / Ni)
        """
        Nf = self.cycles_to_failure(sigma)
        if Nf == float('inf'):
            return self.D_cumule
        
        damage_increment = n_cycles / Nf
        self.D_cumule += damage_increment
        return self.D_cumule
    
    def calculate_damage_batch(self, stresses: List[float], cycles: List[int]) -> float:
        """
        Calculate cumulative damage for a batch of stress-cycles pairs.
        Miner's rule: D = Σ(ni / Ni)
        """
        total_damage = 0.0
        for sigma, n in zip(stresses, cycles):
            Nf = self.cycles_to_failure(sigma)
            if Nf != float('inf'):
                total_damage += n / Nf
        
        self.D_cumule = total_damage
        return total_damage
    
    def remaining_life_cycles(self) -> float:
        """
        Estimate remaining life based on current damage.
        Assumes failure at D = 1 (Miner's rule threshold).
        """
        D_failure = 1.0 - (1 - self.R) ** 2 / 2  # Adjusted for reliability
        remaining = max((D_failure - self.D_cumule) / self.rdamage, 0)
        return remaining
    
    def failure_probability(self) -> float:
        """
        Calculate probability of failure using sigmoid function.
        P = 1 / (1 + exp(-α(D - D50)))
        """
        alpha = 10  # Steepness of transition
        D50 = 0.8   # Damage at 50% failure probability
        return 1 / (1 + np.exp(-alpha * (self.D_cumule - D50)))
    
    def get_physics_features(self) -> Dict:
        """
        Get all physics-based features for ML input.
        """
        return {
            'D_cumule': self.D_cumule,
            'corrected_fatigue_limit': self.corrected_fatigue_limit(),
            'surface_factor': self.surface_factor(),
            'size_factor': self.size_factor(),
            'reliability_factor': self.reliability_factor(),
            'temperature_factor': self.temperature_factor(),
            'corrosion_factor': self.corrosion_factor(),
            'dynamic_factor': self.dynamic_factor(),
            'remaining_life_physics': self.remaining_life_cycles(),
            'failure_prob_physics': self.failure_probability()
        }


# =============================================================================
# PART 2: PHYSICS-INFORMED AI HYBRID MODEL
# =============================================================================

class PhysicsInformedMaintenanceAI:
    """
    Hybrid Physics-AI Model for Predictive Maintenance.
    
    Architecture:
    1. Physical model calculates baseline damage, life, and failure probability
    2. ML model learns correction factors to account for real-world effects
    3. Final prediction = Physics prediction × (1 + ML correction)
    """
    
    def __init__(self, component: str = "shock_absorber"):
        self.component = component
        self.physics_model = FatiguePhysicsModel()
        self.ml_models = {}
        self.scalers = {}
        self.encoders = {}
        self.training_history = {}
        
    def generate_synthetic_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """
        Generate synthetic training data combining physics and real-world effects.
        The physics model provides the baseline, and we add realistic noise/effects
        that the ML will learn to correct.
        """
        np.random.seed(42)
        data = []
        
        for i in range(n_samples):
            # === Usage Parameters ===
            mileage = np.random.uniform(0, 300000)  # km
            vehicle_age = np.random.uniform(0, 20)  # years
            average_speed = np.random.uniform(20, 120)  # km/h
            
            # === Environmental Conditions ===
            temperature = np.random.normal(25, 15)  # °C
            humidity = np.random.uniform(30, 90)  # %
            salt_exposure = np.random.uniform(0, 0.5)  # Salinity factor
            
            # === Road Type Distribution ===
            road_proportions = np.random.dirichlet([1, 2, 1.5, 0.5])
            dominant_road = np.argmax(road_proportions)
            
            # Stress factors by road type
            stress_factors = {0: 1.5, 1: 1.0, 2: 0.7, 3: 2.5}  # urban, road, highway, off-road
            road_stress = stress_factors[dominant_road]
            
            # === Driving Style ===
            aggressiveness = np.random.beta(2, 5)  # Most drivers are moderate
            
            # === Vehicle Parameters ===
            base_mass = np.random.uniform(1000, 2500)  # kg
            loading_factor = np.random.uniform(0.8, 1.5)
            effective_mass = base_mass * loading_factor
            
            # === Surface finish (manufacturing quality) ===
            surfaces = ['poli', 'usinee', 'laminee']
            surface_weights = [0.2, 0.6, 0.2]
            surface = np.random.choice(surfaces, p=surface_weights)
            
            # === Configure Physics Model ===
            self.physics_model = FatiguePhysicsModel(
                Sut=np.random.uniform(550, 750),  # Material variation
                d=np.random.uniform(15, 25),       # Size variation
                surface=surface,
                R=np.random.choice([0.90, 0.95, 0.99]),
                T=temperature,
                Esalt=salt_exposure,
                H=humidity,
                Kt=np.random.uniform(1.3, 1.8),
                q=np.random.uniform(0.65, 0.85)
            )
            
            # === Calculate Stress History ===
            # Simulate cumulative stress history based on mileage and usage
            # Base force of 15000N ensures stress exceeds fatigue limit for damage accumulation
            avg_force = 15000 * road_stress * (1 + aggressiveness) * (effective_mass / 1500)
            avg_stress = self.physics_model.calculate_stress(avg_force)
            
            # Number of cycles proportional to mileage
            # Using 10 stress cycles per km (more realistic for shock absorber fatigue)
            n_cycles_total = int(mileage * 10 * road_stress)
            
            # Update cumulative damage using Miner's rule
            if n_cycles_total > 0:
                self.physics_model.update_damage(avg_stress, n_cycles_total)
            
            # === Get Physics-Based Predictions ===
            physics_features = self.physics_model.get_physics_features()
            
            # === Add Real-World Effects (not captured by simple physics) ===
            # These are effects the ML will learn to correct
            
            # Manufacturing variability
            manufacturing_noise = np.random.normal(1.0, 0.1)
            
            # Road quality uncertainty
            road_quality_factor = np.random.lognormal(0, 0.2)
            
            # Maintenance history effect
            previous_repairs = np.random.poisson(mileage / 50000)
            repair_quality_factor = 1.0 - 0.05 * previous_repairs
            
            # Driver behavior variability
            driver_factor = 1.0 + 0.3 * aggressiveness
            
            # === Ground Truth (with real-world effects) ===
            # Physics prediction adjusted by real-world factors
            physics_remaining_life = physics_features['remaining_life_physics']
            physics_correction = manufacturing_noise * road_quality_factor * repair_quality_factor / driver_factor
            
            # True remaining life (what we want to predict)
            true_remaining_life = max(0, physics_remaining_life * physics_correction * 150 + np.random.normal(0, 5000))
            
            # True failure probability adjusted
            physics_fail_prob = physics_features['failure_prob_physics']
            true_fail_prob = np.clip(physics_fail_prob / physics_correction + np.random.normal(0, 0.05), 0, 1)
            
            # State classification
            if true_remaining_life < 10000 or self.physics_model.D_cumule > 0.9:
                state = 'critical'
            elif true_remaining_life < 30000 or self.physics_model.D_cumule > 0.7:
                state = 'warning'
            else:
                state = 'normal'
            
            # === Compile Sample ===
            sample = {
                # Usage features
                'mileage': mileage,
                'vehicle_age': vehicle_age,
                'average_speed': average_speed,
                'proportion_urban': road_proportions[0],
                'proportion_road': road_proportions[1],
                'proportion_highway': road_proportions[2],
                'proportion_offroad': road_proportions[3],
                'aggressiveness': aggressiveness,
                
                # Environmental features
                'temperature': temperature,
                'humidity': humidity,
                'salt_exposure': salt_exposure,
                
                # Vehicle features
                'vehicle_mass': base_mass,
                'loading_factor': loading_factor,
                'previous_repairs': previous_repairs,
                
                # Physics model parameters
                'Sut': self.physics_model.Sut,
                'diameter': self.physics_model.d,
                'reliability': self.physics_model.R,
                'Kt': self.physics_model.Kt,
                
                # Physics-computed features (key hybrid inputs!)
                'D_cumule': physics_features['D_cumule'],
                'avg_stress': avg_stress,
                'n_cycles_total': n_cycles_total,
                'corrected_fatigue_limit': physics_features['corrected_fatigue_limit'],
                'surface_factor': physics_features['surface_factor'],
                'size_factor': physics_features['size_factor'],
                'temperature_factor': physics_features['temperature_factor'],
                'corrosion_factor': physics_features['corrosion_factor'],
                'remaining_life_physics': physics_remaining_life,
                'failure_prob_physics': physics_fail_prob,
                
                # Ground truth targets
                'remaining_life_true': true_remaining_life,
                'failure_prob_true': true_fail_prob,
                'state': state,
                
                # Correction factor (what ML learns)
                'correction_factor': physics_correction
            }
            
            data.append(sample)
            self.physics_model.reset_damage()
        
        return pd.DataFrame(data)
    
    def preprocess_data(self, df: pd.DataFrame, training: bool = True) -> Tuple:
        """
        Preprocess data for training/prediction.
        Returns features (including physics outputs) and targets.
        """
        # Feature columns (usage + environment + vehicle + physics outputs)
        feature_cols = [
            'mileage', 'vehicle_age', 'average_speed',
            'proportion_urban', 'proportion_road', 'proportion_highway', 'proportion_offroad',
            'aggressiveness', 'temperature', 'humidity', 'salt_exposure',
            'vehicle_mass', 'loading_factor', 'previous_repairs',
            'Sut', 'diameter', 'reliability', 'Kt',
            # Physics features (key hybrid inputs!)
            'D_cumule', 'avg_stress', 'n_cycles_total',
            'corrected_fatigue_limit', 'surface_factor', 'size_factor',
            'temperature_factor', 'corrosion_factor',
            'remaining_life_physics', 'failure_prob_physics'
        ]
        
        X = df[feature_cols].values
        
        # Normalize features
        if training:
            self.scalers['features'] = StandardScaler()
            X_scaled = self.scalers['features'].fit_transform(X)
        else:
            X_scaled = self.scalers['features'].transform(X)
        
        # Targets
        y_life = df['remaining_life_true'].values if 'remaining_life_true' in df.columns else None
        y_correction = df['correction_factor'].values if 'correction_factor' in df.columns else None
        
        if 'state' in df.columns:
            if training:
                self.encoders['state'] = LabelEncoder()
                y_state = self.encoders['state'].fit_transform(df['state'])
            else:
                y_state = self.encoders['state'].transform(df['state'])
        else:
            y_state = None
        
        return X_scaled, y_life, y_correction, y_state, df['remaining_life_physics'].values
    
    def train(self, df: pd.DataFrame):
        """
        Train the hybrid Physics-AI model.
        
        ML learns to predict:
        1. Correction factor for physics prediction
        2. State classification
        """
        print("=" * 60)
        print("PHYSICS-INFORMED AI HYBRID MODEL - TRAINING")
        print("=" * 60)
        
        # Preprocess
        X, y_life, y_correction, y_state, y_physics = self.preprocess_data(df, training=True)
        
        # Split data
        (X_train, X_test, 
         y_life_train, y_life_test,
         y_corr_train, y_corr_test,
         y_state_train, y_state_test,
         y_phys_train, y_phys_test) = train_test_split(
            X, y_life, y_correction, y_state, y_physics, 
            test_size=0.2, random_state=42
        )
        
        # === 1. Train Correction Factor Model ===
        print("\n[1] Training ML Correction Factor Model...")
        
        self.ml_models['correction'] = RandomForestRegressor(
            n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
        )
        self.ml_models['correction'].fit(X_train, y_corr_train)
        
        # Evaluate
        corr_pred = self.ml_models['correction'].predict(X_test)
        corr_r2 = r2_score(y_corr_test, corr_pred)
        print(f"    Correction Factor R² Score: {corr_r2:.4f}")
        
        # === 2. Train Direct Life Prediction (for comparison) ===
        print("\n[2] Training Direct Life Prediction Model...")
        
        self.ml_models['life_direct'] = RandomForestRegressor(
            n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
        )
        self.ml_models['life_direct'].fit(X_train, y_life_train)
        
        life_pred_direct = self.ml_models['life_direct'].predict(X_test)
        direct_mae = mean_absolute_error(y_life_test, life_pred_direct)
        direct_r2 = r2_score(y_life_test, life_pred_direct)
        print(f"    Direct ML - MAE: {direct_mae:.2f} km, R²: {direct_r2:.4f}")
        
        # === 3. Hybrid Prediction (Physics × Correction) ===
        print("\n[3] Evaluating Hybrid Model (Physics × ML Correction)...")
        
        hybrid_pred = y_phys_test * corr_pred * 150  # Apply learned correction
        hybrid_mae = mean_absolute_error(y_life_test, hybrid_pred)
        hybrid_r2 = r2_score(y_life_test, hybrid_pred)
        print(f"    Hybrid - MAE: {hybrid_mae:.2f} km, R²: {hybrid_r2:.4f}")
        
        # === 4. Physics-only baseline ===
        print("\n[4] Physics-Only Baseline...")
        physics_life_pred = y_phys_test * 150
        physics_mae = mean_absolute_error(y_life_test, physics_life_pred)
        physics_r2 = r2_score(y_life_test, physics_life_pred)
        print(f"    Physics-Only - MAE: {physics_mae:.2f} km, R²: {physics_r2:.4f}")
        
        # === 5. Train Classification Model ===
        print("\n[5] Training State Classification Model...")
        
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, 20],
        }
        
        rf_clf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf_clf, param_grid, cv=3, scoring='f1_weighted')
        grid_search.fit(X_train, y_state_train)
        
        self.ml_models['classification'] = grid_search.best_estimator_
        
        y_state_pred = self.ml_models['classification'].predict(X_test)
        print(f"    Best params: {grid_search.best_params_}")
        print("\n    Classification Report:")
        print(classification_report(y_state_test, y_state_pred,
                                   target_names=self.encoders['state'].classes_))
        
        # === Summary ===
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY - MODEL COMPARISON")
        print("=" * 60)
        print(f"{'Model':<25} {'MAE (km)':<15} {'R² Score':<15}")
        print("-" * 55)
        print(f"{'Physics-Only':<25} {physics_mae:<15.2f} {physics_r2:<15.4f}")
        print(f"{'Direct ML':<25} {direct_mae:<15.2f} {direct_r2:<15.4f}")
        print(f"{'Hybrid (Physics+ML)':<25} {hybrid_mae:<15.2f} {hybrid_r2:<15.4f}")
        print("=" * 60)
        
        # Store metrics
        self.training_history = {
            'physics_mae': physics_mae,
            'physics_r2': physics_r2,
            'direct_ml_mae': direct_mae,
            'direct_ml_r2': direct_r2,
            'hybrid_mae': hybrid_mae,
            'hybrid_r2': hybrid_r2,
            'classification_score': grid_search.best_score_
        }
    
    def predict(self, vehicle_data: Dict) -> Dict:
        """
        Make prediction using the hybrid Physics-AI model.
        """
        # Configure physics model with provided parameters
        self.physics_model = FatiguePhysicsModel(
            Sut=vehicle_data.get('Sut', 600),
            d=vehicle_data.get('diameter', 20),
            surface=vehicle_data.get('surface', 'usinee'),
            R=vehicle_data.get('reliability', 0.95),
            T=vehicle_data.get('temperature', 25),
            Esalt=vehicle_data.get('salt_exposure', 0.2),
            H=vehicle_data.get('humidity', 60),
            Kt=vehicle_data.get('Kt', 1.5),
            q=vehicle_data.get('q', 0.75)
        )
        
        # Calculate physics-based features
        mileage = vehicle_data.get('mileage', 0)
        
        # Calculate road stress factor properly (matching training data)
        prop_urban = vehicle_data.get('proportion_urban', 0.25)
        prop_road = vehicle_data.get('proportion_road', 0.50)
        prop_highway = vehicle_data.get('proportion_highway', 0.20)
        prop_offroad = vehicle_data.get('proportion_offroad', 0.05)
        
        # Weighted average of stress factors: urban=1.5, road=1.0, highway=0.7, offroad=2.5
        road_stress = (prop_urban * 1.5 + prop_road * 1.0 + 
                       prop_highway * 0.7 + prop_offroad * 2.5)
        
        aggressiveness = vehicle_data.get('aggressiveness', 0.3)
        mass = vehicle_data.get('vehicle_mass', 1500)
        loading = vehicle_data.get('loading_factor', 1.0)
        effective_mass = mass * loading
        
        # Force calculation matching training data generation
        # Base force of 15000N ensures stress exceeds fatigue limit
        avg_force = 15000 * road_stress * (1 + aggressiveness) * (effective_mass / 1500)
        avg_stress = self.physics_model.calculate_stress(avg_force)
        
        # Number of cycles proportional to mileage (matching training: 10 cycles per km)
        n_cycles = int(mileage * 10 * road_stress)
        
        # Update cumulative damage using Miner's rule
        if n_cycles > 0 and avg_stress > 0:
            self.physics_model.update_damage(avg_stress, n_cycles)
        
        physics_features = self.physics_model.get_physics_features()
        
        # Prepare input for ML
        input_data = {
            'mileage': mileage,
            'vehicle_age': vehicle_data.get('vehicle_age', 5),
            'average_speed': vehicle_data.get('average_speed', 60),
            'proportion_urban': vehicle_data.get('proportion_urban', 0.25),
            'proportion_road': vehicle_data.get('proportion_road', 0.50),
            'proportion_highway': vehicle_data.get('proportion_highway', 0.20),
            'proportion_offroad': vehicle_data.get('proportion_offroad', 0.05),
            'aggressiveness': aggressiveness,
            'temperature': vehicle_data.get('temperature', 25),
            'humidity': vehicle_data.get('humidity', 60),
            'salt_exposure': vehicle_data.get('salt_exposure', 0.2),
            'vehicle_mass': mass,
            'loading_factor': loading,
            'previous_repairs': vehicle_data.get('previous_repairs', 0),
            'Sut': self.physics_model.Sut,
            'diameter': self.physics_model.d,
            'reliability': self.physics_model.R,
            'Kt': self.physics_model.Kt,
            'D_cumule': physics_features['D_cumule'],
            'avg_stress': avg_stress,
            'n_cycles_total': n_cycles,
            'corrected_fatigue_limit': physics_features['corrected_fatigue_limit'],
            'surface_factor': physics_features['surface_factor'],
            'size_factor': physics_features['size_factor'],
            'temperature_factor': physics_features['temperature_factor'],
            'corrosion_factor': physics_features['corrosion_factor'],
            'remaining_life_physics': physics_features['remaining_life_physics'],
            'failure_prob_physics': physics_features['failure_prob_physics']
        }
        
        df_input = pd.DataFrame([input_data])
        X = self.scalers['features'].transform(df_input.values)
        
        # Get predictions
        correction_factor = self.ml_models['correction'].predict(X)[0]
        life_direct = self.ml_models['life_direct'].predict(X)[0]
        state_probs = self.ml_models['classification'].predict_proba(X)[0]
        state_pred = self.ml_models['classification'].predict(X)[0]
        
        # Hybrid prediction
        life_physics = physics_features['remaining_life_physics'] * 150
        life_hybrid = life_physics * correction_factor
        
        # State name
        state_name = self.encoders['state'].inverse_transform([state_pred])[0]
        
        # Urgency based on hybrid prediction
        if life_hybrid < 10000 or physics_features['D_cumule'] > 0.9:
            urgency = "CRITICAL"
            recommendation = "Immediate replacement required"
        elif life_hybrid < 30000 or physics_features['D_cumule'] > 0.7:
            urgency = "WARNING"
            recommendation = "Schedule replacement within 3 months"
        else:
            urgency = "NORMAL"
            recommendation = "Continue standard maintenance"
        
        return {
            # Physics outputs
            'D_cumule': physics_features['D_cumule'],
            'remaining_life_physics': life_physics,
            'failure_prob_physics': physics_features['failure_prob_physics'],
            
            # ML outputs
            'correction_factor': correction_factor,
            'remaining_life_ml_direct': life_direct,
            
            # Hybrid output (FINAL PREDICTION)
            'remaining_life_hybrid': max(0, life_hybrid),
            'predicted_state': state_name,
            'state_probabilities': {
                cls: prob for cls, prob in 
                zip(self.encoders['state'].classes_, state_probs)
            },
            
            # Recommendations
            'urgency': urgency,
            'recommendation': recommendation,
            'prediction_reliability': float(max(state_probs))
        }
    
    def analyze_feature_importance(self, n_top: int = 10) -> Dict:
        """Analyze which features are most important for predictions."""
        feature_names = [
            'mileage', 'vehicle_age', 'average_speed',
            'proportion_urban', 'proportion_road', 'proportion_highway', 'proportion_offroad',
            'aggressiveness', 'temperature', 'humidity', 'salt_exposure',
            'vehicle_mass', 'loading_factor', 'previous_repairs',
            'Sut', 'diameter', 'reliability', 'Kt',
            'D_cumule', 'avg_stress', 'n_cycles_total',
            'corrected_fatigue_limit', 'surface_factor', 'size_factor',
            'temperature_factor', 'corrosion_factor',
            'remaining_life_physics', 'failure_prob_physics'
        ]
        
        # Correction model importance
        corr_imp = self.ml_models['correction'].feature_importances_
        corr_features = sorted(zip(feature_names, corr_imp), key=lambda x: x[1], reverse=True)
        
        # Classification importance
        clf_imp = self.ml_models['classification'].feature_importances_
        clf_features = sorted(zip(feature_names, clf_imp), key=lambda x: x[1], reverse=True)
        
        # Identify physics vs usage features
        physics_features = ['D_cumule', 'avg_stress', 'n_cycles_total', 'corrected_fatigue_limit',
                           'surface_factor', 'size_factor', 'temperature_factor', 'corrosion_factor',
                           'remaining_life_physics', 'failure_prob_physics']
        
        physics_importance_corr = sum(imp for name, imp in corr_features if name in physics_features)
        usage_importance_corr = sum(imp for name, imp in corr_features if name not in physics_features)
        
        return {
            'correction_top_features': corr_features[:n_top],
            'classification_top_features': clf_features[:n_top],
            'physics_features_importance': physics_importance_corr,
            'usage_features_importance': usage_importance_corr,
            'physics_contribution_pct': physics_importance_corr * 100
        }
    
    def save(self, path: str = "hybrid_model"):
        """Save the trained model."""
        if not os.path.exists(path):
            os.makedirs(path)
        
        joblib.dump(self.ml_models, f"{path}/ml_models.pkl")
        joblib.dump(self.scalers, f"{path}/scalers.pkl")
        joblib.dump(self.encoders, f"{path}/encoders.pkl")
        joblib.dump(self.training_history, f"{path}/history.pkl")
        print(f"Model saved to {path}/")
    
    def load(self, path: str = "hybrid_model"):
        """Load a pre-trained model."""
        self.ml_models = joblib.load(f"{path}/ml_models.pkl")
        self.scalers = joblib.load(f"{path}/scalers.pkl")
        self.encoders = joblib.load(f"{path}/encoders.pkl")
        self.training_history = joblib.load(f"{path}/history.pkl")
        print(f"Model loaded from {path}/")


# =============================================================================
# MAIN - EXAMPLE USAGE
# =============================================================================

def main():
    """Demonstration of the Physics-Informed AI Hybrid Model."""
    
    print("\n" + "=" * 70)
    print("  PHYSICS-INFORMED AI HYBRID MODEL FOR SHOCK ABSORBER MAINTENANCE")
    print("=" * 70 + "\n")
    
    # Initialize hybrid model
    hybrid_model = PhysicsInformedMaintenanceAI("shock_absorber")
    
    # Generate training data
    print("[Step 1] Generating synthetic training data...")
    data = hybrid_model.generate_synthetic_data(5000)
    print(f"         Generated {len(data)} samples\n")
    
    # Train the model
    print("[Step 2] Training hybrid model...")
    hybrid_model.train(data)
    
    # Test prediction
    print("\n[Step 3] Testing on example vehicle...")
    test_vehicle = {
        'mileage': 120000,
        'vehicle_age': 6,
        'average_speed': 70,
        'proportion_urban': 0.3,
        'proportion_road': 0.4,
        'proportion_highway': 0.2,
        'proportion_offroad': 0.1,
        'aggressiveness': 0.4,
        'temperature': 30,
        'humidity': 70,
        'salt_exposure': 0.3,
        'vehicle_mass': 1600,
        'loading_factor': 1.2,
        'previous_repairs': 1,
        'Sut': 650,
        'diameter': 18,
        'reliability': 0.95,
        'Kt': 1.5
    }
    
    result = hybrid_model.predict(test_vehicle)
    
    print("\n" + "-" * 50)
    print("PREDICTION RESULTS")
    print("-" * 50)
    print(f"Physics-only remaining life: {result['remaining_life_physics']:,.0f} km")
    print(f"ML correction factor:        {result['correction_factor']:.3f}")
    print(f"HYBRID remaining life:       {result['remaining_life_hybrid']:,.0f} km")
    print(f"Cumulative damage (D):       {result['D_cumule']:.4f}")
    print(f"Predicted state:             {result['predicted_state'].upper()}")
    print(f"Urgency:                     {result['urgency']}")
    print(f"Recommendation:              {result['recommendation']}")
    
    # Feature importance
    print("\n[Step 4] Analyzing feature importance...")
    importance = hybrid_model.analyze_feature_importance(5)
    print(f"\nPhysics features contribute {importance['physics_contribution_pct']:.1f}% to correction model")
    print("\nTop 5 features for correction factor:")
    for name, imp in importance['correction_top_features']:
        print(f"  - {name}: {imp:.3f}")
    
    # Save model
    print("\n[Step 5] Saving model...")
    hybrid_model.save("hybrid_model")
    
    print("\n" + "=" * 70)
    print("  DEMONSTRATION COMPLETE")
    print("=" * 70 + "\n")
    
    return hybrid_model


if __name__ == "__main__":
    model = main()
