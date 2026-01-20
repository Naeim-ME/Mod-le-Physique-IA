# Shock Absorber Predictive Maintenance - Physics-AI Hybrid

An AI system for predictive maintenance of automotive shock absorbers that combines **physical fatigue models** with **machine learning** to predict remaining useful life and failure probability.

## 🚀 Two Model Approaches

### 1. Physics-AI Hybrid Model (Recommended)
**File:** `physics_ai_hybrid_model.py`

Combines classical fatigue theory with ML:
- **Wöhler S-N Curve** - cycles to failure calculation
- **Miner's Rule** - cumulative damage accumulation
- **ML Correction Factors** - learns real-world effects not captured by physics

```python
from physics_ai_hybrid_model import PhysicsInformedMaintenanceAI

model = PhysicsInformedMaintenanceAI()
data = model.generate_synthetic_data(5000)
model.train(data)

result = model.predict({
    'mileage': 120000,
    'vehicle_age': 6,
    'temperature': 30,
    'vehicle_mass': 1600,
    'aggressiveness': 0.4,
    # ... other parameters
})

print(f"Hybrid remaining life: {result['remaining_life_hybrid']:,.0f} km")
print(f"Cumulative damage (D): {result['D_cumule']:.4f}")
print(f"Urgency: {result['urgency']}")
```

### 2. Pure ML Model
**File:** `shock_absorber_predictive_maintenance.py`

Data-driven approach using Random Forest, Gradient Boosting, and MLP.

## Installation

```bash
pip install -r requirements.txt
```

## Architecture (Hybrid Model)

```
Sensor/Usage Data ─┬─► Physical Model ─► Physics Features ─┬─► ML Correction ─► Final Prediction
                   └────────────────────────────────────────┘
```

**Hybrid Formula:**
```
Final Prediction = Physics Prediction × ML Correction Factor
```

## Key Features

| Physics Model | ML Model |
|---------------|----------|
| Wöhler S-N curve | RandomForestRegressor |
| Miner's cumulative damage | RandomForestClassifier |
| Surface factor (Ka) | Feature importance analysis |
| Size factor (Kb) | Correction factor learning |
| Temperature/Corrosion factors | State classification |

## Output Predictions

- **Remaining lifespan** (km) - both physics-only and hybrid
- **Cumulative damage (D)** - Miner's rule value
- **Component state** (normal / warning / critical)
- **Urgency level** and recommendations
- **ML correction factor** - how much ML adjusts the physics prediction

## Model Persistence

```python
# Save
model.save("hybrid_model")

# Load
model.load("hybrid_model")
```

## License

MIT License

## Author

Naim AAZIZ
