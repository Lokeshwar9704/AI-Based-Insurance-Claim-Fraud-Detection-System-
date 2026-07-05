# Models Directory

Generated files appear here after training:

- `fraud_model_v3.pkl`     — Trained RF+ET+GB ensemble (serialised with joblib)
- `encoders_v3.pkl`        — Label encoders for categorical columns  
- `feature_names_v3.pkl`   — Ordered list of feature names
- `model_config_v3.json`   — Training metrics and configuration

Run `ml/train_model.py` to generate these files.
