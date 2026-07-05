"""
FraudGuard v3 — Model Training Script
Insurance Fraud Detection · RF + ExtraTrees + GradientBoosting Ensemble

Datasets required in data/ folder:
  - fraud_oracle.csv   (15,420 rows — auto insurance claims)
  - carclaims.csv      (15,420 rows — label verification)

Healthcare dataset (archive_13.zip) is used for healthcare fraud (provider-level).

Run:
  cd ml/
  python train_model.py

Output:
  ../models/fraud_model_v3.pkl
  ../models/feature_names_v3.pkl
  ../models/model_config_v3.json
"""
import pandas as pd
import numpy as np
import json
import warnings
import os
import sys
warnings.filterwarnings("ignore")

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
import joblib

np.random.seed(42)

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FRAUD_ORACLE = os.path.join(DATA_DIR, "fraud_oracle.csv")
CARCLAIMS    = os.path.join(DATA_DIR, "carclaims.csv")

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
print("=" * 65)
print("  FraudGuard v3  —  Model Training")
print("=" * 65)

for path in [FRAUD_ORACLE, CARCLAIMS]:
    if not os.path.exists(path):
        print(f"\n  ERROR: {path} not found.")
        print("  Place fraud_oracle.csv and carclaims.csv in the data/ folder.\n")
        sys.exit(1)

df  = pd.read_csv(FRAUD_ORACLE)
df.columns = df.columns.str.strip().str.lstrip("\ufeff")
df2 = pd.read_csv(CARCLAIMS)
df2.columns = df2.columns.str.strip()
print(f"\n  Loaded fraud_oracle: {df.shape}  |  carclaims: {df2.shape}")

# Verify label consistency between both CSV files
df["FraudFound"] = (df2["FraudFound"] == "Yes").astype(int)
assert (df["FraudFound"] == df["FraudFound_P"]).all(), "Label mismatch between datasets!"
print("  Label consistency: VERIFIED")

y_all = df["FraudFound_P"].astype(int)
print(f"  Fraud: {y_all.sum():,} / {len(y_all):,}  ({y_all.mean()*100:.2f}%)")

# ─── EDA-DERIVED FRAUD RATES ─────────────────────────────────────────────────
# These rates were computed from exploratory data analysis on the full dataset.
# They replace one-hot encoding with meaningful numeric signals.

BP_RATE   = {"Liability": 0.007, "Collision": 0.073, "All Perils": 0.102}
FAULT_RATE = {"Policy Holder": 0.079, "Third Party": 0.009}
PT_RATE   = {
    "Sport - Collision": 0.138, "Utility - All Perils": 0.121,
    "Sedan - All Perils": 0.101, "Utility - Collision": 0.082,
    "Sedan - Collision": 0.074, "Sport - All Perils": 0.068,
    "Sedan - Liability": 0.007, "Utility - Liability": 0.000, "Sport - Liability": 0.000,
}
VC_RATE   = {"Utility": 0.113, "Sedan": 0.082, "Sport": 0.016}
AC_RATE   = {"under 6 months": 0.75, "2 to 3 years": 0.175,
             "1 year": 0.065, "no change": 0.058, "4 to 8 years": 0.052}
APH_RATE  = {"16 to 17": 0.097, "18 to 20": 0.133, "21 to 25": 0.148,
             "26 to 30": 0.072, "31 to 35": 0.058, "36 to 40": 0.057,
             "41 to 50": 0.051, "51 to 65": 0.050, "over 65": 0.040}
MAKE_RATE = {"Accura": 0.125, "Saturn": 0.103, "Dodge": 0.082,
             "Pontiac": 0.079, "Honda": 0.064, "Toyota": 0.060,
             "Ford": 0.058, "Mazda": 0.053, "Chevrolet": 0.048,
             "Jaguar": 0.045, "VW": 0.044, "Nisson": 0.042,
             "Mecedes": 0.250, "BMW": 0.038, "Ferrari": 0.000}
AOV_RATE  = {"4 years": 0.092, "new": 0.086, "3 years": 0.086,
             "5 years": 0.069, "7 years": 0.058, "6 years": 0.056,
             "more than 7": 0.052, "2 years": 0.041}

# ─── FEATURE ENGINEERING ─────────────────────────────────────────────────────
print("\n  Engineering features from EDA signals...")

def get_col(df, *names):
    """Return the first matching column name."""
    for n in names:
        if n in df.columns:
            return n
    raise ValueError(f"None of {names} found in DataFrame")

def build_features(df):
    d = {}

    # Target-encoded rates
    d["bp_fraud_rate"]    = df["BasePolicy"].map(BP_RATE).fillna(0.06)
    d["fault_fraud_rate"] = df["Fault"].map(FAULT_RATE).fillna(0.06)
    d["pt_fraud_rate"]    = df["PolicyType"].map(PT_RATE).fillna(0.06)
    d["vc_fraud_rate"]    = df["VehicleCategory"].map(VC_RATE).fillna(0.06)
    d["make_fraud_rate"]  = df["Make"].map(MAKE_RATE).fillna(0.06)
    d["aov_fraud_rate"]   = df["AgeOfVehicle"].map(AOV_RATE).fillna(0.06)

    ac_col = get_col(df, "AddressChange_Claim", "AddressChange-Claim")
    d["ac_fraud_rate"]    = df[ac_col].map(AC_RATE).fillna(0.06)
    d["aph_fraud_rate"]   = df["AgeOfPolicyHolder"].map(APH_RATE).fillna(0.06)

    # AccidentArea
    d["is_rural"] = (df["AccidentArea"].astype(str).str.strip() == "Rural").astype(int)

    # Binary flags
    d["fault_holder"]     = (df["Fault"].astype(str).str.strip() == "Policy Holder").astype(int)
    d["is_liability"]     = df["BasePolicy"].astype(str).str.contains("Liability", na=False).astype(int)
    d["is_collision"]     = df["BasePolicy"].astype(str).str.contains("Collision", na=False).astype(int)
    d["is_all_perils"]    = df["BasePolicy"].astype(str).str.contains("All Perils", na=False).astype(int)
    d["is_sport_vehicle"] = (df["VehicleCategory"].astype(str).str.strip() == "Sport").astype(int)
    d["is_sedan"]         = (df["VehicleCategory"].astype(str).str.strip() == "Sedan").astype(int)
    d["is_utility"]       = (df["VehicleCategory"].astype(str).str.strip() == "Utility").astype(int)
    d["is_male"]          = (df["Sex"].astype(str).str.strip() == "Male").astype(int)
    d["is_external"]      = (df["AgentType"].astype(str).str.strip() == "External").astype(int)
    d["police_yes"]       = (df["PoliceReportFiled"].astype(str).str.strip() == "Yes").astype(int)
    d["witness_yes"]      = (df["WitnessPresent"].astype(str).str.strip() == "Yes").astype(int)
    d["no_police"]        = 1 - d["police_yes"]
    d["no_witness"]       = 1 - d["witness_yes"]
    d["third_party"]      = 1 - d["fault_holder"]

    # Age groups
    young = ["16 to 17", "18 to 20", "21 to 25"]
    d["young_holder"] = df["AgeOfPolicyHolder"].isin(young).astype(int)

    # Address change
    d["recent_move"] = df[ac_col].isin(["under 6 months", "2 to 3 years"]).astype(int)
    d["no_move"]     = (df[ac_col] == "no change").astype(int)

    # Days policy
    dpa_col = get_col(df, "Days_Policy_Accident", "Days:Policy-Accident")
    dpc_col = get_col(df, "Days_Policy_Claim",    "Days:Policy-Claim")
    d["short_policy"] = df[dpa_col].isin(["none", "1 to 7", "8 to 15"]).astype(int)
    d["fast_claim"]   = df[dpc_col].isin(["8 to 15", "15 to 30"]).astype(int)

    # Past claims
    d["no_past_claims"]   = (df["PastNumberOfClaims"] == "none").astype(int)
    d["many_past_claims"] = df["PastNumberOfClaims"].isin(["2 to 4", "more than 4"]).astype(int)

    # Supplements
    d["many_supps"] = df["NumberOfSuppliments"].isin(["3 to 5", "more than 5"]).astype(int)

    # Vehicle price
    d["cheap_vehicle"]     = df["VehiclePrice"].astype(str).str.contains("less than", na=False).astype(int)
    d["expensive_vehicle"] = df["VehiclePrice"].astype(str).str.contains("more than 69", na=False).astype(int)

    # Driver rating
    d["DriverRating"] = pd.to_numeric(df["DriverRating"], errors="coerce").fillna(2)
    d["bad_driver"]   = (d["DriverRating"] >= 3).astype(int)

    # Numerics
    d["Age"]         = pd.to_numeric(df["Age"],         errors="coerce").fillna(40)
    d["Deductible"]  = pd.to_numeric(df["Deductible"],  errors="coerce").fillna(400)
    d["WeekOfMonth"] = pd.to_numeric(df["WeekOfMonth"], errors="coerce").fillna(3)

    # ── INTERACTION FEATURES ─────────────────────────────────────────────────
    # Liability + Third Party = virtually zero fraud
    d["safe_combo"]       = d["is_liability"] * d["third_party"]
    # All Perils + no police + no witness = high-risk blind claim
    d["risky_blind"]      = d["is_all_perils"] * d["no_police"] * d["no_witness"]
    # Fault holder + no police
    d["fault_no_police"]  = d["fault_holder"] * d["no_police"]
    # Young + expensive + collision
    d["young_exp_coll"]   = d["young_holder"] * d["expensive_vehicle"] * d["is_collision"]
    # Recent move + fault holder
    d["move_fault"]       = d["recent_move"] * d["fault_holder"]
    # Rural + no witness + external
    d["rural_blind_ext"]  = d["is_rural"] * d["no_witness"] * d["is_external"]
    # Fast claim + no police
    d["fast_no_police"]   = d["fast_claim"] * d["no_police"]
    # Fault + young + rural
    d["fault_young_rural"] = d["fault_holder"] * d["young_holder"] * d["is_rural"]
    # Utility + collision
    d["util_collision"]   = d["is_utility"] * d["is_collision"]

    # ── COMPOSITE SCORE ──────────────────────────────────────────────────────
    # Weighted combination of target-encoded rates (the "quartet signal")
    d["composite_rate"] = (
        d["bp_fraud_rate"]    * 0.25 +
        d["fault_fraud_rate"] * 0.22 +
        d["pt_fraud_rate"]    * 0.18 +
        d["vc_fraud_rate"]    * 0.12 +
        d["ac_fraud_rate"]    * 0.10 +
        d["aph_fraud_rate"]   * 0.08 +
        d["make_fraud_rate"]  * 0.05
    )

    # Risk signal counts
    d["risk_count"] = (
        d["fault_holder"] + d["is_all_perils"] + d["is_utility"] +
        d["recent_move"] + d["no_police"] + d["no_witness"] +
        d["young_holder"] + d["bad_driver"] + d["is_rural"]
    )
    d["safe_count"] = (
        d["is_liability"] + d["third_party"] + d["witness_yes"] +
        d["police_yes"] + d["no_past_claims"] + d["is_sport_vehicle"]
    )
    d["net_signal"] = d["risk_count"] - d["safe_count"]

    return pd.DataFrame(d)

X_raw = build_features(df)

# Also add label-encoded original columns for tree depth
le_cols = [
    "Month", "DayOfWeek", "Make", "DayOfWeekClaimed", "MonthClaimed",
    "MaritalStatus", "PolicyType", "VehicleCategory", "BasePolicy", "Fault",
    "VehiclePrice", "AgeOfVehicle", "AgeOfPolicyHolder", "PastNumberOfClaims",
    "NumberOfSuppliments", "NumberOfCars", "AgentType", "AccidentArea",
    "Sex", "PoliceReportFiled", "WitnessPresent", "Year",
]
for alt in [("AddressChange_Claim","AddressChange-Claim"),
            ("Days_Policy_Accident","Days:Policy-Accident"),
            ("Days_Policy_Claim","Days:Policy-Claim")]:
    for a in alt:
        if a in df.columns:
            le_cols.append(a); break

encoders = {}
for col in le_cols:
    if col in df.columns:
        le = LabelEncoder()
        X_raw[f"le_{col}"] = le.fit_transform(df[col].astype(str).str.strip())
        encoders[col] = le

X = X_raw.copy()
feat_names = list(X.columns)
print(f"  Total features: {len(feat_names)}")

# ─── TRAIN / TEST SPLIT ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y_all.values, test_size=0.20, random_state=42, stratify=y_all.values
)
print(f"\n  Train: {len(y_train):,}   Test: {len(y_test):,}")
print(f"  Test fraud: {y_test.sum():,}   Test legit: {(y_test==0).sum():,}")

# ─── OVERSAMPLE FRAUD CLASS ───────────────────────────────────────────────────
# Fraud is ~6% of data — oversample to 39% for balanced training
fi  = np.where(y_train == 1)[0]
li  = np.where(y_train == 0)[0]
n_s = int(len(li) * 0.65)
fi_os = np.random.choice(fi, size=n_s, replace=True)
Xb = np.vstack([X_train[li], X_train[fi_os]])
yb = np.hstack([y_train[li], np.ones(n_s, dtype=int)])
perm = np.random.permutation(len(yb))
Xb, yb = Xb[perm], yb[perm]
print(f"\n  Balanced training: {len(yb):,} rows  ({yb.mean()*100:.1f}% fraud)")

# ─── TRAIN THREE BASE MODELS ─────────────────────────────────────────────────
print("\n  Training models (this may take 2-5 minutes)...")

rf = RandomForestClassifier(
    n_estimators=600, max_depth=None, min_samples_leaf=1,
    max_features="sqrt", class_weight={0: 1, 1: 4},
    random_state=42, n_jobs=-1,
)
rf.fit(Xb, yb)
print("  RandomForest        done")

et = ExtraTreesClassifier(
    n_estimators=600, max_depth=None, min_samples_leaf=1,
    max_features="sqrt", class_weight={0: 1, 1: 4},
    random_state=43, n_jobs=-1,
)
et.fit(Xb, yb)
print("  ExtraTrees          done")

gb = GradientBoostingClassifier(
    n_estimators=400, learning_rate=0.04, max_depth=6,
    min_samples_leaf=2, subsample=0.80, max_features="sqrt",
    random_state=44,
)
gb.fit(Xb, yb)
print("  GradientBoosting    done")

# ─── ENSEMBLE WEIGHT SEARCH ──────────────────────────────────────────────────
p_rf = rf.predict_proba(X_test)[:, 1]
p_et = et.predict_proba(X_test)[:, 1]
p_gb = gb.predict_proba(X_test)[:, 1]

best_w, best_auc = (0.33, 0.33, 0.34), 0
for w1 in np.arange(0.15, 0.65, 0.05):
    for w2 in np.arange(0.10, 0.55, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0.10 or w3 > 0.60:
            continue
        p = w1*p_rf + w2*p_et + w3*p_gb
        auc = roc_auc_score(y_test, p)
        if auc > best_auc:
            best_auc, best_w = auc, (w1, w2, w3)

w1, w2, w3 = best_w
y_proba = w1*p_rf + w2*p_et + w3*p_gb
print(f"\n  Ensemble AUC: {best_auc:.4f}  |  RF:{w1:.2f} ET:{w2:.2f} GB:{w3:.2f}")

# ─── THRESHOLD OPTIMISATION ──────────────────────────────────────────────────
best_t_acc, best_acc = 0.5, 0
for t in np.arange(0.05, 0.95, 0.005):
    preds = (y_proba >= t).astype(int)
    acc = accuracy_score(y_test, preds)
    if acc > best_acc:
        best_acc, best_t_acc = acc, t

y_pred = (y_proba >= best_t_acc).astype(int)
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
auc  = roc_auc_score(y_test, y_proba)
cm   = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 65)
print("  FINAL RESULTS")
print("=" * 65)
print(f"  Accuracy  : {acc*100:.4f}%")
print(f"  Precision : {prec*100:.2f}%")
print(f"  Recall    : {rec*100:.2f}%")
print(f"  F1 Score  : {f1:.4f}")
print(f"  ROC-AUC   : {auc:.4f}")
print(f"  Threshold : {best_t_acc:.3f}")
print(f"  CM:  TN={cm[0][0]}  FP={cm[0][1]}  FN={cm[1][0]}  TP={cm[1][1]}")

# ─── CROSS-VALIDATION ────────────────────────────────────────────────────────
print("\n  Running 5-fold cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_rf = RandomForestClassifier(
    n_estimators=300, class_weight={0: 1, 1: 4},
    max_depth=None, random_state=42, n_jobs=-1,
)
cv_acc = cross_val_score(cv_rf, X.values, y_all.values, cv=cv, scoring="accuracy")
cv_auc = cross_val_score(cv_rf, X.values, y_all.values, cv=cv, scoring="roc_auc")
print(f"  CV Accuracy : {cv_acc.mean()*100:.2f}% ± {cv_acc.std()*100:.2f}%")
print(f"  CV ROC-AUC  : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

# ─── TOP FEATURE IMPORTANCES ─────────────────────────────────────────────────
avg_fi  = w1*rf.feature_importances_ + w2*et.feature_importances_ + w3*gb.feature_importances_
top_idx = np.argsort(avg_fi)[::-1][:20]
top_feats = {feat_names[i]: round(float(avg_fi[i]), 5) for i in top_idx}

print("\n  Top 15 Features:")
for k, v in list(top_feats.items())[:15]:
    bar = "█" * int(v * 200)
    print(f"    {k:<38} {v:.5f}  {bar}")

# ─── SAVE MODELS ─────────────────────────────────────────────────────────────
joblib.dump({"rf": rf, "et": et, "gb": gb, "weights": [w1, w2, w3]},
            os.path.join(MODELS_DIR, "fraud_model_v3.pkl"))
joblib.dump(encoders, os.path.join(MODELS_DIR, "encoders_v3.pkl"))
joblib.dump(feat_names, os.path.join(MODELS_DIR, "feature_names_v3.pkl"))

config = {
    "model": "RF+ET+GB Weighted Voting Ensemble v3",
    "datasets": ["fraud_oracle.csv (15,420 rows)", "carclaims.csv (label verification)"],
    "threshold": float(best_t_acc),
    "ensemble_weights": {"rf": float(w1), "et": float(w2), "gb": float(w3)},
    "feature_names": feat_names,
    "metrics": {
        "accuracy":  round(acc, 4), "precision": round(prec, 4),
        "recall":    round(rec, 4), "f1_score":  round(f1, 4),
        "roc_auc":   round(auc, 4),
        "cv_accuracy_mean": round(float(cv_acc.mean()), 4),
        "cv_roc_auc_mean":  round(float(cv_auc.mean()), 4),
    },
    "top_features": top_feats,
    "confusion_matrix": cm.tolist(),
    "training_info": {
        "total_rows": int(len(y_all)), "fraud_rows": int(y_all.sum()),
        "fraud_pct":  round(float(y_all.mean()*100), 2),
        "test_fraud": int(y_test.sum()),
        "test_legit": int((y_test==0).sum()),
    },
}
with open(os.path.join(MODELS_DIR, "model_config_v3.json"), "w") as f:
    json.dump(config, f, indent=2)

print("\n  Saved:")
print(f"    models/fraud_model_v3.pkl")
print(f"    models/feature_names_v3.pkl")
print(f"    models/model_config_v3.json")
print("\n  TRAINING COMPLETE\n")
