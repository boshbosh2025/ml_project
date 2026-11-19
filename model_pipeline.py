import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# ===== 1. Préparation des données =====
def prepare_data(filepath):
    """Charge, nettoie et prépare les données selon la logique OM5."""
    df = pd.read_csv(filepath)
    
    # Nettoyage
    df = df.replace("?", np.nan)
    df_clean = df.drop(columns=["weight"], errors="ignore")
    cols_to_fill = ["A1Cresult", "max_glu_serum", "medical_specialty", "payer_code", "race", "diag_1", "diag_2", "diag_3"]
    for col in cols_to_fill:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna("Unknown" if col != "A1Cresult" and col != "max_glu_serum" else "Not measured")
    
    df_om5 = df_clean.copy()
    
    # Feature engineering
    med_columns_om5 = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]
    monthly_meds_om5 = ["metformin", "glimepiride", "glipizide", "glyburide", "pioglitazone", "rosiglitazone", "insulin"]
    
    df_om5["is_new_med"] = (df_om5[med_columns_om5] != "No").any(axis=1).astype(int)
    df_om5["is_monthly_med"] = (df_om5[monthly_meds_om5] != "No").any(axis=1).astype(int)
    
    # Cible
    df_om5["target_om5"] = df_om5["change"].map({"Ch": 1, "No": 0})
    df_om5 = df_om5.dropna(subset=["target_om5"]).copy()
    y_om5 = df_om5["target_om5"].astype(int)
    
    # Groupes et exclusions
    groups_om5 = df_om5["patient_nbr"].copy()
    excluded_columns_om5 = ["encounter_id", "patient_nbr", "change", "target_om5", "readmitted", "discharge_disposition_id", "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", *med_columns_om5, "diabetesMed", "is_new_med", "is_monthly_med"]
    df_om5_features = df_om5.drop(columns=[c for c in excluded_columns_om5 if c in df_om5.columns], errors="ignore").copy()
    
    features_om5 = ["age", "gender", "race", "diag_1", "diag_2", "diag_3", "number_diagnoses", "A1Cresult", "max_glu_serum", "admission_type_id", "admission_source_id", "medical_specialty", "payer_code", "number_outpatient", "number_emergency", "number_inpatient"]
    features_om5 = [c for c in features_om5 if c in df_om5_features.columns]
    X_om5 = df_om5_features[features_om5].copy()
    
    age_mapping_om5 = {"[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75, "[80-90)": 85, "[90-100)": 95}
    X_om5["age_midpoint"] = X_om5["age"].map(age_mapping_om5)
    X_om5["previous_visits"] = X_om5["number_outpatient"] + X_om5["number_emergency"] + X_om5["number_inpatient"]
    X_om5 = X_om5.drop(columns=["age"]).copy()
    
    # Split par patient
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X_om5, y_om5, groups=groups_om5))
    
    return X_om5.iloc[train_idx], X_om5.iloc[test_idx], y_om5.iloc[train_idx], y_om5.iloc[test_idx]

# ===== 2. Entraînement =====
def train_model(X_train, y_train):
    """Entraîne le Gradient Boosting avec le pipeline de prétraitement."""
    categorical_features = ["gender", "race", "diag_1", "diag_2", "diag_3", "A1Cresult", "max_glu_serum", "admission_type_id", "admission_source_id", "medical_specialty", "payer_code"]
    numeric_features = ["age_midpoint", "number_diagnoses", "number_outpatient", "number_emergency", "number_inpatient", "previous_visits"]
    
    categorical_features = [c for c in categorical_features if c in X_train.columns]
    numeric_features = [c for c in numeric_features if c in X_train.columns]
    
    prep = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, min_frequency=200), categorical_features),
        ("num", "passthrough", numeric_features),
    ])
    
    gb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100,
        l2_regularization=10, early_stopping=True, random_state=42,
        categorical_features=[True] * len(categorical_features) + [False] * len(numeric_features),
    )
    
    pipeline = Pipeline([("prep", prep), ("gb", gb)])
    cols = categorical_features + numeric_features
    pipeline.fit(X_train[cols], y_train)
    return pipeline

# ===== 3. Évaluation =====
def evaluate_model(model, X_test, y_test):
    """Évalue les performances (AUC et Accuracy)."""
    p_test = model.predict_proba(X_test)[:, 1]
    pred_test = (p_test >= 0.5).astype(int)
    
    auc = roc_auc_score(y_test, p_test)
    acc = accuracy_score(y_test, pred_test)
    print(f"AUC test : {auc:.3f}")
    print(f"Accuracy : {acc:.3f}")
    return auc

# ===== 4. Sauvegarde et chargement =====
def save_model(model, filepath):
    """Sauvegarde le modèle."""
    joblib.dump(model, filepath)
    print(f"Modèle sauvegardé dans {filepath}")

def load_model(filepath):
    """Charge un modèle."""
    return joblib.load(filepath)
