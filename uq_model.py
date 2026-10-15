"""Modèle de prédiction de solubilité avec quantification d'incertitude."""

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# ============================================================
# CHARGEMENT ET PRÉPARATION DES DONNÉES
# ============================================================

def compute_features(smiles: str) -> dict:
    """Calcule les descripteurs moléculaires pour un SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    return {
        "MolWt": Descriptors.MolWt(mol),
        "MolLogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumHDonors": Lipinski.NumHDonors(mol),
        "NumHAcceptors": Lipinski.NumHAcceptors(mol),
        "NumRotatableBonds": Lipinski.NumRotatableBonds(mol),
        "RingCount": Lipinski.RingCount(mol),
        "HeavyAtomCount": mol.GetNumHeavyAtoms(),
        "FractionCSP3": Descriptors.FractionCSP3(mol),
        "NumAromaticRings": Descriptors.NumAromaticRings(mol),
    }


def load_and_prepare_data(data_path: str) -> tuple:
    """Charge le dataset ESOL et prépare les features/labels."""
    df = pd.read_csv(data_path)
    
    print(f"Dataset chargé : {len(df)} molécules")
    print(f"Colonnes disponibles : {df.columns.tolist()}")
    
    # Colonnes du dataset Delaney/ESOL
    smiles_col = "smiles"
    label_col = "measured log solubility in mols per litre"
    
    features_list = []
    labels_list = []
    valid_smiles = []
    
    for idx, row in df.iterrows():
        smiles = row.get(smiles_col)
        logS = row.get(label_col)
        
        if pd.isna(smiles) or pd.isna(logS):
            continue
        
        feats = compute_features(smiles)
        if feats is not None:
            features_list.append(feats)
            labels_list.append(float(logS))
            valid_smiles.append(smiles)
    
    X = pd.DataFrame(features_list)
    y = np.array(labels_list)
    
    print(f"Molécules valides : {len(X)}")
    
    return X, y, valid_smiles

# ============================================================
# ENTRAÎNEMENT DU MODÈLE
# ============================================================

def train_uq_model(X: pd.DataFrame, y: np.ndarray):
    """Entraîne un Random Forest avec calibration conformal (MAPIE 1.x)."""
    from mapie.regression import SplitConformalRegressor
    from mapie.conformity_scores import ResidualNormalisedScore
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    
    # Split train/calibration/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"Train: {len(X_train)} | Calibration: {len(X_calib)} | Test: {len(X_test)}")
    
    # Modèle de base
    base_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
    )
    
    # ============================================================
    # SCORE DE NON-CONFORMITÉ NORMALISÉ (au lieu du score absolu)
    # ============================================================
        # Étape 1 : créer le modèle qui prédit les résidus
    residual_estimator = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    
    # Étape 2 : passer ce modèle au score normalisé
    conformity_score = ResidualNormalisedScore(
        residual_estimator=residual_estimator,
        random_state=42,
    )
    
    mapie = SplitConformalRegressor(
        estimator=base_model,
        confidence_level=0.9,
        prefit=False,
        conformity_score=conformity_score,  # ← LE CHANGEMENT CLÉ
    )
    
    # Entraîner et calibrer
    mapie.fit(X_train, y_train)
    mapie.conformalize(X_calib, y_calib)
    
    # Prédiction
    y_pred, y_pis = mapie.predict_interval(X_test)
    
    # Métriques
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Couverture empirique
    coverage = np.mean(
        (y_test >= y_pis[:, 0, 0]) & (y_test <= y_pis[:, 1, 0])
    )
    
    print(f"\n=== Métriques du modèle ===")
    print(f"MAE : {mae:.3f} logS")
    print(f"R² : {r2:.3f}")
    print(f"Couverture empirique (cible 90%) : {coverage*100:.1f}%")
    
    return {
        "base_model": base_model,
        "mapie": mapie,
        "X_train": X_train,
        "y_train": y_train,
        "mae": mae,
        "r2": r2,
        "coverage": coverage,
    }


def save_model(model_dict: dict, path: str = "uq_model.pkl"):
    """Sauvegarde le modèle entraîné."""
    with open(path, "wb") as f:
        pickle.dump(model_dict, f)
    print(f"Modèle sauvegardé : {path}")


def load_model(path: str = "uq_model.pkl") -> dict:
    """Charge le modèle entraîné."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# PRÉDICTION AVEC INTERVALLE
# ============================================================

def predict_with_uncertainty(model_dict: dict, smiles: str, confidence: float = 0.9) -> dict:
    """Prédit la solubilité avec intervalle de confiance (MAPIE 1.x)."""
    feats = compute_features(smiles)
    if feats is None:
        return {"error": f"SMILES invalide : {smiles}"}
    
    X_new = pd.DataFrame([feats])
    
    # Prédiction avec intervalle (MAPIE 1.x)
    
    y_pred, y_pis = model_dict["mapie"].predict_interval(X_new)
    
    pred = float(y_pred[0])
    lower = float(y_pis[0, 0, 0])
    upper = float(y_pis[0, 1, 0])
    
    return {
        "smiles": smiles,
        "predicted_logS": round(pred, 3),
        "interval_lower": round(lower, 3),
        "interval_upper": round(upper, 3),
        "interval_width": round(upper - lower, 3),
        "coverage_level": f"{confidence*100:.0f}%",
        "descriptors": feats,
    }


# ============================================================
# SCRIPT D'ENTRAÎNEMENT
# ============================================================

if __name__ == "__main__":
    import sys
    
    # Télécharger le dataset ESOL si nécessaire
    data_path = Path("esol.csv")
    
    if not data_path.exists():
        print("Téléchargement du dataset ESOL (Delaney)...")
    # URL correcte du dataset Delaney/ESOL
        url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
        df = pd.read_csv(url)
        df.to_csv(data_path, index=False)
        print(f"Dataset sauvegardé : {data_path}")
        print(f"Colonnes : {df.columns.tolist()}")
    
    print("\n=== Entraînement du modèle UQ ===")
    X, y, smiles_list = load_and_prepare_data(str(data_path))
    
    model_dict = train_uq_model(X, y)
    save_model(model_dict)
    
    # Test rapide
    print("\n=== Test de prédiction ===")
    test_smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # Ibuprofène
    result = predict_with_uncertainty(model_dict, test_smiles)
    
    for key, value in result.items():
        if key != "descriptors":
            print(f"{key}: {value}")
