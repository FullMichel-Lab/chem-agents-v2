"""Outils pour la récupération de données et le calcul de descripteurs."""

import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from langchain_core.tools import tool
from typing import Optional


@tool
def search_pubchem_by_name(molecule_name: str) -> dict:
    """Recherche une molécule dans PubChem par son nom.
    
    Args:
        molecule_name: Nom de la molécule (ex: "aspirin", "caffeine")
    
    Returns:
        Dictionnaire avec le CID et les propriétés, ou une erreur.
    """
    try:
        compounds = pcp.get_compounds(molecule_name, 'name')
        if not compounds:
            return {"error": f"Molécule '{molecule_name}' non trouvée dans PubChem"}
        
        compound = compounds[0]
        return {
            "cid": compound.cid,
            "molecular_formula": compound.molecular_formula,
            "molecular_weight": compound.molecular_weight,
            "canonical_smiles": compound.connectivity_smiles,
            "xlogp": compound.xlogp,
            "tpsa": compound.tpsa,
            "h_bond_donor_count": compound.h_bond_donor_count,
            "h_bond_acceptor_count": compound.h_bond_acceptor_count,
        }
    except Exception as e:
        return {"error": f"Erreur lors de la recherche: {str(e)}"}


@tool
def get_pubchem_properties(cid: int) -> dict:
    """Récupère les propriétés d'une molécule PubChem par son CID.
    
    Args:
        cid: Identifiant PubChem Compound
    
    Returns:
        Dictionnaire avec les propriétés demandées.
    """
    try:
        compound = pcp.Compound.from_cid(cid)
        return {
            "cid": compound.cid,
            "molecular_weight": compound.molecular_weight,
            "xlogp": compound.xlogp,
            "tpsa": compound.tpsa,
            "h_bond_donor_count": compound.h_bond_donor_count,
            "h_bond_acceptor_count": compound.h_bond_acceptor_count,
            "canonical_smiles": compound.connectivity_smiles,
        }
    except Exception as e:
        return {"error": f"Erreur pour CID {cid}: {str(e)}"}


@tool
def compute_rdkit_descriptors(smiles: str) -> dict:
    """Calcule les descripteurs moléculaires RDKit à partir d'un SMILES.
    
    Args:
        smiles: Représentation SMILES de la molécule
    
    Returns:
        Dictionnaire avec les descripteurs calculés.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"SMILES invalide: '{smiles}'"}
    
    return {
        "smiles": smiles,
        "molecular_weight": round(Descriptors.MolWt(mol), 2),
        "mol_logp": round(Descriptors.MolLogP(mol), 2),
        "tpsa": round(Descriptors.TPSA(mol), 2),
        "h_donors": Lipinski.NumHDonors(mol),
        "h_acceptors": Lipinski.NumHAcceptors(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "ring_count": Lipinski.RingCount(mol),
        "heavy_atom_count": mol.GetNumHeavyAtoms(),
    }


@tool
def predict_property_from_descriptors(
    mol_weight: float,
    mol_logp: float,
    tpsa: float,
    h_donors: int,
    h_acceptors: int,
    target_property: str
) -> dict:
    """Prédit une propriété physico-chimique à partir des descripteurs RDKit.
    
    Args:
        mol_weight: Masse molaire
        mol_logp: Coefficient de partition octanol/eau
        tpsa: Surface polaire topologique
        h_donors: Nombre de donneurs de liaisons hydrogène
        h_acceptors: Nombre d'accepteurs de liaisons hydrogène
        target_property: Propriété cible ("logp", "solubility", "permeability")
    
    Returns:
        Prédiction avec justification.
    """
    if target_property.lower() == "logp":
        return {
            "property": "logp",
            "predicted_value": mol_logp,
            "method": "RDKit MolLogP",
            "reliability": "high",
            "justification": f"Calcul direct RDKit: {mol_logp}"
        }
    
    elif target_property.lower() == "solubility":
        # Heuristique basée sur la règle de Lipinski
        if mol_logp > 5 or mol_weight > 500:
            solubility = "faible"
            reason = f"LogP élevé ({mol_logp}) et/ou masse élevée ({mol_weight})"
        elif mol_logp < 0 or tpsa > 100:
            solubility = "haute"
            reason = f"LogP faible ({mol_logp}) et polarité élevée (TPSA={tpsa})"
        else:
            solubility = "moyenne"
            reason = f"Propriétés intermédiaires (LogP={mol_logp}, TPSA={tpsa})"
        
        return {
            "property": "solubility",
            "predicted_value": solubility,
            "method": "Lipinski heuristic",
            "reliability": "medium",
            "justification": reason
        }
    
    elif target_property.lower() == "permeability":
        # Heuristique pour la perméabilité membranaire
        if 1 < mol_logp < 4 and tpsa < 90 and h_donors < 5:
            permeability = "haute"
            reason = "Respecte les critères de perméabilité optimale"
        else:
            permeability = "faible"
            reason = f"LogP={mol_logp} (optimal: 1-4), TPSA={tpsa} (optimal: <90)"
        
        return {
            "property": "permeability",
            "predicted_value": permeability,
            "method": "Lipinski heuristic",
            "reliability": "medium",
            "justification": reason
        }
    
    else:
        return {
            "error": f"Propriété '{target_property}' non supportée",
            "supported": ["logp", "solubility", "permeability"]
        }

@tool
def find_similar_molecules(smiles: str, threshold: float = 0.8, max_records: int = 20) -> dict:
    """Recherche des molécules structuralement similaires dans PubChem.

    Utilise l'API de similarité PubChem avec un seuil de Tanimoto.
    Le seuil par défaut de 0.8 offre un bon équilibre entre similarité
    structurale et diversité des résultats.

    Args:
        smiles: Représentation SMILES de la molécule de référence
        threshold: Seuil minimal de similarité Tanimoto (0-1), défaut 0.8
        max_records: Nombre maximal de molécules à retourner, défaut 20

    Returns:
        Dictionnaire contenant la liste des molécules similaires avec
        leurs propriétés principales, ou une erreur structurée.
    """
    try:
        compounds = pcp.get_compounds(
            smiles, 'smiles',
            searchtype='similarity',
            threshold=threshold,
            MaxRecords=max_records
        )
        
        if not compounds:
            return {
                "similar_molecules": [],
                "count": 0,
                "message": f"Aucune molécule similaire trouvée avec un seuil de {threshold}"
            }
        
        results = []
        for comp in compounds:
            if comp.cid and comp.connectivity_smiles:
                results.append({
                    "cid": comp.cid,
                    "smiles": comp.connectivity_smiles,
                    "molecular_weight": getattr(comp, 'molecular_weight', None),
                    "xlogp": getattr(comp, 'xlogp', None),
                    "tpsa": getattr(comp, 'tpsa', None),
                    "h_bond_donor_count": getattr(comp, 'h_bond_donor_count', None),
                    "h_bond_acceptor_count": getattr(comp, 'h_bond_acceptor_count', None),
                })
        
        return {
            "similar_molecules": results,
            "count": len(results),
            "threshold_used": threshold
        }
    
    except Exception as e:
        return {"error": f"Recherche de similarité échouée: {str(e)}"}


@tool
def compute_tanimoto_similarity(smiles1: str, smiles2: str) -> dict:
    """Calcule la similarité Tanimoto entre deux molécules.

    Utilise les empreintes de Morgan (rayon 2, 2048 bits), qui est
    la méthode standard en chémoinformatique pour évaluer la similarité
    structurale.

    Args:
        smiles1: SMILES de la première molécule
        smiles2: SMILES de la deuxième molécule

    Returns:
        Dictionnaire avec la valeur de similarité Tanimoto (0-1).
    """
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    if mol1 is None or mol2 is None:
        return {"error": "Un des SMILES est invalide"}
    
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    
    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    
    return {
        "smiles1": smiles1,
        "smiles2": smiles2,
        "tanimoto_similarity": round(similarity, 4)
    }


@tool
def compute_sa_score(smiles: str) -> dict:
    """Calcule le score de synthétabilité (SAscore) d'une molécule.

    Le SAscore, développé par Ertl et Schuffenhauer, varie de 1
    (facile à synthétiser) à 10 (très difficile à synthétiser).
    Un score inférieur à 6 est généralement considéré acceptable
    pour une molécule drug-like.

    Args:
        smiles: Représentation SMILES de la molécule

    Returns:
        Dictionnaire avec le score SA et son interprétation.
    """
    import os
    import sys
    from rdkit.Chem import RDConfig
    
    # Ajouter le répertoire contrib de RDKit au chemin Python
    sa_path = os.path.join(RDConfig.RDContribDir, 'SA_Score')
    if sa_path not in sys.path:
        sys.path.append(sa_path)
    
    try:
        import sascorer
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": f"SMILES invalide: '{smiles}'"}
        
        score = sascorer.calculateScore(mol)
        
        if score <= 3:
            interpretation = "facile à synthétiser"
        elif score <= 6:
            interpretation = "accessibilité synthétique modérée"
        else:
            interpretation = "difficile à synthétiser"
        
        return {
            "smiles": smiles,
            "sa_score": round(score, 2),
            "interpretation": interpretation
        }
    
    except ImportError:
        return {"error": "Le module sascorer n'est pas disponible dans cette installation RDKit"}


@tool
def predict_solubility_with_uncertainty(smiles: str, alpha: float = 0.9) -> dict:
    """Prédit la solubilité aqueuse avec un intervalle de confiance.
    
    Utilise un modèle Random Forest calibré avec Conformal Prediction
    pour fournir une prédiction ET un intervalle de confiance.
    
    Args:
        smiles: SMILES de la molécule
        alpha: niveau de signification (0.1 = 90% coverage)
    
    Returns:
        Dictionnaire avec prédiction et intervalle.
    """
    from uq_model import load_model, predict_with_uncertainty
    import os
    
    model_path = "uq_model.pkl"
    if not os.path.exists(model_path):
        return {
            "error": "Modèle UQ non entraîné. Lancez d'abord 'python uq_model.py'."
        }
    
    model_dict = load_model(model_path)
    return predict_with_uncertainty(model_dict, smiles, alpha)

# Liste des outils disponibles
ALL_TOOLS = [
    search_pubchem_by_name,
    get_pubchem_properties,
    compute_rdkit_descriptors,
    predict_property_from_descriptors,
    find_similar_molecules,
    compute_tanimoto_similarity,
    compute_sa_score,
    predict_solubility_with_uncertainty,
]
