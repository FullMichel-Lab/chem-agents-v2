"""Point d'entrée principal avec tests de validation."""

import os
import sys
import time
import json
import re
from dotenv import load_dotenv

from agents import create_llm, create_chem_supervisor
from tools import find_similar_molecules, compute_sa_score, predict_solubility_with_uncertainty
from state import ChemAgentState

load_dotenv()


def detect_optimization_intent(query: str) -> bool:
    """Détecte si la requête contient une intention d'optimisation."""
    keywords = [
        "optimis", "optimiz", "optimisation", "optimization",
        "amélior", "amelior", "amélioration",
        "propose des modifications", "propose des améliorations",
        "analogues", "similaires", "similar",
        "suggestions structurales", "suggestions de modification",
        "modifier", "modification",
    ]
    query_lower = query.lower()
    return any(kw in query_lower for kw in keywords)


def detect_uncertainty_intent(query: str) -> bool:
    """Détecte si la requête contient une intention de quantification d'incertitude."""
    keywords = [
        "incertitude", "intervalle de confiance", "intervalle",
        "uncertainty", "confidence interval",
        "quantification", "barre d'erreur",
    ]
    query_lower = query.lower()
    return any(kw in query_lower for kw in keywords)


def extract_smiles_from_messages(messages: list) -> str:
    """Extrait un SMILES depuis les appels d'outils déjà effectués."""
    for m in messages:
        tool_calls = getattr(m, 'tool_calls', None)
        if tool_calls:
            for tc in tool_calls:
                name = tc.get('name', '') if isinstance(tc, dict) else getattr(tc, 'name', '')
                args = tc.get('args', {}) if isinstance(tc, dict) else getattr(tc, 'args', {})
                if 'smiles' in args:
                    smiles = args['smiles']
                    if smiles and isinstance(smiles, str):
                        from rdkit import Chem
                        if Chem.MolFromSmiles(smiles) is not None:
                            return smiles
    return extract_smiles_from_query_fallback(messages)


def extract_smiles_from_query_fallback(messages: list) -> str:
    """Extrait un SMILES du texte des messages en fallback."""
    full_text = " ".join(str(getattr(m, 'content', '')) for m in messages)
    matches = re.findall(r'[A-Z][A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]{9,}', full_text)
    from rdkit import Chem
    for candidate in matches:
        candidate = candidate.rstrip('.,;:)')
        if Chem.MolFromSmiles(candidate) is not None:
            return candidate
    return None


def extract_modified_smiles(analysis_text: str) -> list:
    """Extrait les SMILES modifiés depuis la réponse du LLM."""
    from rdkit import Chem
    smiles_list = []
    pattern = r'SMILES\s+modifi[ée][^\n:]*:\s*[`\'"]?([A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]+)'
    matches = re.findall(pattern, analysis_text, re.IGNORECASE)
    print(f"→ Regex a trouvé {len(matches)} candidat(s) SMILES")
    for smiles in matches:
        smiles = smiles.strip().rstrip('.,;:)`\'"')
        print(f"  → Test du SMILES : '{smiles}'")
        if Chem.MolFromSmiles(smiles) is not None:
            print(f"    ✅ Valide, ajouté à la liste")
            smiles_list.append(smiles)
        else:
            print(f"    ❌ Invalide, ignoré")
    return smiles_list


def run_uq_test(query: str):
    """Exécute un test de prédiction avec incertitude, en contournant le LLM."""
    print(f"\n{'='*60}")
    print("TEST 8 - Prédiction avec intervalle de confiance (UQ)")
    print(f"Requête: {query}")
    print('='*60)
    
    start_time = time.time()
    
    # Extraire le SMILES de la requête
    smiles = None
    match = re.search(r'SMILES\s*:?\s*([A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]{5,})', query)
    if match:
        smiles = match.group(1).strip().rstrip(')')
    else:
        # Fallback : chercher un SMILES connu
        match = re.search(r'ibuprofène|ibuprofen', query, re.IGNORECASE)
        if match:
            smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
    
    if not smiles:
        print("❌ Impossible d'extraire le SMILES de la requête.")
        return None
    
    # Validation RDKit
    from rdkit import Chem
    smiles = smiles.strip().rstrip(')').rstrip('(')
    if Chem.MolFromSmiles(smiles) is None:
        print(f"❌ SMILES invalide : '{smiles}'")
        return None
    
    print(f"✅ SMILES validé : '{smiles}'")
    print("\n→ Appel du modèle UQ (Random Forest + Conformal Prediction)...")
    
    # Appel direct de l'outil UQ
    uq_result = predict_solubility_with_uncertainty.invoke({"smiles": smiles})
    
    elapsed = time.time() - start_time
    
    print(f"\n--- Résultat de la prédiction UQ (temps: {elapsed:.1f}s) ---")
    
    if "error" in uq_result:
        print(f"❌ Erreur : {uq_result['error']}")
        return None
    
    # Affichage structuré
    print(f"\n📊 Prédiction de solubilité aqueuse (logS)")
    print(f"   Molécule       : {smiles}")
    print(f"   Valeur prédite : {uq_result['predicted_logS']} logS")
    print(f"   Intervalle {uq_result['coverage_level']} : "
          f"[{uq_result['interval_lower']}, {uq_result['interval_upper']}] logS")
    print(f"   Largeur        : {uq_result['interval_width']} logS")
    
    if "descriptors" in uq_result:
        print(f"\n📋 Descripteurs utilisés :")
        for key, val in uq_result["descriptors"].items():
            print(f"   {key} : {val}")
    
    print(f"\n💡 Interprétation : La solubilité de la molécule est estimée à "
          f"{uq_result['predicted_logS']} logS, avec une incertitude de "
          f"±{uq_result['interval_width']/2:.2f} logS (intervalle à {uq_result['coverage_level']}).")
    
    return uq_result


def run_single_test(app, query: str, test_name: str):
    """Exécute un test unique et affiche les résultats."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"Requête: {query}")
    print('='*60)
    
    initial_state = {
        "messages": [{"role": "user", "content": query}],
        "molecule_name": None,
        "molecule_cid": None,
        "smiles": None,
        "target_property": None,
        "pubchem_data": None,
        "rdkit_descriptors": None,
        "prediction_result": None,
        "suggestions": [],
        "status": "pending",
        "errors": [],
    }
    
    start_time = time.time()
    
    try:
        result = app.invoke(initial_state, config={"recursion_limit": 30})
        
        # Détection d'intention d'optimisation
        if detect_optimization_intent(query):
            messages = result.get("messages", [])
            
            similarity_called = False
            for m in messages:
                tool_calls = getattr(m, 'tool_calls', None)
                if tool_calls:
                    for tc in tool_calls:
                        name = tc.get('name', '') if isinstance(tc, dict) else getattr(tc, 'name', '')
                        if 'find_similar_molecules' in name:
                            similarity_called = True
                            break
                if similarity_called:
                    break
            
            if not similarity_called:
                print("\n" + "="*60)
                print("⚠️  Intention d'optimisation détectée.")
                print("→ Appel DIRECT de find_similar_molecules (contournement LLM)...")
                print("="*60 + "\n")
                
                smiles = extract_smiles_from_messages(messages)
                
                if smiles:
                    smiles = smiles.strip().rstrip(')').rstrip('(')
                    from rdkit import Chem
                    if Chem.MolFromSmiles(smiles) is None:
                        print(f"❌ SMILES invalide après nettoyage : '{smiles}'")
                        smiles = None
                    else:
                        print(f"✅ SMILES validé : '{smiles}'")
                
                if smiles:
                    similarity_result = find_similar_molecules.invoke({
                        "smiles": smiles,
                        "threshold": 0.75,
                        "max_records": 5
                    })
                    
                    print(f"\n--- Analogues PubChem trouvés ---")
                    
                    if "similar_molecules" in similarity_result:
                        analogues = similarity_result["similar_molecules"]
                        print(f"Nombre d'analogues : {len(analogues)}")
                        
                        for i, ana in enumerate(analogues[:10], 1):
                            print(f"\n[{i}] CID {ana.get('cid')}")
                            print(f"    SMILES: {ana.get('smiles')}")
                            print(f"    MW: {ana.get('molecular_weight')}")
                            print(f"    XLogP: {ana.get('xlogp')}")
                            print(f"    TPSA: {ana.get('tpsa')}")
                        
                        compact_list = []
                        for a in analogues[:5]:
                            compact_list.append(
                                f"CID {a.get('cid')} | SMILES: {a.get('smiles')} | XLogP: {a.get('xlogp')}"
                            )
                        compact_text = "\n".join(compact_list)
                        cid_list = ', '.join(str(a.get('cid')) for a in analogues[:5])
                        
                        followup_messages = [
                            {
                                "role": "user",
                                "content": (
                                    f"Molécule cible (ibuprofène) : LogP=3.07, MW=206.28, TPSA=37.3\n\n"
                                    f"5 analogues PubChem réels :\n{compact_text}\n\n"
                                    f"⚠️ RÈGLE ABSOLUE : Tu ne dois citer QUE les CIDs suivants : "
                                    f"{cid_list}. N'invente JAMAIS un autre CID.\n\n"
                                    f"TÂCHE : Propose 2 modifications structurales de l'ibuprofène.\n\n"
                                    f"FORMAT OBLIGATOIRE pour chaque modification :\n"
                                    f"Modification N :\n"
                                    f"- Description : [texte]\n"
                                    f"- SMILES modifié : [SMILES valide et complet]\n"
                                    f"- Référence PubChem : CID [numéro]\n"
                                    f"- Propriété observée : [valeur]\n"
                                    f"- Impact attendu : [texte]\n\n"
                                    f"Tu DOIS fournir un SMILES valide pour chaque modification. "
                                    f"Sans SMILES, ta réponse sera rejetée. Sois CONCIS : maximum 250 mots."
                                )
                            }
                        ]
                        
                        from langchain_groq import ChatGroq
                        llm_direct = ChatGroq(
                            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
                            api_key=os.getenv("GROQ_API_KEY"),
                            temperature=0.1,
                            timeout=120,
                        )
                        
                        print("\n--- Analyse par le LLM ---")
                        try:
                            response = llm_direct.invoke(followup_messages)
                            analysis_text = response.content if hasattr(response, 'content') else str(response)
                            print(analysis_text)
                            
                            messages.append({
                                "role": "assistant",
                                "content": analysis_text
                            })
                            
                            print("\n--- Évaluation de la synthétabilité (SAscore) ---")
                            modified_smiles_list = extract_modified_smiles(analysis_text)
                            
                            if modified_smiles_list:
                                print(f"Nombre de SMILES modifiés détectés : {len(modified_smiles_list)}")
                                sa_results = []
                                for i, mod_smiles in enumerate(modified_smiles_list, 1):
                                    sa_result = compute_sa_score.invoke({"smiles": mod_smiles})
                                    if "error" in sa_result:
                                        print(f"\n[Modification {i}] Erreur : {sa_result['error']}")
                                    else:
                                        sa_score = sa_result.get("sa_score")
                                        interpretation = sa_result.get("interpretation")
                                        print(f"\n[Modification {i}] SMILES : {mod_smiles}")
                                        print(f"  SAscore : {sa_score} ({interpretation})")
                                        
                                        if sa_score <= 3:
                                            verdict = "✅ Facile à synthétiser"
                                        elif sa_score <= 6:
                                            verdict = "⚠️  Accessibilité modérée"
                                        else:
                                            verdict = "❌ Difficile à synthétiser"
                                        print(f"  Verdict : {verdict}")
                                        
                                        sa_results.append({
                                            "modification_index": i,
                                            "smiles": mod_smiles,
                                            "sa_score": sa_score,
                                            "interpretation": interpretation,
                                            "verdict": verdict,
                                        })
                                
                                messages.append({
                                    "role": "system",
                                    "content": f"Évaluation SAscore : {sa_results}"
                                })
                            else:
                                print("Aucun SMILES modifié détecté dans la réponse du LLM.")
                                print("→ Le LLM n'a pas respecté le format demandé.")
                        except Exception as e:
                            print(f"Erreur lors de l'analyse LLM : {e}")
                else:
                    print("⚠️  Impossible d'extraire le SMILES de la requête.")
        
        elapsed = time.time() - start_time
        
        print(f"\n--- Résultats (temps: {elapsed:.1f}s) ---")
        print(f"Statut: {result.get('status', 'unknown')}")
        
        messages = result.get("messages", [])
        if messages:
            print(f"\n--- Trace complète ({len(messages)} messages) ---")
            for i, msg in enumerate(messages):
                role = getattr(msg, 'type', msg.__class__.__name__)
                content = getattr(msg, 'content', str(msg))
                if isinstance(content, str) and len(content) > 400:
                    content = content[:400] + "... [tronqué]"
                print(f"\n[{i}] {role}: {content}")
                tool_calls = getattr(msg, 'tool_calls', None)
                if tool_calls:
                    for tc in tool_calls:
                        name = tc.get('name', '?') if isinstance(tc, dict) else getattr(tc, 'name', '?')
                        args = tc.get('args', {}) if isinstance(tc, dict) else getattr(tc, 'args', {})
                        print(f"    → Outil appelé: {name}")
                        print(f"      Arguments: {args}")
        
        return result
    
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\nERREUR après {elapsed:.1f}s: {type(e).__name__} - {str(e)}")
        return None


def main():
    """Fonction principale."""
    print("="*60)
    print("SYSTÈME MULTI-AGENTS POUR L'ANALYSE MOLÉCULAIRE")
    print("Architecture: LangGraph Supervisor")
    print("="*60)
    
    test_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    # Test 8 : UQ - ne nécessite pas le superviseur LangGraph
    if test_arg == "8":
        query = "Prédis la solubilité de l'ibuprofène (SMILES: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O) avec un intervalle de confiance à 90%."
        run_uq_test(query)
        print("\n" + "="*60)
        print("FIN DES TESTS")
        print("="*60)
        return
    
    # Vérifier qu'Ollama est accessible (pour les autres tests)
    print("\nVérification de la connexion Ollama...")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m['name'] for m in response.json().get("models", [])]
        print(f"Modèles disponibles: {models}")
    except Exception as e:
        print(f"AVERTISSEMENT: Ollama non accessible - {e}")
        return
    
    llm = create_llm()
    app = create_chem_supervisor(llm)
    
    tests = {
        "1": ("Analyse la molécule 'aspirin' et donne-moi ses propriétés PubChem.",
              "TEST 1 - Récupération PubChem par nom"),
        "2": ("Calcule les descripteurs RDKit pour le SMILES: CC(=O)OC1=CC=CC=C1C(=O)O",
              "TEST 2 - Descripteurs RDKit par SMILES"),
        "3": ("Prédis le LogP pour le SMILES: CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
              "TEST 3 - Prédiction LogP (caféine)"),
        "4": ("Prédis la solubilité pour le SMILES: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
              "TEST 4 - Prédiction solubilité (ibuprofène)"),
        "5": ("Calcule les descripteurs pour le SMILES: C1=CC=CC",
              "TEST 5 - Robustesse (SMILES invalide)"),
        "6": ("Analyse la molécule 'xyzabc123' dans PubChem.",
              "TEST 6 - Robustesse (molécule inexistante)"),
        "7": ("Analyse l'ibuprofène (SMILES: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O) "
              "et propose des optimisations basées sur des analogues PubChem réels.",
              "TEST 7 - Optimisation avec similarité PubChem"),
    }
    
    if test_arg == "all":
        for key in sorted(tests.keys()):
            query, name = tests[key]
            run_single_test(app, query, name)
        # Ajouter le test 8 à la fin
        print("\n\n" + "="*60)
        print("PASSAGE AU TEST 8 (UQ)")
        print("="*60)
        query_8 = "Prédis la solubilité de l'ibuprofène (SMILES: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O) avec un intervalle de confiance à 90%."
        run_uq_test(query_8)
    elif test_arg in tests:
        query, name = tests[test_arg]
        run_single_test(app, query, name)
    else:
        print(f"Test inconnu: {test_arg}. Utilise 1-8 ou 'all'.")
        return
    
    print("\n" + "="*60)
    print("FIN DES TESTS")
    print("="*60)


if __name__ == "__main__":
    main()
