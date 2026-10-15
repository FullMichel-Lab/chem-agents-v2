"""Définition de l'état partagé du StateGraph."""

from typing import TypedDict, Annotated, Optional
import operator
from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps


class ChemAgentState(TypedDict):
    """État partagé entre tous les agents du workflow."""
    
    # Historique des messages (traçabilité)
    messages: Annotated[list, add_messages]
    
    # ⚠️ CHAMP OBLIGATOIRE POUR LANGGRAPH-SUPERVISOR
    # Déclaré comme canal géré (managed channel) pour éviter
    # l'avertissement "wrote to unknown channel remaining_steps"
    remaining_steps: RemainingSteps
    
    # Entrées utilisateur
    molecule_name: Optional[str]
    molecule_cid: Optional[int]
    smiles: Optional[str]
    target_property: Optional[str]
    
    # Résultats intermédiaires
    pubchem_data: Optional[dict]
    rdkit_descriptors: Optional[dict]
    prediction_result: Optional[dict]
    
    # Suggestions (accumulées)
    suggestions: Annotated[list[str], operator.add]
    
    # Statut et erreurs
    status: str
    errors: Annotated[list[str], operator.add]
