"""Création des agents spécialisés et du superviseur LangGraph."""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

from tools import (
    search_pubchem_by_name,
    get_pubchem_properties,
    compute_rdkit_descriptors,
    predict_property_from_descriptors,
    find_similar_molecules,
    compute_tanimoto_similarity,
    compute_sa_score,
    predict_solubility_with_uncertainty,
)
from prompts import (
    SUPERVISOR_PROMPT,
    RETRIEVE_PROMPT,
    COMPUTE_PROMPT,
    PREDICT_PROMPT,
    OPTIMIZE_PROMPT,
)
from state import ChemAgentState

load_dotenv()


def create_llm():
    """Crée l'instance LLM connectée à l'API Groq."""
    #model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY manquante. Ajoute-la dans le fichier .env"
        )
    
    print(f"→ Modèle Groq : {model_name}")
    
    return ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=0.1,
        max_retries=2,
        timeout=120,
    )


def create_chem_agents(llm):
    """Crée les quatre agents spécialisés."""
    
    retrieve_agent = create_react_agent(
        model=llm,
        tools=[search_pubchem_by_name, get_pubchem_properties],
        name="retrieve_data",
        prompt=RETRIEVE_PROMPT,
    )
    
    compute_agent = create_react_agent(
        model=llm,
        tools=[compute_rdkit_descriptors],
        name="compute_descriptors",
        prompt=COMPUTE_PROMPT,
    )
    
    predict_agent = create_react_agent(
        model=llm,
        tools=[predict_property_from_descriptors,
                predict_solubility_with_uncertainty,
               ],
        name="predict_property",
        prompt=PREDICT_PROMPT,
    )
    
    # Agent 4 : Optimisation (avec similarité PubChem et SAscore)
    optimize_agent = create_react_agent(
    model=llm,
    tools=[
        compute_rdkit_descriptors,
        predict_property_from_descriptors,
        find_similar_molecules,
        compute_tanimoto_similarity,
        compute_sa_score,
    ],
    name="suggest_optimization",
    prompt=OPTIMIZE_PROMPT,
)
    
    return [retrieve_agent, compute_agent, predict_agent, optimize_agent]


def create_chem_supervisor(llm):
    """Crée le superviseur qui orchestre les agents."""
    print("\n→ Création des agents spécialisés...")
    agents = create_chem_agents(llm)
    
    print("→ Création du superviseur LangGraph...")
    workflow = create_supervisor(
        agents=agents,
        model=llm,
        prompt=SUPERVISOR_PROMPT,
        state_schema=ChemAgentState,
    )
    
    print("→ Compilation du workflow...\n")
    return workflow.compile()
