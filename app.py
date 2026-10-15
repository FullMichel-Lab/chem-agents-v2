"""Streamlit user interface for the chemical multi-agent system.

Version 2.1 - Fully English interface with:
- PubChem similarity-based optimization with SAscore
- Uncertainty quantification (UQ) prediction
"""

import streamlit as st
import time
import json
import re
from datetime import datetime

from agents import create_llm, create_chem_supervisor
from state import ChemAgentState
from tools import (
    find_similar_molecules,
    compute_sa_score,
    predict_solubility_with_uncertainty,
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Multi-Agent Chemical Analyzer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS STYLES
# ============================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1F4E79;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #F0F8FF;
        border-left: 5px solid #1F4E79;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .optimization-box {
        background-color: #FFF8F0;
        border-left: 5px solid #ED7D31;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .uq-box {
        background-color: #F0FFF4;
        border-left: 5px solid #70AD47;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .audit-box {
        background-color: #FAFAFA;
        border: 1px solid #DDD;
        padding: 1rem;
        border-radius: 5px;
        font-family: monospace;
        font-size: 0.85rem;
        max-height: 500px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "app" not in st.session_state:
    with st.spinner("Initializing multi-agent system..."):
        try:
            llm = create_llm()
            st.session_state.app = create_chem_supervisor(llm)
            st.session_state.llm_ready = True
        except Exception as e:
            st.session_state.llm_ready = False
            st.session_state.init_error = str(e)

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def run_analysis(query: str) -> dict:
    """Run the multi-agent workflow and return the result."""
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
        result = st.session_state.app.invoke(
            initial_state,
            config={"recursion_limit": 30}
        )
        elapsed = time.time() - start_time
        result["_elapsed"] = elapsed
        result["_success"] = True
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "_elapsed": elapsed,
            "_success": False,
            "_error": str(e),
            "messages": [],
        }


def run_optimization(smiles: str, target_property: str) -> dict:
    """Run the optimization workflow with PubChem similarity."""
    from rdkit import Chem
    
    start_time = time.time()
    
    # Validate SMILES
    if Chem.MolFromSmiles(smiles) is None:
        return {
            "_success": False,
            "_error": f"Invalid SMILES: {smiles}",
            "_elapsed": 0,
        }
    
    # Step 1: Search PubChem analogues
    with st.spinner("Searching PubChem analogues..."):
        similarity_result = find_similar_molecules.invoke({
            "smiles": smiles,
            "threshold": 0.75,
            "max_records": 5
        })
    
    if "error" in similarity_result:
        return {
            "_success": False,
            "_error": similarity_result["error"],
            "_elapsed": time.time() - start_time,
        }
    
    analogues = similarity_result.get("similar_molecules", [])
    
    if not analogues:
        return {
            "_success": False,
            "_error": "No analogues found in PubChem",
            "_elapsed": time.time() - start_time,
        }
    
    # Step 2: LLM analysis
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
                f"Target molecule: SMILES={smiles}\n\n"
                f"5 real PubChem analogues:\n{compact_text}\n\n"
                f"⚠️ ABSOLUTE RULE: You must ONLY cite the following CIDs: "
                f"{cid_list}. NEVER invent another CID.\n\n"
                f"TASK: Propose 2 structural modifications of the target molecule.\n\n"
                f"MANDATORY FORMAT for each modification:\n"
                f"Modification N:\n"
                f"- Description: [text]\n"
                f"- Modified SMILES: [valid and complete SMILES]\n"
                f"- PubChem reference: CID [number]\n"
                f"- Observed property: [value]\n"
                f"- Expected impact: [text]\n\n"
                f"You MUST provide a valid SMILES for each modification. "
                f"Without SMILES, your response will be rejected. Be CONCISE: maximum 250 words."
            )
        }
    ]
    
    from langchain_groq import ChatGroq
    import os
    llm_direct = ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1,
        timeout=120,
    )
    
    with st.spinner("LLM analysis in progress..."):
        response = llm_direct.invoke(followup_messages)
        analysis_text = response.content if hasattr(response, 'content') else str(response)
    
    # Step 3: Extract modified SMILES
    smiles_pattern = r'SMILES\s+modifi[ée][^\n:]*:\s*[`\'"]?([A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]+)'
    modified_smiles_list = []
    for s in re.findall(smiles_pattern, analysis_text, re.IGNORECASE):
        s = s.strip().rstrip('.,;:)`\'"')
        if Chem.MolFromSmiles(s) is not None:
            modified_smiles_list.append(s)
    
    # Step 4: Compute SAscore
    sa_results = []
    for i, mod_smiles in enumerate(modified_smiles_list, 1):
        sa_result = compute_sa_score.invoke({"smiles": mod_smiles})
        if "error" not in sa_result:
            sa_score = sa_result.get("sa_score")
            if sa_score <= 3:
                verdict = "✅ Easy"
            elif sa_score <= 6:
                verdict = "⚠️ Moderate"
            else:
                verdict = "❌ Difficult"
            sa_results.append({
                "index": i,
                "smiles": mod_smiles,
                "sa_score": sa_score,
                "interpretation": sa_result.get("interpretation"),
                "verdict": verdict,
            })
    
    return {
        "_success": True,
        "_elapsed": time.time() - start_time,
        "analogues": analogues,
        "analysis_text": analysis_text,
        "modified_smiles": modified_smiles_list,
        "sa_results": sa_results,
    }


def run_uq_prediction(smiles: str, confidence: float = 0.9) -> dict:
    """Run prediction with confidence interval."""
    from rdkit import Chem
    
    start_time = time.time()
    
    if Chem.MolFromSmiles(smiles) is None:
        return {
            "_success": False,
            "_error": f"Invalid SMILES: {smiles}",
            "_elapsed": 0,
        }
    
    with st.spinner("Predicting with uncertainty quantification..."):
        uq_result = predict_solubility_with_uncertainty.invoke({
            "smiles": smiles,
            "confidence": confidence,
        })
    
    if "error" in uq_result:
        return {
            "_success": False,
            "_error": uq_result["error"],
            "_elapsed": time.time() - start_time,
        }
    
    return {
        "_success": True,
        "_elapsed": time.time() - start_time,
        **uq_result,
    }


def extract_audit_trace(messages: list) -> list:
    """Extract audit trace from messages."""
    trace = []
    for i, msg in enumerate(messages):
        role = getattr(msg, 'type', msg.__class__.__name__)
        content = getattr(msg, 'content', str(msg))
        
        entry = {
            "index": i,
            "role": role,
            "content": content,
            "tool_calls": [],
        }
        
        tool_calls = getattr(msg, 'tool_calls', None)
        if tool_calls:
            for tc in tool_calls:
                if isinstance(tc, dict):
                    entry["tool_calls"].append({
                        "name": tc.get("name", "?"),
                        "args": tc.get("args", {}),
                    })
                else:
                    entry["tool_calls"].append({
                        "name": getattr(tc, "name", "?"),
                        "args": getattr(tc, "args", {}),
                    })
        
        trace.append(entry)
    
    return trace


def normalize_reliability(method: str, reliability: str) -> str:
    """Normalize reliability based on method."""
    method_lower = method.lower()
    
    if any(kw in method_lower for kw in ["pubchem", "xlogp3", "xlogp"]):
        return "🟡 Medium"
    if any(kw in method_lower for kw in ["rdkit", "mollogp", "mol_logp"]):
        return "🟢 High"
    if any(kw in method_lower for kw in ["lipinski", "heuristic", "heuristique"]):
        return "🟡 Medium"
    if any(kw in method_lower for kw in ["uq", "conformal", "random forest"]):
        return "🟢 High"
    
    reliability_lower = str(reliability).lower()
    return {
        "high": "🟢 High",
        "medium": "🟡 Medium",
        "low": "🔴 Low",
        "élevée": "🟢 High",
        "moyenne": "🟡 Medium",
        "faible": "🔴 Low",
    }.get(reliability_lower, f"⚪ {reliability}")


def format_result_display(content: str) -> None:
    """Display result in a formatted, readable way."""
    import json as json_module
    
    cleaned = content.strip()
    
    if cleaned.startswith("TERMINÉ"):
        cleaned = cleaned[len("TERMINÉ"):].strip()
    if cleaned.startswith("TERMINATED"):
        cleaned = cleaned[len("TERMINATED"):].strip()
    if cleaned.endswith("TERMINÉ"):
        cleaned = cleaned[:-len("TERMINÉ")].strip()
    if cleaned.endswith("TERMINATED"):
        cleaned = cleaned[:-len("TERMINATED")].strip()
    
    # Try to detect JSON block
    json_match = re.search(r'\{[\s\S]*\}', cleaned)
    
    if json_match:
        json_str = json_match.group(0)
        try:
            data = json_module.loads(json_str)
            before = cleaned[:json_match.start()].strip()
            after = cleaned[json_match.end():].strip()
            
            if before:
                st.markdown(before)
            display_json_result(data)
            if after:
                st.markdown(after)
            return
        except json_module.JSONDecodeError:
            pass
    
    # Try to detect structured text format
    if parse_structured_text(cleaned):
        return
    
    st.markdown(cleaned)


def parse_structured_text(text: str) -> bool:
    """Parse and display structured text."""
    patterns = {
        "value": [
            r'(?:LogP|logP|Predicted\s+value|Valeur\s+pr[ée]dite|Predicted\s+LogP)\s*[:=]\s*([^\n]+)',
            r'(?:Valeur\s+brute|Raw\s+value)\s*(?:pr[ée]dite)?\s*[:=]\s*([^\n]+)',
        ],
        "method": [
            r'(?:M[ée]thode(?:\s+utilis[ée]e)?|Method)\s*[:=]\s*([^\n]+)',
        ],
        "reliability": [
            r'(?:Niveau\s+de\s+fiabilit[ée]|Fiabilit[ée]|Reliability)\s*[:=]\s*([^\n]+)',
        ],
        "justification": [
            r'(?:Justification|Justification\s+fournie)\s*[:=]\s*([^\n]+(?:\n(?!\s*[A-Z][a-z]+\s*[:=])[^\n]+)*)',
        ],
    }
    
    extracted = {}
    for key, regex_list in patterns.items():
        for regex in regex_list:
            match = re.search(regex, text, re.IGNORECASE | re.MULTILINE)
            if match:
                extracted[key] = match.group(1).strip()
                break
    
    if "value" in extracted and "method" in extracted:
        st.markdown("#### 📊 Analysis Result")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted value", extracted.get("value", "N/A"))
        with col2:
            st.metric("Method", extracted.get("method", "N/A"))
        with col3:
            reliability = extracted.get("reliability", "N/A")
            method = extracted.get("method", "")
            reliability_display = normalize_reliability(method, reliability)
            st.metric("Reliability", reliability_display)
        
        if "justification" in extracted:
            st.markdown("**Justification:**")
            st.info(extracted["justification"])
        
        return True
    
    if "value" in extracted:
        st.markdown("#### 📊 Result")
        st.metric("Value", extracted["value"])
        if "method" in extracted:
            st.metric("Method", extracted["method"])
        if "justification" in extracted:
            st.info(extracted["justification"])
        return True
    
    return False


def display_json_result(data: dict) -> None:
    """Display a result dictionary in a structured way."""
    
    if "property" in data and "value" in data:
        property_name = data.get("property", "Property")
        value = data.get("value")
        method = data.get("method", "N/A")
        reliability = data.get("reliability", "N/A")
        justification = data.get("justification", "")
        
        st.markdown(f"#### 📊 Result: {property_name}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted value", f"{value}")
        with col2:
            st.metric("Method", method)
        with col3:
            reliability_display = normalize_reliability(method, reliability)
            st.metric("Reliability", reliability_display)
        
        if justification:
            st.markdown("**Justification:**")
            st.info(justification)
        return
    
    if "logP" in data or "logp" in data:
        for key in ["logP", "logp"]:
            if key in data and isinstance(data[key], dict):
                inner = data[key]
                st.markdown(f"#### 📊 Result: {key.upper()}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted value", f"{inner.get('valeur_brute', inner.get('value', 'N/A'))}")
                with col2:
                    method = inner.get('méthode', inner.get('method', 'N/A'))
                    st.metric("Method", method)
                with col3:
                    reliability = inner.get('fiabilité', inner.get('reliability', 'N/A'))
                    reliability_display = normalize_reliability(method, reliability)
                    st.metric("Reliability", reliability_display)
                
                if inner.get('justification'):
                    st.markdown("**Justification:**")
                    st.info(inner['justification'])
                return
    
    if "cid" in data and "molecular_weight" in data:
        st.markdown(f"#### 🧬 PubChem Properties — CID {data.get('cid')}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Formula", data.get("molecular_formula", "N/A"))
            st.metric("Molecular weight", f"{data.get('molecular_weight', 'N/A')} Da")
        with col2:
            st.metric("XLogP", data.get("xlogp", "N/A"))
            st.metric("TPSA", f"{data.get('tpsa', 'N/A')} Å²")
        with col3:
            st.metric("H-bond donors", data.get("h_bond_donor_count", "N/A"))
            st.metric("H-bond acceptors", data.get("h_bond_acceptor_count", "N/A"))
        
        if "canonical_smiles" in data:
            st.markdown("**Canonical SMILES:**")
            st.code(data["canonical_smiles"], language="text")
        
        for key in ["permeability_prediction", "solubility_prediction", "logp_prediction"]:
            if key in data and isinstance(data[key], dict):
                st.markdown(f"**{key.replace('_', ' ').title()}:**")
                display_json_result(data[key])
        return
    
    if "assessment" in data:
        st.markdown("#### 📊 Assessment")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Assessment", data.get("assessment", "N/A"))
        with col2:
            st.metric("Method", data.get("method", "N/A"))
        with col3:
            st.metric("Reliability", data.get("reliability", "N/A"))
        
        if data.get("justification"):
            st.markdown("**Justification:**")
            st.info(data["justification"])
        return
    
    st.json(data)


# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="main-header">🧪 Multi-Agent Chemical Analyzer</div>', 
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">LangGraph Supervisor · openai/gpt-oss-120b via Groq · RDKit · PubChem · Conformal Prediction</div>', 
            unsafe_allow_html=True)

if not st.session_state.llm_ready:
    st.error(f"❌ Initialization error: {st.session_state.init_error}")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    analysis_mode = st.radio(
        "Analysis mode",
        [
            "📊 Standard analysis",
            "🔧 Structural optimization",
            "📈 Uncertainty quantification",
        ],
        help="Choose the type of analysis to perform"
    )
    
    st.divider()
    
    if analysis_mode == "📊 Standard analysis":
        input_mode = st.radio(
            "Input mode",
            ["Molecule name", "SMILES", "PubChem CID"],
        )
        
        target_property = st.selectbox(
            "Target property",
            ["logp", "solubility", "permeability"],
            format_func=lambda x: {
                "logp": "LogP (lipophilicity)",
                "solubility": "Aqueous solubility",
                "permeability": "Membrane permeability",
            }[x]
        )
    
    elif analysis_mode == "🔧 Structural optimization":
        st.info("💡 Optimization searches real PubChem analogues and proposes structural modifications with synthetic accessibility assessment.")
        target_property = "optimization"
        input_mode = "SMILES"
    
    else:  # UQ
        st.info("💡 UQ prediction provides a solubility value with a statistically valid confidence interval.")
        confidence_pct = st.selectbox(
            "Confidence level",
            options=[80, 85, 90, 95],
            index=2,
            format_func=lambda x: f"{x}%"
        )
        confidence_level = confidence_pct / 100.0
        target_property = "solubility_uq"
        input_mode = "SMILES"
        
    st.divider()
    
    st.markdown("### 📊 Statistics")
    st.metric("Analyses performed", len(st.session_state.history))
    
    if st.session_state.history:
        avg_time = sum(h["elapsed"] for h in st.session_state.history) / len(st.session_state.history)
        st.metric("Average time", f"{avg_time:.1f}s")
        
        success_rate = sum(1 for h in st.session_state.history if h["success"]) / len(st.session_state.history) * 100
        st.metric("Success rate", f"{success_rate:.0f}%")
    
    st.divider()
    
    if st.button("🗑️ Clear history"):
        st.session_state.history = []
        st.rerun()


# ============================================================
# INPUT AREA
# ============================================================

st.markdown(f"### 📥 Input — {analysis_mode}")

col1, col2 = st.columns([3, 1])

with col1:
    if analysis_mode == "📊 Standard analysis":
        if input_mode == "Molecule name":
            user_input = st.text_input(
                "Molecule name",
                value="aspirin",
                placeholder="e.g., aspirin, caffeine, ibuprofen...",
                key="input_name"
            )
        elif input_mode == "SMILES":
            user_input = st.text_input(
                "SMILES",
                value="CC(=O)OC1=CC=CC=C1C(=O)O",
                placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O",
                key="input_smiles"
            )
        else:
            user_input = st.text_input(
                "PubChem CID",
                value="2244",
                placeholder="e.g., 2244",
                key="input_cid"
            )
    
    elif analysis_mode == "🔧 Structural optimization":
        user_input = st.text_input(
            "SMILES of the molecule to optimize",
            value="CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            placeholder="e.g., CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            key="input_opt_smiles",
            help="Enter the SMILES of the molecule you want to optimize"
        )
    
    else:  # UQ
        user_input = st.text_input(
            "SMILES of the molecule",
            value="CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            placeholder="e.g., CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            key="input_uq_smiles",
            help="Enter the SMILES of the molecule to predict solubility for"
        )

with col2:
    st.write("")
    st.write("")
    run_button = st.button("🚀 Run analysis", type="primary", use_container_width=True)


# ============================================================
# EXECUTION
# ============================================================

if run_button and user_input:
    start_time = time.time()
    
    # ============================================================
    # MODE 1: STANDARD ANALYSIS
    # ============================================================
    if analysis_mode == "📊 Standard analysis":
        prop_label = {
            "logp": "LogP",
            "solubility": "solubility",
            "permeability": "permeability",
        }[target_property]
        
        if input_mode == "Molecule name":
            query = f"Analyze the molecule '{user_input}' and predict its {prop_label}."
        elif input_mode == "SMILES":
            query = f"Predict the {prop_label} for the SMILES: {user_input}"
        else:
            query = f"Analyze the molecule with CID {user_input} and predict its {prop_label}."
        
        with st.spinner(f"Running multi-agent workflow for: {user_input}"):
            result = run_analysis(query)
        
        st.session_state.history.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "input": user_input,
            "mode": analysis_mode,
            "property": target_property,
            "elapsed": result.get("_elapsed", 0),
            "success": result.get("_success", False),
        })
        
        st.markdown("### 📊 Result")
        
        if result.get("_success"):
            messages = result.get("messages", [])
            
            if messages:
                last_msg = messages[-1]
                content = getattr(last_msg, 'content', str(last_msg))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⏱️ Time", f"{result['_elapsed']:.1f}s")
                with col2:
                    st.metric("📨 Messages", len(messages))
                with col3:
                    st.metric("✅ Status", "Success")
                
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                format_result_display(content)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error(f"❌ Error: {result.get('_error', 'Unknown error')}")
        
        # Audit
        st.markdown("### 🔍 Process Audit")
        with st.expander("📜 View full trace", expanded=False):
            if result.get("_success") and result.get("messages"):
                for entry in extract_audit_trace(result["messages"]):
                    role = entry["role"]
                    content = entry["content"]
                    if isinstance(content, str) and len(content) > 800:
                        content = content[:800] + "... [truncated]"
                    
                    color = {"human": "🔵", "ai": "🟢", "tool": "🟡"}.get(role, "⚪")
                    st.markdown(f"**{color} [{entry['index']}] {role}**")
                    st.markdown(f"```\n{content}\n```")
                    
                    if entry["tool_calls"]:
                        for tc in entry["tool_calls"]:
                            st.markdown(f"→ **Tool:** `{tc['name']}`")
                            st.markdown(f"   Args: `{json.dumps(tc['args'], ensure_ascii=False)}`")
                    st.divider()
            else:
                st.info("No trace available.")
        
        # Export
        st.markdown("### 💾 Export")
        col1, col2 = st.columns(2)
        
        with col1:
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "mode": analysis_mode,
                "input": user_input,
                "elapsed_seconds": result.get("_elapsed"),
                "success": result.get("_success"),
                "messages": [
                    {
                        "role": getattr(m, 'type', 'unknown'),
                        "content": str(getattr(m, 'content', m)),
                    }
                    for m in result.get("messages", [])
                ],
            }
            st.download_button(
                label="📥 JSON report",
                data=json.dumps(export_data, indent=2, ensure_ascii=False),
                file_name=f"report_{user_input}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )
        
        with col2:
            text_report = f"""MULTI-AGENT ANALYSIS REPORT
============================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Mode: {analysis_mode}
Input: {user_input}
Time: {result.get('_elapsed', 0):.1f}s
Status: {'Success' if result.get('_success') else 'Failure'}

FINAL RESPONSE
--------------
{result.get('messages', [{}])[-1].content if result.get('messages') else 'N/A'}
"""
            st.download_button(
                label="📄 TXT report",
                data=text_report,
                file_name=f"report_{user_input}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )
    
    # ============================================================
    # MODE 2: STRUCTURAL OPTIMIZATION
    # ============================================================
    elif analysis_mode == "🔧 Structural optimization":
        result = run_optimization(user_input, target_property)
        
        st.session_state.history.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "input": user_input,
            "mode": analysis_mode,
            "property": "optimization",
            "elapsed": result.get("_elapsed", 0),
            "success": result.get("_success", False),
        })
        
        st.markdown("### 🔧 Optimization Result")
        
        if result.get("_success"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⏱️ Time", f"{result['_elapsed']:.1f}s")
            with col2:
                st.metric("🧬 Analogues found", len(result.get("analogues", [])))
            with col3:
                st.metric("✏️ Modifications", len(result.get("modified_smiles", [])))
            
            # PubChem analogues
            st.markdown("#### 🧬 Real PubChem Analogues")
            for i, ana in enumerate(result.get("analogues", [])[:5], 1):
                with st.container():
                    st.markdown(
                        f"**[{i}] CID {ana.get('cid')}** — "
                        f"MW: {ana.get('molecular_weight')} Da | "
                        f"XLogP: {ana.get('xlogp')} | "
                        f"TPSA: {ana.get('tpsa')} Å²"
                    )
                    st.code(ana.get('smiles'), language="text")
            
            # LLM analysis
            st.markdown("#### 🤖 LLM Analysis")
            st.markdown('<div class="optimization-box">', unsafe_allow_html=True)
            st.markdown(result.get("analysis_text", ""))
            st.markdown('</div>', unsafe_allow_html=True)
            
            # SAscore
            if result.get("sa_results"):
                st.markdown("#### ⚗️ Synthetic Accessibility Assessment (SAscore)")
                for sa in result["sa_results"]:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**Modification {sa['index']}** — SMILES:")
                        st.code(sa['smiles'], language="text")
                    with col2:
                        st.markdown(f"**SAscore: {sa['sa_score']}**")
                        st.markdown(sa['interpretation'])
                        if sa['sa_score'] <= 3:
                            st.success("✅ Easy to synthesize")
                        elif sa['sa_score'] <= 6:
                            st.warning("⚠️ Moderate accessibility")
                        else:
                            st.error("❌ Difficult to synthesize")
        else:
            st.error(f"❌ Error: {result.get('_error', 'Unknown error')}")
    
    # ============================================================
    # MODE 3: UNCERTAINTY QUANTIFICATION
    # ============================================================
    else:  # UQ
        result = run_uq_prediction(user_input, confidence_level)
        
        st.session_state.history.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "input": user_input,
            "mode": analysis_mode,
            "property": "solubility_uq",
            "elapsed": result.get("_elapsed", 0),
            "success": result.get("_success", False),
        })
        
        st.markdown("### 📈 Prediction with Uncertainty")
        
        if result.get("_success"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⏱️ Time", f"{result['_elapsed']:.1f}s")
            with col2:
                st.metric("🎯 Predicted value", f"{result['predicted_logS']} logS")
            with col3:
                st.metric("📊 CI width", f"{result['interval_width']} logS")
            
            st.markdown('<div class="uq-box">', unsafe_allow_html=True)
            st.markdown(f"#### Aqueous Solubility Prediction")
            st.markdown(f"**Predicted value:** {result['predicted_logS']} logS")
            st.markdown(f"**Confidence interval at {result['coverage_level']}:** "
                       f"[{result['interval_lower']}, {result['interval_upper']}] logS")
            st.markdown(f"**Uncertainty:** ±{result['interval_width']/2:.2f} logS")
            st.markdown(f"**Interpretation:** The solubility is estimated at "
                       f"{result['predicted_logS']} logS with an uncertainty of "
                       f"±{result['interval_width']/2:.2f} logS.")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Molecular descriptors
            if "descriptors" in result:
                st.markdown("#### 📋 Molecular Descriptors Used")
                
                desc = result["descriptors"]
                
                desc_display = {
                    "Molecular weight (Da)": desc.get }
