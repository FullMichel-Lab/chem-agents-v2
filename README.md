# Multi-Agent Chemical Analyzer with Uncertainty Quantification

A LangGraph-based multi-agent system for molecular property prediction,
structural optimization, and uncertainty quantification.

## Overview

This project implements a Supervisor architecture in LangGraph where a
central routing agent orchestrates four specialized agents:

- `retrieve_data` : Fetches molecular properties from PubChem
- `compute_descriptors` : Computes RDKit molecular descriptors
- `predict_property` : Predicts physicochemical properties with UQ
- `suggest_optimization` : Proposes structural modifications anchored in PubChem analogues

## Key Features

- **Multi-agent orchestration** with LangGraph Supervisor
- **PubChem-anchored optimization** : every suggestion cites a real PubChem CID
- **Synthetic accessibility scoring** (SAscore) for proposed modifications
- **Uncertainty quantification** via Conformal Prediction (MAPIE v1)
- **Full audit trail** of every agent interaction
- **Interactive Streamlit interface** with three analysis modes
- **Robustness** to invalid inputs (SMILES, non-existent molecules)

## Architecture

The system uses `openai/gpt-oss-120b` via the Groq API for reasoning
and tool calling. Uncertainty quantification is provided by a Random Forest
model trained on the ESOL dataset (1128 molecules) and calibrated with
`SplitConformalRegressor` + `ResidualNormalisedScore`.

## Installation

### 1. Clone the repository

\\\ash
git clone https://github.com/TON_USERNAME/chem-agents-v2.git
cd chem-agents-v2
\\\

### 2. Create a virtual environment

\\\ash
python -m venv venv
\\\

**Windows:**
\\\powershell
.\venv\Scripts\Activate.ps1
\\\

**Linux/macOS:**
\\\ash
source venv/bin/activate
\\\

### 3. Install dependencies

\\\ash
pip install -r requirements.txt
\\\

### 4. Configure environment variables

Copy `.env.example` to `.env` and fill in your Groq API key:

\\\
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=openai/gpt-oss-120b
\\\

Get a free API key at https://console.groq.com/keys

### 5. Train the UQ model (if not already trained)

\\\ash
python uq_model.py
\\\

This will download the ESOL dataset and train the Random Forest model
with conformal calibration. The trained model will be saved as
`uq_model.pkl`.

## Usage

### Command-line interface

\\\ash
python main.py all      # Run all 8 tests
python main.py 7        # Run a specific test
\\\

### Web interface (Streamlit)

\\\ash
streamlit run app.py
\\\

Then open http://localhost:8501 in your browser.

The interface offers three analysis modes:

1. **Standard analysis** — PubChem retrieval + RDKit + property prediction
2. **Structural optimization** — PubChem analogues + SAscore validation
3. **Uncertainty quantification** — Solubility with 90% confidence intervals

## Results

- **100% success rate** on 11 validation tests
- **UQ model performance** : MAE = 0.523 logS, R² = 0.841, coverage = 85.4%
- **Optimization** : 2 modifications proposed with SAscore = 1.21 and 2.16
- **UQ prediction** (ibuprofen) : -3.201 logS, 90% CI [-4.295, -2.106]

## Citation

If you use this work, please cite:

\\\
TEUMA MBEZI, M. (2026). Multi-Agent Chemical Analyzer with Uncertainty
Quantification. https://github.com/TON_USERNAME/chem-agents-v2
\\\

## License

MIT License — see LICENSE file for details.

## Contact

Michel TEUMA MBEZI — 3michelteuma@gmail.com
