# SNS24 Messaging and Diagnosis System

## Overview
This project is a Proof of Concept (POC) machine learning application designed to optimize the initial patient screening process for national health services. By leveraging Natural Language Processing (NLP) and ensemble machine learning models, the system ingests free-text symptom descriptions from patients, performs an automated diagnostic assessment, and routes the patient to the most appropriate healthcare facility based on real-time spatial data and clinical urgency.

This project was developed to address bottlenecks in traditional phone-based health triage services (such as Portugal's SNS24), demonstrating how AI can streamline preliminary patient intake, reduce call center congestion, and ensure rapid routing for high-priority cases.

## Core Architecture & Algorithms

The system is built on a modular, hybrid architecture combining modern NLP frameworks with traditional, interpretable machine learning algorithms:

* **Symptom Extraction (NLP):** Utilizes `spaCy` (pt_core_news_sm) with Rule-Based Phrase Matching. This approach accurately tokenizes and extracts specific medical entities (symptoms) from unstructured, conversational Portuguese text.
* **Diagnostic Classification (Machine Learning):** Employs a `scikit-learn` Random Forest Classifier. This ensemble method was chosen for its robustness against overfitting on tabular data and its interpretability, allowing the system to output confidence intervals for its diagnostic predictions.
* **Clinical Triage Logic:** Implements a deterministic heuristic layer inspired by the Manchester Triage System. It maps the probabilistic outputs of the machine learning model to standard clinical urgency levels (Red, Orange, Yellow, Green, Blue).
* **Geospatial Routing:** Calculates the optimal healthcare facility using a raw Python implementation of the Haversine formula, computing the shortest surface distance between the patient's coordinates and our facility database while filtering for required medical specialties.
* **Application Interface:** A highly interactive, low-latency frontend built entirely in Python using `Streamlit`.

## Project Structure
```text
auto-triage/
├── data/                       
│   ├── symptoms_data.csv       # Synthetic binary symptom-disease matrix
│   └── hospitals.csv           # Geospatial and specialty data for regional hospitals
├── models/                     
│   └── random_forest.pkl       # Serialized predictive model
├── src/                        
│   ├── ml_trainer.py           # ML pipeline and model export logic
│   ├── nlp_extractor.py        # spaCy integration and tokenization rules
│   ├── triage_logic.py         # Urgency heuristics (Manchester Triage standard)
│   └── routing.py              # Haversine distance algorithm implementation
├── app.py                      # Streamlit frontend architecture
├── requirements.txt            # Environment dependencies
└── README.md                   # System documentation
```

## System Requirements

- Python 3.8+
- Pip package manager

## Installation & Setup

1. **Install core dependencies:** It is recommended to run this project within a virtual environment.

```bash
pip install -r requirements.txt
```

2. **Download the Portuguese NLP Model:** The `spaCy` library requires the specific Portuguese language pack for accurate tokenization.

```bash
python -m spacy download pt_core_news_sm
```

3. **Train the Initial Model:** Before running the application, you must generate the serialized machine learning model. This script reads the synthetic dataset and exports the `.pkl` file to the models directory.

```bash
python src/ml_trainer.py
```

4. **Launch the Application:** Initialize the Streamlit server to interact with the frontend.

```bash
streamlit run app.py
```
