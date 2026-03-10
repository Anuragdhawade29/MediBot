# MediBot: RAG-Based Medical Assistant

## Assessment Information
Full Name: Anurag Rajesh Dhawade  
Email ID: anuragdhawade09@gmail.com  
College Name: G.H. Raisoni College of Engineering 
Selected Skill Track: AI & Machine Learning

MediBot is a Streamlit medical assistant built on Retrieval-Augmented Generation (RAG).  
It uses a local FAISS vector store from medical PDFs and answers with context-grounded responses.

## Overview

This project includes:

- PDF ingestion and FAISS index creation
- Quick Checkup mode for symptom-based triage
- Adaptive follow-up questions (yes/no flow)
- Risk scoring with severity categories
- Differential diagnosis suggestions
- Recommended lab/test suggestions
- Clinical Query mode for open medical questions

## Project Structure

| File | Purpose |
|---|---|
| `create_memory_for_llm.py` | Loads PDFs from `data/`, chunks text, creates embeddings, saves FAISS index |
| `connect_memory_with_llm.py` | CLI retrieval + LLM query over FAISS memory |
| `medibot.py` | Main Streamlit app with Quick Checkup + Clinical Query |
| `vectorstore/db_faiss/` | Persisted FAISS index (`index.faiss`, `index.pkl`) |
| `data/` | Source medical PDFs |

## Core Features

### 1) Quick Checkup
- Symptom selection
- Age and temperature input
- Past history selection (BP, Diabetes, Heart disease, etc.)
- Current medicines and additional notes
- Adaptive follow-up questions
- Structured triage output:
  - Risk Score Summary
  - Possible Conditions
  - Recommended Tests
  - Clinical guidance from RAG

### 2) Risk Scoring
Current scoring factors include:
- Fever > 101 F
- Breathing difficulty
- Age > 60
- Diabetes
- Heart disease
- Follow-up severity signals (for example, breathing worse while lying down)

Risk levels:
- `0-3`: Mild
- `4-6`: Moderate
- `7+`: High risk

### 3) Clinical Query
- Free-text medical Q&A
- Category-based suggestion groups
- Dynamic suggestions from recent quick-check symptoms
- Answers grounded in retrieved medical context

## Environment Variables

Create a `.env` file in project root:

```env
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_hf_token
```

Notes:
- `medibot.py` uses `GROQ_API_KEY` for LLM generation.
- `HF_TOKEN` may be used in related scripts depending on endpoint usage.

## Installation

```powershell
cd C:\Users\ASUS\OneDrive\Documents\Work\Imp\medical-chatbot-refactored
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Build Vector Memory (One-Time / When Data Changes)

```powershell
.\.venv\Scripts\python.exe create_memory_for_llm.py
```

This creates/updates:
- `vectorstore/db_faiss/index.faiss`
- `vectorstore/db_faiss/index.pkl`

## Run the App

```powershell
.\.venv\Scripts\python.exe -m streamlit run medibot.py --server.port 8503
```

Open:
- `http://localhost:8503`

## Run CLI Version

```powershell
.\.venv\Scripts\python.exe connect_memory_with_llm.py
```

## Troubleshooting

### Invalid Groq key (401 / invalid_api_key)
- Ensure `.env` has valid `GROQ_API_KEY`
- Remove extra quotes/spaces around key
- Restart Streamlit after updating `.env`

### Missing FAISS index
- Run `create_memory_for_llm.py` first

