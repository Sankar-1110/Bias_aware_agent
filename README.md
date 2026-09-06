# Bias-Aware Agent for Fair Knowledge Retrieval

An Agentic RAG system that retrieves relevant knowledge, detects potential bias in retrieved content, and generates bias-aware responses using an LLM-driven reasoning workflow.

## Overview

The system combines **Agentic RAG**, **ReAct-based reasoning**, semantic retrieval, and bias detection to improve the fairness and reliability of knowledge retrieval.

The agent dynamically uses retrieval and bias-detection capabilities before generating the final response.

## Key Features

- **Agentic RAG** for knowledge retrieval and response generation
- **ReAct-based reasoning** for dynamic tool orchestration
- **Semantic Retrieval** using vector embeddings and ChromaDB
- **Local Embeddings** with `all-MiniLM-L6-v2`
- **Bias Detection** using a pretrained RoBERTa-based model fine-tuned on the **BABE dataset**
- **Gemini 2.5 Flash** for agent reasoning and response generation
- **LangGraph** for agent workflow and state orchestration
- **LangChain** for LLM, retrieval, prompt, and tool integration

## Architecture

```text
                         User Query
                             |
                             v
                    +------------------+
                    |  LangGraph Agent |
                    +--------+---------+
                             |
                       ReAct Reasoning
                             |
              +--------------+--------------+
              |                             |
              v                             v
       Retrieval Tool               Bias Detection
              |                             |
              v                             v
          ChromaDB                  RoBERTa Model
              |                    (BABE Dataset)
              v                             |
     Relevant News Content                  |
              |                             |
              +--------------+--------------+
                             |
                             v
                    Gemini 2.5 Flash
                             |
                             v
                     Final Response
```
# Tech Stack
``` text
Python
LangChain
LangGraph
ChromaDB
Gemini 2.5 Flash
all-MiniLM-L6-v2
RoBERTa
ReAct
```

# Project Structure:
``` text
BiasAwareAgent/
│
├── config/          # Configuration files
├── data/            # Data and knowledge sources
├── chroma/          # Vector database data
├── results/         # Evaluation/results
├── src/             # Agent and application source code
│
├── requirements.txt
├── README.md
└── .gitignore
```
