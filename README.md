# Medicine Suggestion Using RAG

A system for recommending medicines using Retrieval-Augmented Generation (RAG).

## Overview

This project leverages RAG architecture to provide contextual medicine suggestions with knowledge graph.

## Requirements

- **Neo4J**: Graph database for storing medicine knowledge
    - Version: 5+
    - URL: [https://neo4j.com/download/](https://neo4j.com/download/)
- **Ollama**: For running LLMs locally
    - **Models**:
        - [`phi4`](https://ollama.com/library/phi4) or [`llama3.2`](https://ollama.com/library/llama3.2)
    - **Embedding Model**:
        - [`mxbai-embed-large`](https://ollama.com/library/mxbai-embed-large)
- **Python**: For application development
    - Version: 3.9+
    - URL: [python.org/downloads](https://www.python.org/downloads/)

## Setup

### 1. Create virtual environment

```bash
python -m venv .venv
```

### 2. Install all required dependencies

```bash
pip install -r requirements.txt
```

## Usage

Run the application:

```bash
python main.py
```
## License

This project is unlicensed and free to use.

## Contributors

- Imon Hossain
- Jahid Hasan Jewel
- Rakib Hosen
- Rajib Ahamed
- Natasha Chowdhury
