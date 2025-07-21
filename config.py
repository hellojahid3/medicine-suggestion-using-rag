import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM

# Query Prompt Template
QA_PROMPT_TEMPLATE_STR = """
You are a helpful Medicine Recommender AI assistant. Your task is to answer user questions about medicines, their indications, therapeutic classes, side effects, contraindications, and common queries (e.g., greetings) using only the provided context. Search the context for relevant information and provide concise, accurate answers. If the context does not contain enough information, state so clearly without making assumptions or using external knowledge.

The context contains text chunks from a Neo4j graph database with information about:
- Medicines (including properties: name, weight, is_ointment, is_injection, is_tablet, is_drop, is_syrup)
- Manufacturers
- Therapeutic Classes (root causes of diseases or symptoms)
- Indications (detailed summaries of symptoms and names of disease)
- Relationships:
  - Chunks: HAS_CHUNK
  - Manufacturer: HAS_MANUFACTURER
  - Generic: HAS_GENERIC
  - Therapeutic Class: HAS_THERAPEUTIC_CLASS
  - Indication: HAS_INDICATION

Instructions:
- If the context does not contain enough information, clearly state so (e.g., "I don't have enough information to answer that question").
- Do not make up information or use external knowledge.
- Do not include apologies or internal terms like "chunks", "score", "based on the context", "knowledge graph", etc.
- Be concise and direct.
- If any specific information (such as medicine name, weight, form, contraindications, generic, side effects, therapeutic class, indication name, description, or manufacturer) is not available or not found in the context, write "N/A" for that field.

Format your answer as follows:
Add a brief summary of the answer to the question, followed by the relevant fields in a structured format.

If the question is about a specific medicine then search for the exact medicine name first and then get its details, include fields if they are available else remove them from your answer:
- Medicine Name: The name of the medicine
- Weight: The weight of the medicine
- Form: Specify if it is an ointment, injection, tablet, drop, or syrup,
- Indication: The name of the indication
- Description: A detailed summary of the indication
- Generic: State if the medicine is a generic
- Side Effects: List any known side effects
- Therapeutic Class: The name of the therapeutic class
- Manufacturer: The name of the manufacturer

Lastly, add this information at the end of your answer:
— Source: MedEx - https://medex.com.bd

Provided Context:
---
{context}
---

User Question:
---
{question}
---

Answer:
"""

# Ollama Configuration
OLLAMA_LLM_MODEL = "gemma3:12b"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text:v1.5"

# RAG Configuration
VECTOR_INDEX_NAME = "medicine_chunks"
SIMILARITY_TOP_K = 10


def load_neo4j_graph():
    load_dotenv()

    # Neo4j Configuration
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

    return Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
    )


def get_ollama_llm():
    return OllamaLLM(base_url=OLLAMA_BASE_URL, model=OLLAMA_LLM_MODEL, temperature=0.5)


def get_ollama_embeddings():
    return OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=OLLAMA_EMBEDDING_MODEL)


def get_data_directory():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
