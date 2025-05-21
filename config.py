import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM

# --- Ollama Configuration ---
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "llama3"
OLLAMA_EMBEDDING_MODEL = "mxbai-embed-large"

# --- RAG Configuration ---
VECTOR_INDEX_NAME = "medicine_chunks"
SIMILARITY_TOP_K = 3


def load_neo4j_graph():
    """Loads the LangChain Neo4jGraph instance."""

    load_dotenv()

    # --- Neo4j Configuration ---
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
    """Initializes and returns the Ollama LLM for generation."""
    return OllamaLLM(base_url=OLLAMA_BASE_URL, model=OLLAMA_LLM_MODEL)


def get_ollama_embeddings():
    """Initializes and returns the Ollama Embeddings model."""
    return OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=OLLAMA_EMBEDDING_MODEL)
