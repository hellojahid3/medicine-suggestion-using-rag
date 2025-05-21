import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM

# Query Prompt Template
QA_PROMPT_TEMPLATE_STR = """
You are a helpful AI assistant. Your task is to answer the user's question or query based only on the provided context retrieved from a knowledge graph.

The context consists of text chunks from documents related to medicines and its attributes such as Medicine, Generic, TherapeuticClass, Indication, Manufacturer.

Each of the medicine is also contain medicine property like weight, medicine type, manufacturer
Each of the Indication contain a full details about the symptoms or manifestations / disease / problem solution / treatment and possible prevention of the Indication.
Each of the Therapeutic Class is the main or the root casue of the disease or symptoms or manifestations.

If you able to identify the symptoms or manifestations or disease name from the user given text/question/query then start finding the root cause of the problem and mention the teatment from indication and suggestion medicine that needed to prevention or a proper treatment. You will find relattionship between the indication and medicine of the treatment. Question may contain those words for Symptoms or manifestations or disease: pain, Issue, Challenge, Obstacle, Crisis, Setback, Difficulty, Dilemma, Illness, Infection, Disorder, Condition, Sickness, Virus, Pathogen, Epidemic, Syndrome, Inflammation, Confusion, Dizziness, Fatigue, Anxiety, Depression, Memory loss, Mood swings, Insomnia, Pain, Swelling, Numbness, Tingling, Weakness, Fever, Chills, Nausea, Vomiting, Diarrhea, Cough, Shortness of breath, Blurriness (vision), Double vision, Ringing in ears (tinnitus), Loss of taste or smell, Light sensitivity, Hearing loss, Weight loss, Fatigue, Malaise, Sweating, Loss of appetite, discomfort.

If user input question or query match with any medicine name then you should start finding the medicine name and the related indication of this medicine use case. As example: Abdorin, Abdorin, Colicon, Dirin, Loperin, Loramide, Magnide, Magnito, Magnox, Magnum, Magalcon Plus, Magalrat Plus, Oxecone-MS, Pepcon Plus, Apedom, Deflux Meltab, Degut, Itopa-50, Itopid, Hemostat, Hemostop,. To quickly determine if it's a medicine or not, you can apply this method - 1. Check if it ends in common pharmaceutical suffixes (e.g., -olol, -statin, -pril, -mab, -cillin, -azole, -vir, etc. 2. Capitalized words in a list of symptoms. 3. Appears alongside words like “prescribed,” “take,” “pill,” “tablet”. 4. Paired with dosage (e.g., "Take 10 mg of…")

If the context does not contain enough information to answer the question, state that clearly (e.g., "Based on the provided context, I cannot answer this question.").

Do not make up information or use external knowledge.
Be concise and directly answer the question.

Provided Context:
---
{context}
---

User Question: {question}

Answer:
"""

# Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "llama3"
OLLAMA_EMBEDDING_MODEL = "mxbai-embed-large"

# RAG Configuration
VECTOR_INDEX_NAME = "medicine_chunks"
SIMILARITY_TOP_K = 3


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
    return OllamaLLM(base_url=OLLAMA_BASE_URL, model=OLLAMA_LLM_MODEL)


def get_ollama_embeddings():
    return OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=OLLAMA_EMBEDDING_MODEL)


def get_data_directory():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
