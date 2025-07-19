from langchain.prompts import PromptTemplate
from KG.embeddings import get_text_embedding
from config import (
    load_neo4j_graph,
    get_ollama_llm,
    VECTOR_INDEX_NAME,
    SIMILARITY_TOP_K,
    QA_PROMPT_TEMPLATE_STR,
)

# Load Neo4j graph and Ollama LLM
graph = load_neo4j_graph()
llm = get_ollama_llm()

# Prepare the prompt template for RAG
qa_prompt = PromptTemplate(
    template=QA_PROMPT_TEMPLATE_STR, input_variables=["context", "question"]
)


def search_similar_chunks(query_text: str, top_k=SIMILARITY_TOP_K):
    query_embedding = get_text_embedding(query_text)

    # Cypher query for vector similarity search
    cypher_query = """
    CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
    YIELD node AS chunk, score
    RETURN chunk.text AS text, score
    ORDER BY score DESC
    """

    params = {
        "index_name": VECTOR_INDEX_NAME,
        "top_k": top_k,
        "query_embedding": query_embedding,
    }

    try:
        results = graph.query(cypher_query, params=params)
        return results if results else []
    except Exception as e:
        print(f"Error during vector search: {e}")
        print(
            "Ensure the vector index is created and populated, and the query syntax is correct for your Neo4j setup."
        )
        return []


def ask_question_with_rag(question: str):
    print("\nThinking...")

    # 1. Search for similar chunks
    retrieved_chunks_data = search_similar_chunks(question)

    if not retrieved_chunks_data:
        print("No relevant chunks found in the knowledge graph.")
        return "I could not find relevant information in the knowledge graph to answer your question."

    # 2. Construct context
    context_parts = []
    for item in retrieved_chunks_data:
        context_parts.append(
            f"Chunk (Similarity: {item['score']:.4f}):\n{item['text']}"
        )

    # 2a. Query context
    context_str = "\n\n---\n\n".join(context_parts)

    # 3. Query LLM with context
    qa_chain = qa_prompt | llm

    try:
        response = qa_chain.invoke({"context": context_str, "question": question})
        return response
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return "There was an error processing your question with the language model."