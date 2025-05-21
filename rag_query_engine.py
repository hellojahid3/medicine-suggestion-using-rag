# rag_query_engine.py
from langchain.prompts import PromptTemplate
from config import (
    load_neo4j_graph,
    get_ollama_llm,
    VECTOR_INDEX_NAME,
    SIMILARITY_TOP_K
)
from KG.embeddings import get_text_embedding # For embedding the query

# Initialize once
graph = load_neo4j_graph()
llm = get_ollama_llm()


QA_PROMPT_TEMPLATE_STR = """
You are a helpful AI assistant. Your task is to answer the user's question based *only* on the provided context retrieved from a knowledge graph.
The context consists of text chunks from documents related to medicines and its attributes such as Medicine, Generic, TherapeuticClass, Indication, Manufacturer.
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
qa_prompt = PromptTemplate(template=QA_PROMPT_TEMPLATE_STR, input_variables=["context", "question"])


def search_similar_chunks(query_text: str, top_k=SIMILARITY_TOP_K):
    """
    Embeds the query and performs a vector similarity search in Neo4j.
    """
    query_embedding = get_text_embedding(query_text)

    # Ensure the graph object is the LangChain Neo4jGraph instance for this method
    # Or adapt if it's a raw driver using a different way to call vector search
    # This uses the vector search capability typically added to Neo4jGraph or via direct Cypher
    
    # Cypher query for vector similarity search
    # Note: The exact syntax might depend on how Neo4jVector or other integrations set up vector search.
    # This is a direct Cypher approach.
    cypher_query = """
    CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
    YIELD node AS chunk, score
    RETURN chunk.text AS text, score
    ORDER BY score DESC
    """
    # For older Neo4j versions or different setups, you might use:
    # `MATCH (c:Chunk) WHERE exists(c.embedding) WITH c, vector.similarity.cosine(c.embedding, $query_embedding) AS score ...`

    params = {
        "index_name": VECTOR_INDEX_NAME,
        "top_k": top_k,
        "query_embedding": query_embedding
    }
    
    try:
        results = graph.query(cypher_query, params=params)
        # results will be a list of dictionaries, e.g., [{'text': 'chunk text 1', 'score': 0.9}, ...]
        return results if results else []
    except Exception as e:
        print(f"Error during vector search: {e}")
        print("Ensure the vector index is created and populated, and the query syntax is correct for your Neo4j setup.")
        return []


def ask_question_with_rag(question: str):
    """
    Orchestrates the RAG process:
    1. Embeds the question.
    2. Searches for similar chunks in Neo4j.
    3. Constructs a prompt with retrieved context.
    4. Queries the LLM for an answer.
    """
    print(f"\nProcessing RAG for question: \"{question}\"")

    # 1. Search for similar chunks
    retrieved_chunks_data = search_similar_chunks(question)

    if not retrieved_chunks_data:
        print("No relevant chunks found in the knowledge graph.")
        # You could try a fallback here, like a keyword search or a more general graph query
        return "I could not find relevant information in the knowledge graph to answer your question."

    # 2. Construct context
    context_parts = []
    for item in retrieved_chunks_data:
        context_parts.append(f"Chunk (Similarity: {item['score']:.4f}):\n{item['text']}")
    
    context_str = "\n\n---\n\n".join(context_parts)

    # 3. Query LLM with context
    # Create a chain to pass context and question to LLM
    qa_chain = qa_prompt | llm # Using LangChain Expression Language (LCEL)
    
    try:
        response = qa_chain.invoke({"context": context_str, "question": question})
        return response
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return "There was an error processing your question with the language model."
