from config import load_neo4j_graph
from RAG.query_engine import ask_question_with_rag

# Load Neo4j graph connection
graph_instance = load_neo4j_graph()

try:
    graph_instance.query("RETURN 1")
except Exception as e:
    print(
        "Failed to connect to Neo4j. Please check your configuration and ensure Neo4j is running."
    )
    print(f"Error: {e}")
    exit()


if __name__ == "__main__":
    # Interactive RAG Querying
    while True:
        user_question = input("Ask a question (or type 'exit' to quit): ")
        if user_question.lower() == "exit":
            break

        answer = ask_question_with_rag(user_question)
        print(f"Answer: {answer}\n")
