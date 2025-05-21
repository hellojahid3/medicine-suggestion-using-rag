import os
import json
from config import load_neo4j_graph, get_data_directory
from RAG.query_engine import ask_question_with_rag
from KG.chunking import split_data_from_file
from KG.kg import (
    create_structured_medicine_graph,
    ingest_chunks_with_embeddings,
    create_vector_index,
)

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


# Ingestion Process
def run_ingestion_pipeline(graph, file_paths):
    for file_path in file_paths:
        print(f"\n--- Processing file: {file_path} ---")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data_list = json.load(f)
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            continue
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {file_path}")
            continue

        if not raw_data_list:
            print(f"No data found in {file_path}")
            continue

        # Create structured nodes and relationships
        print(f"Creating structured graph from {file_path}...")
        create_structured_medicine_graph(graph=graph, medicine_data_list=raw_data_list)

        # Split data into chunks for RAG
        print(f"Splitting data from {file_path} into chunks...")
        chunks_with_metadata = split_data_from_file(file_path)

        # Ingest chunks with embeddings
        if chunks_with_metadata:
            print(f"Ingesting {len(chunks_with_metadata)} chunks with embeddings...")
            ingest_chunks_with_embeddings(
                graph=graph,
                chunks_with_metadata=chunks_with_metadata,
                medicine_node_name_field="medicine_name_ref",
            )
        else:
            print("No chunks generated to ingest.")

    # Create vector index
    print("\nCreating vector index for chunks (if it doesn't exist)...")
    create_vector_index(graph=graph)
    print("--- Ingestion pipeline completed. ---")


if __name__ == "__main__":
    # Configuration for Ingestion Pipeline
    INGEST_DATA = False

    if INGEST_DATA:
        # Data Ingestion: Specify the JSON files to ingest
        file_names = ["Medicine.json"]
        data_directory = get_data_directory()
        files_to_ingest = [
            os.path.join(data_directory, file_name) for file_name in file_names
        ]

        run_ingestion_pipeline(graph=graph_instance, file_paths=files_to_ingest)

    # Interactive RAG Querying
    while True:
        user_question = input("Ask a question (or type 'exit' to quit): ")
        if user_question.lower() == "exit":
            break

        answer = ask_question_with_rag(user_question)
        print(f"Answer: {answer}\n")
