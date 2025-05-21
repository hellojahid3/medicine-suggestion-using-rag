# main.py
import json
from KG.kg import (
    create_structured_medicine_graph,
    ingest_chunks_with_embeddings,
    create_vector_index
)
from KG.chunking import split_data_from_file
from config import load_neo4j_graph
from rag_query_engine import ask_question_with_rag # Import the RAG function

# --- Step 0: Load Neo4j graph connection ---
# This graph instance will be used by all KG operations
graph_instance = load_neo4j_graph()
try:
    graph_instance.query("RETURN 1") # Test connection
    print("Successfully connected to Neo4j.")
except Exception as e:
    print("Failed to connect to Neo4j. Please check your configuration and ensure Neo4j is running.")
    print(f"Error: {e}")
    exit()


# --- Step 1: Ingestion Process ---
def run_ingestion_pipeline(graph, file_paths):
    for file_path in file_paths:
        print(f"\n--- Processing file: {file_path} ---")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data_list = json.load(f) # Assumes a list of medicine objects
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            continue
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {file_path}")
            continue
        
        if not raw_data_list:
            print(f"No data found in {file_path}")
            continue

        # 1a. Create structured nodes and relationships (Medicines, Generics, etc.)
        print(f"Creating structured graph from {file_path}...")
        create_structured_medicine_graph(graph=graph, medicine_data_list=raw_data_list)
        
        # 1b. Split data into chunks for RAG
        print(f"Splitting data from {file_path} into chunks...")
        chunks_with_metadata = split_data_from_file(file_path) # Pass file_path directly
        
        # 1c. Ingest chunks with embeddings
        if chunks_with_metadata:
            print(f"Ingesting {len(chunks_with_metadata)} chunks with embeddings...")
            ingest_chunks_with_embeddings(
                graph=graph,
                chunks_with_metadata=chunks_with_metadata,
                medicine_node_name_field="medicine_name_ref" # Matches key in split_data_from_file
            )
        else:
            print("No chunks generated to ingest.")
            
    # 1d. Create vector index (do this after all chunks are ingested)
    print("\nCreating vector index for chunks (if it doesn't exist)...")
    create_vector_index(graph=graph)
    print("--- Ingestion pipeline completed. ---")


if __name__ == "__main__":
    # --- Configuration for Ingestion ---
    # Set to True to run the ingestion pipeline.
    # Set to False if data is already ingested and you only want to query.
    # INGEST_DATA = False
    INGEST_DATA = True

    if INGEST_DATA:
        # List of JSON files to process. Assumes they are in a 'data/' subdirectory.
        # Ensure your JSON files contain a LIST of medicine objects.
        # Example: data/medicines_a.json, data/medicines_b.json
        files_to_ingest = ['data/Medicine.json'] # Adjust as per your file structure
        
        run_ingestion_pipeline(graph=graph_instance, file_paths=files_to_ingest)
    else:
        print("Skipping data ingestion as INGEST_DATA is set to False.")

    # --- Interactive RAG Querying ---
    while True:
      user_question = input("Ask a question (or type 'exit' to quit): ")
      if user_question.lower() == 'exit':
        break

      answer = ask_question_with_rag(user_question)
      print(f"Question: {user_question}")
      print(f"RAG Answer: {answer}\n")
