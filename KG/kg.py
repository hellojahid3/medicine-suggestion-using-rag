from KG.embeddings import get_text_embedding
from config import VECTOR_INDEX_NAME


def create_structured_medicine_graph(graph, medicine_data_list):
  """
  Creates structured nodes (Medicine, Generic, TherapeuticClass, Indication, Manufacturer)
  and their relationships for each medicine in the list.
  """
  for medicine in medicine_data_list:
    medicine_name = medicine.get("Medicine Name")
    if not medicine_name:
      print(f"Skipping medicine due to missing 'Medicine Name': {medicine}")
      continue

    # 1. Create/Merge Manufacturer Node (if it doesn't exist)
    manufacturer_name = medicine.get("Manufacturer")
    if manufacturer_name:
      graph.query(
        "MERGE (mf:Manufacturer {name: $name})",
        params={"name": manufacturer_name},
      )

    # 2. Create/Merge Main Medicine Node and link to Manufacturer
    main_node_query = """
    MERGE (m:Medicine {name: $name})
    SET m.weight_mg = $weight_mg,
      m.weight_ml_other = $weight_ml_other,
      m.is_tablet = $is_tablet,
      m.is_syrup = $is_syrup,
      m.is_ointment = $is_ointment,
      m.is_drop = $is_drop,
      m.is_injection = $is_injection
    """
    params = {
      "name": medicine_name,
      "weight_mg": medicine.get("Weight (mg)"),
      "weight_ml_other": medicine.get("Weight (ml/other)"),
      "is_tablet": True if medicine.get("Tablet") == 1 else False,
      "is_syrup": True if medicine.get("Syrup") == 1 else False,
      "is_ointment": True if medicine.get("Ointment") == 1 else False,
      "is_drop": True if medicine.get("Drop") == 1 else False,
      "is_injection": True if medicine.get("Injection") == 1 else False,
    }
    if manufacturer_name:  # If manufacturer exists, add relationship
      main_node_query += """
      WITH m
      MATCH (mf:Manufacturer {name: $manufacturer_name})
      MERGE (mf)-[:PRODUCES]->(m)
      """
      params["manufacturer_name"] = manufacturer_name

    graph.query(main_node_query, params=params)

    # 3. Create Generic node and relationship to Medicine
    generic_name = medicine.get("Generics indicated")
    if generic_name:
      graph.query(
        """
        MERGE (g:Generic {name: $generic_name})
        WITH g
        MATCH (m:Medicine {name: $medicine_name})
        MERGE (m)-[:HAS_GENERIC]->(g)
        """,
        params={"generic_name": generic_name, "medicine_name": medicine_name},
      )

    # 4. Create Therapeutic Class node and relationship
    class_name = medicine.get("Therapeutic Class")
    if class_name:
      graph.query(
        """
        MERGE (t:TherapeuticClass {name: $class_name})
        WITH t
        MATCH (m:Medicine {name: $medicine_name})
        MERGE (m)-[:BELONGS_TO_CLASS]->(t)
        """,
        params={"class_name": class_name, "medicine_name": medicine_name},
      )

    # 5. Create Indication nodes and relationships
    # The main "Indications" string is one indication name, details are in "Indications Details"
    # This "Indications Details" is part of the text that gets chunked.
    indication_name_str = medicine.get("Indications")
    if indication_name_str:
      # Assuming "Indications" field is a single primary indication name.
      # If it can be multiple comma-separated, you'll need to split it.
      # For this example, let's treat it as one primary indication for structured linking.
      # The full details are in the text chunks.
      primary_indication = indication_name_str.strip()
      if primary_indication:
        indication_query = """
          MERGE (i:Indication {name: $indication_name})
          ON CREATE SET i.details_summary = $details_summary 
          ON MATCH SET i.details_summary = $details_summary // Update if needed
          WITH i
          MATCH (m:Medicine {name: $medicine_name})
          MERGE (m)-[:INDICATED_FOR]->(i)
        """
        graph.query(
          indication_query,
          params={
            "indication_name": primary_indication,
            "details_summary": (
              medicine.get("Indications Details", "")[:250] + "..."
              if medicine.get("Indications Details")
              else ""
            ),  # Store a snippet
            "medicine_name": medicine_name,
          },
        )
    print(f"Created structured graph for: {medicine_name}")


def ingest_chunks_with_embeddings(
  graph, chunks_with_metadata, medicine_node_name_field="medicine"
):
  """
  Ingests chunks, generates embeddings, stores them, and links chunks to parent Medicine nodes.
  """
  for chunk_info in chunks_with_metadata:
    text_to_embed = chunk_info["text"]
    embedding_vector = get_text_embedding(text_to_embed)  # Generate embedding

    params = {
      "chunkId": chunk_info["chunkId"],
      "text": text_to_embed,
      "medicine_name": chunk_info.get(
        medicine_node_name_field
      ),  # e.g., "Abdorin"
      "chunkSeqId": chunk_info["chunkSeqId"],
      "source": chunk_info.get("source", "unknown"),
      "embedding": embedding_vector,  # Store the embedding vector
    }

    # Cypher query to create Chunk node and link it to the Medicine node
    # Ensure the Medicine node it links to has a unique identifier, e.g., its name
    cypher_query = """
    MATCH (m:Medicine {name: $medicine_name}) // Ensure this Medicine node exists
    MERGE (c:Chunk {chunkId: $chunkId})
    ON CREATE SET c.text = $text,
            c.chunkSeqId = $chunkSeqId,
            c.source = $source,
            c.embedding = $embedding  // Store the embedding property
    MERGE (m)-[:HAS_CHUNK]->(c)
    RETURN c.chunkId as chunkNodeId
    """
    result = graph.query(cypher_query, params=params)
    if result and result[0].get("chunkNodeId") is not None:
      print(
        f"Ingested chunk {chunk_info['chunkId']} for {chunk_info.get(medicine_node_name_field)} with embedding."
      )
    else:
      print(
        f"Failed to ingest or link chunk {chunk_info['chunkId']} for {chunk_info.get(medicine_node_name_field)}. Medicine node might be missing."
      )


def create_vector_index(
  graph, index_name=VECTOR_INDEX_NAME, node_label="Chunk", property_name="embedding"
):
  """
  Creates a vector index in Neo4j for the specified node label and property.
  """
  # Check if index exists
  index_exists_query = "SHOW INDEXES WHERE name = $index_name"
  existing_indexes = graph.query(
    index_exists_query, params={"index_name": index_name}
  )

  if not existing_indexes or not any(
    idx["name"] == index_name for idx in existing_indexes
  ):
    query = f"""
    CREATE VECTOR INDEX `{index_name}` IF NOT EXISTS
    FOR (c:{node_label}) ON (c.{property_name})
    OPTIONS {{indexConfig: {{
      `vector.similarity_function`: 'cosine'
    }}}}
    """
    try:
      graph.query(query)
      print(
        f"Vector index '{index_name}' created successfully or already exists."
      )
    except Exception as e:
      print(f"Error creating vector index '{index_name}': {e}")
      print(
        "Please ensure your Neo4j version supports vector indexes (5.11+ for basic, 5.13+ for CREATE VECTOR INDEX)."
      )
      print(
        "For older versions or Aura, you might need different syntax or APOC procedures."
      )
  else:
    print(f"Vector index '{index_name}' already exists.")
