from KG.embeddings import get_text_embedding
from config import VECTOR_INDEX_NAME


def create_structured_medicine_graph(graph, medicine_data_list):
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

        if manufacturer_name:
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
        indication_name_str = medicine.get("Indications")
        if indication_name_str:
            primary_indication = indication_name_str.strip()

            if primary_indication:
                indication_query = """
                MERGE (i:Indication {name: $indication_name})
                ON CREATE SET i.details_summary = $details_summary
                ON MATCH SET i.details_summary = $details_summary
                WITH i
                MATCH (m:Medicine {name: $medicine_name})
                MERGE (m)-[:INDICATED_FOR]->(i)
                """

                graph.query(
                    indication_query,
                    params={
                        "indication_name": primary_indication,
                        "details_summary": (
                            medicine.get("Indications Details", "")
                            if medicine.get("Indications Details")
                            else ""
                        ),
                        "medicine_name": medicine_name,
                    },
                )
        print(f"Created structured graph for: {medicine_name}")


def ingest_chunks_with_embeddings(
    graph, chunks_with_metadata, medicine_node_name_field="medicine"
):
    for chunk_info in chunks_with_metadata:
        text_to_embed = chunk_info["text"]
        embedding_vector = get_text_embedding(text_to_embed)

        params = {
            "chunkId": chunk_info["chunkId"],
            "text": text_to_embed,
            "medicine_name": chunk_info.get(medicine_node_name_field),
            "chunkSeqId": chunk_info["chunkSeqId"],
            "source": chunk_info.get("source", "unknown"),
            "embedding": embedding_vector,
        }

        # Cypher query to create Chunk node and link it to the Medicine node
        cypher_query = """
        MATCH (m:Medicine {name: $medicine_name})
        MERGE (c:Chunk {chunkId: $chunkId})
        ON CREATE SET c.text = $text,
                c.chunkSeqId = $chunkSeqId,
                c.source = $source,
                c.embedding = $embedding
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
    else:
        print(f"Vector index '{index_name}' already exists.")
