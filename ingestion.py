import uuid
from qdrant_client import models
from config import embeddings_batch, VECTOR_DIMENSION

# Создание коллекции
def create_collection(client, collection_name, vector_dimension=VECTOR_DIMENSION):
    """Create a Qdrant collection if it doesn't exist."""
    try:
        client.get_collection(collection_name)
        print(f"Skipping creating collection; '{collection_name}' already exists.")
    except Exception as e:
        if 'Not found: Collection' in str(e):
            print(f"Collection '{collection_name}' not found. Creating it now...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_dimension, distance=models.Distance.COSINE)
            )
            print(f"Collection '{collection_name}' created successfully.")
        else:
            print(f"Error while checking collection: {e}")


# Загрузка в Neo4j
def ingest_to_neo4j(neo4j_driver, nodes, relationships, batch_size=100):
    """Ingest nodes and relationships into Neo4j using batch operations."""
    with neo4j_driver.session() as session:
        # Nodes
        node_items = list(nodes.items())
        for i in range(0, len(node_items), batch_size):
            batch = node_items[i:i+batch_size]
            query = "UNWIND $nodes as node CREATE (n:Entity {id: node.id, name: node.name})"
            session.run(
                query,
                nodes=[{"id": node_id, "name": name} for name, node_id in batch]
            )
            print(f"Created nodes batch {i//batch_size + 1}/{(len(node_items) + batch_size - 1)//batch_size}")

        # Relationships
        for i in range(0, len(relationships), batch_size):
            batch = relationships[i:i+batch_size]
            query = """
            UNWIND $rels as rel
            MATCH (a:Entity {id: rel.source})
            MATCH (b:Entity {id: rel.target})
            CREATE (a)-[r:RELATIONSHIP {type: rel.type}]->(b)
            """
            session.run(query, rels=batch)
            print(f"Created relationships batch {i//batch_size + 1}/{(len(relationships) + batch_size - 1)//batch_size}")
    return nodes


# Загрузка в Qdrant
def ingest_to_qdrant(qdrant_client, collection_name, raw_data, node_id_mapping):
    """Ingest data to Qdrant with optimized batching."""
    paragraphs = [p for p in raw_data.split("\n") if p.strip()]
    print(f"Generating embeddings for {len(paragraphs)} paragraphs...")
    embeddings_result = embeddings_batch(paragraphs)

    points = []
    node_ids = list(node_id_mapping.values())

    for i in range(min(len(embeddings_result), len(node_ids))):
        points.append({
            "id": str(uuid.uuid4()),
            "vector": embeddings_result[i],
            "payload": {"id": node_ids[i], "text": paragraphs[i]}
        })

    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        qdrant_client.upsert(collection_name=collection_name, points=batch)
        print(f"Uploaded batch {i // batch_size + 1}/{(len(points) + batch_size - 1) // batch_size} to Qdrant")


if __name__ == "__main__":
    from clients import initialize_clients
    from config import VECTOR_DIMENSION

    driver, client, coll = initialize_clients()
    print("Проверка создания коллекции...")
    create_collection(client, coll, VECTOR_DIMENSION)

    # тестовые данные
    nodes = {"ТестУзел": "123"}
    rels = [{"source": "123", "target": "123", "type": "ТЕСТ"}]
    print("Загрузка в Neo4j (мини тест)...")
    ingest_to_neo4j(driver, nodes, rels)

    print("Загрузка в Qdrant (мини тест)...")
    ingest_to_qdrant(client, coll, "тестовый параграф", nodes)

    driver.close()