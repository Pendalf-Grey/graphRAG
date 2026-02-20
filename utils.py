from config import VECTOR_DIMENSION
from ingestion import create_collection

# Очистка данных
def clear_data(neo4j_driver, qdrant_client, collection_name):
    """Clear all data from Neo4j and the specified Qdrant collection."""
    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    try:
        qdrant_client.delete_collection(collection_name)
        print(f"Collection '{collection_name}' deleted successfully.")
        create_collection(qdrant_client, collection_name, VECTOR_DIMENSION)
    except Exception as e:
        print(f"Error clearing Qdrant collection: {str(e)}")
    return True


# Проверка наличия данных
def check_data_exists(neo4j_driver, qdrant_client, collection_name):
    """Check if data already exists in Neo4j and Qdrant."""
    with neo4j_driver.session() as session:
        result = session.run("MATCH (n) RETURN count(n) as count")
        neo4j_count = result.single()["count"]
    try:
        collection_info = qdrant_client.get_collection(collection_name)
        qdrant_count = collection_info.points_count
    except Exception:
        qdrant_count = 0
    return neo4j_count > 0 and qdrant_count > 0

