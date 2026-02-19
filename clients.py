from neo4j import GraphDatabase
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

# Инициализация клиентов
def initialize_clients():
    """Initialize Neo4j and Qdrant clients from environment variables."""
    load_dotenv('.env')  # загружаем файл .env с конкретными кредами

    qdrant_host = os.getenv("QDRANT_HOST")
    qdrant_port = os.getenv("QDRANT_PORT")
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    collection_name = os.getenv("COLLECTION_NAME", "graphRAGstoreds")

    neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
    qdrant_client = QdrantClient(
        host=qdrant_host,
        port=int(qdrant_port) if qdrant_port else None
    )

    return neo4j_driver, qdrant_client, collection_name

if __name__ == "__main__":
    driver, client, coll = initialize_clients()
    print("Neo4j driver и Qdrant client инициализированы")
    print(f"Коллекция по умолчанию: {coll}")
    driver.close()  # закрываем драйвер после проверки