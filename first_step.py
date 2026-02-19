from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import QdrantNeo4jRetriever
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI
from collections import defaultdict
from neo4j_graphrag.retrievers import QdrantNeo4jRetriever
from processors.processor_factory import get_processor
from functools import lru_cache
from typing import List, Dict, Any, Optional, Union, Iterator

import os
import time
import json
import requests
import uuid


# Load environment variables
load_dotenv()

# Get credentials from environment variables
# Код из git-project
processor = get_processor()
llm_parser = processor["llm_parser"]
embeddings = processor["embeddings"]
embeddings_batch = processor["embeddings_batch"]
graphrag_query = processor["graphrag_query"]
GraphComponents = processor["GraphComponents"]
single = processor["Single"]
MODEL_PROVIDER = processor["MODEL_PROVIDER"]
VECTOR_DIMENSION = processor["VECTOR_DIMENSION"]
LLM_MODEL = processor["LLM_MODEL"]
EMBEDDING_MODEL = processor["EMBEDDING_MODEL"]


load_dotenv()

# Use the new environment variable names for Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_INFERENCE_MODEL = os.getenv("OLLAMA_INFERENCE_MODEL", "qwen2.5:14b")
# Use OLLAMA_VECTOR_DIMENSION for this provider
OLLAMA_VECTOR_DIMENSION = int(os.getenv("OLLAMA_VECTOR_DIMENSION", "768"))

data_file = os.getenv("DATA_FILE", "data.txt")

# Нету в мануале
# Красивая иництализация Neo4j driver и Qdrant client
def initialize_clients():
    """Initialize Neo4j and Qdrant clients from environment variables"""
    from dotenv import load_dotenv
    load_dotenv('.env')

    # Get credentials from environment variables
    qdrant_host = os.getenv("QDRANT_HOST")
    qdrant_port = os.getenv("QDRANT_PORT")
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    collection_name = os.getenv("COLLECTION_NAME", "graphRAGstoreds")

    # Initialize clients
    neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
    qdrant_client = QdrantClient(
        host=qdrant_host,
        port=int(qdrant_port) if qdrant_port else None
    )

    return neo4j_driver, qdrant_client, collection_name




# Функция-аналог для использования локального Qwen вместо OpeAI
# Выставлено ограничение для кэша
@lru_cache(maxsize=128)
def cached_ollama_call(prompt: str, model: Optional[str] = None) -> str:
    """
    Cached version of Ollama API call to avoid redundant calls.
    """
    if model is None:
        model = OLLAMA_INFERENCE_MODEL

    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise graph relationship extractor. Extract all relationships from the text "
                    "and format them as a JSON object with this exact structure:\n"
                    '{ "graph": [ {"node": "Person/Entity", "target_node": "Related Entity", "relationship": "Type of Relationship"}, ... ] }\n'
                    "Include ALL relationships mentioned in the text, including implicit ones. Be thorough and precise."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        response.raise_for_status()
        result = response.json()
        if 'message' in result and 'content' in result['message']:
            return result['message']['content']
        else:
            print(f"Warning: Unexpected API response format: {result}")
            return """{"graph": []}"""
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to Ollama API at {OLLAMA_BASE_URL}")
        return """{"graph": []}"""
    except Exception as e:
        print(f"Error in Ollama API call: {str(e)}")
        return """{"graph": []}"""


# Функция-аналог для использования локального Qwen вместо OpeAI
def extract_graph_components(raw_data):
    """Extract graph components from text data"""
    prompt = f"Extract nodes and relationships from the following text:\n{raw_data}"

    parsed_response = llm_parser(prompt).graph  # Assuming this returns a list of dictionaries

    nodes = {}
    relationships = []

    for entry in parsed_response:
        node = entry.node
        target_node = entry.target_node  # Get target node if available
        relationship = entry.relationship  # Get relationship if available

        # Add nodes to the dictionary with a unique ID
        if node not in nodes:
            nodes[node] = str(uuid.uuid4())

        if target_node and target_node not in nodes:
            nodes[target_node] = str(uuid.uuid4())

        # Add relationship to the relationships list with node IDs
        if target_node and relationship:
            relationships.append({
                "source": nodes[node],
                "target": nodes[target_node],
                "type": relationship
            })

    return nodes, relationships




# Почти такая же функция как в мануале
# Здесь добавили проверки и сразу батчи
def ingest_to_neo4j(neo4j_driver, nodes, relationships, batch_size=100):
    """
    Ingest nodes and relationships into Neo4j using batch operations.
    """
    with neo4j_driver.session() as session:
        # Create nodes in Neo4j in batches
        node_items = list(nodes.items())
        for i in range(0, len(node_items), batch_size):
            batch = node_items[i:i+batch_size]
            # Create batch query
            query = "UNWIND $nodes as node CREATE (n:Entity {id: node.id, name: node.name})"
            session.run(
                query,
                nodes=[{"id": node_id, "name": name} for name, node_id in batch]
            )
            print(f"Created nodes batch {i//batch_size + 1}/{(len(node_items) + batch_size - 1)//batch_size}")

        # Create relationships in Neo4j in batches
        for i in range(0, len(relationships), batch_size):
            batch = relationships[i:i+batch_size]
            # Create batch query
            query = """
            UNWIND $rels as rel
            MATCH (a:Entity {id: rel.source})
            MATCH (b:Entity {id: rel.target})
            CREATE (a)-[r:RELATIONSHIP {type: rel.type}]->(b)
            """
            session.run(
                query,
                rels=batch
            )
            print(f"Created relationships batch {i//batch_size + 1}/{(len(relationships) + batch_size - 1)//batch_size}")

    return nodes


# То же самое, что и в мануале
def create_collection(client, collection_name, vector_dimension):
    """Create a Qdrant collection if it doesn't exist"""
    # Try to fetch the collection status
    try:
        collection_info = client.get_collection(collection_name)
        print(f"Skipping creating collection; '{collection_name}' already exists.")
    except Exception as e:
        # If collection does not exist, an error will be thrown, so we create the collection
        if 'Not found: Collection' in str(e):
            print(f"Collection '{collection_name}' not found. Creating it now...")

            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_dimension, distance=models.Distance.COSINE)
            )

            print(f"Collection '{collection_name}' created successfully.")
        else:
            print(f"Error while checking collection: {e}")


# Функция-аналог для использования локального Qwen вместо OpeAI
# Добавлены проверки
def ollama_embeddings(text: str) -> List[float]:
    """
    Generate embeddings for a single text string using Ollama.
    """
    try:
        payload = {
            "model": OLLAMA_EMBEDDING_MODEL,
            "prompt": text
        }
        response = requests.post(f"{OLLAMA_BASE_URL}/api/embeddings", json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("embedding", [0] * OLLAMA_VECTOR_DIMENSION)
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return [0] * OLLAMA_VECTOR_DIMENSION


# Почти то же самое, что и в мануале
# Добавляем батчи для производительности
# Добавляем проекрки
def ingest_to_qdrant(qdrant_client, collection_name, raw_data, node_id_mapping):
    """
    Ingest data to Qdrant with optimized batching.
    """
    # Split the text into meaningful chunks (paragraphs)
    paragraphs = [p for p in raw_data.split("\n") if p.strip()]

    print(f"Generating embeddings for {len(paragraphs)} paragraphs...")
    embeddings_result = embeddings_batch(paragraphs)

    # Prepare batch points
    points = []
    node_ids = list(node_id_mapping.values())

    # Use min to handle cases where we have more paragraphs than node IDs or vice versa
    for i in range(min(len(embeddings_result), len(node_ids))):
        points.append({
            "id": str(uuid.uuid4()),
            "vector": embeddings_result[i],
            "payload": {"id": node_ids[i], "text": paragraphs[i]}
        })

    # Use batch upsert for better performance
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        qdrant_client.upsert(
            collection_name=collection_name,
            points=batch
        )
        print(f"Uploaded batch {i // batch_size + 1}/{(len(points) + batch_size - 1) // batch_size} to Qdrant")


# То же самое, что и в мануале
# Добавили выборку top_k = 10 вместо 5
def retriever_search(neo4j_driver, qdrant_client, collection_name, query):
    """Search for relevant nodes and relationships"""
    retriever = QdrantNeo4jRetriever(
        driver=neo4j_driver,
        client=qdrant_client,
        collection_name=collection_name,
        id_property_external="id",
        id_property_neo4j="id",
    )

    results = retriever.search(query_vector=embeddings(query), top_k=10)

    return results

# То же самое, что и в мануале
def fetch_related_graph(neo4j_driver, entity_ids):
    """Fetch the related graph from Neo4j for the given entity IDs"""
    query = """
    MATCH (e:Entity)-[r1]-(n1)-[r2]-(n2)
    WHERE e.id IN $entity_ids
    RETURN e, r1 as r, n1 as related, r2, n2
    UNION
    MATCH (e:Entity)-[r]-(related)
    WHERE e.id IN $entity_ids
    RETURN e, r, related, null as r2, null as n2
    """
    with neo4j_driver.session() as session:
        result = session.run(query, entity_ids=entity_ids)
        subgraph = []
        for record in result:
            subgraph.append({
                "entity": record["e"],
                "relationship": record["r"],
                "related_node": record["related"]
            })
            if record["r2"] and record["n2"]:
                subgraph.append({
                    "entity": record["related"],
                    "relationship": record["r2"],
                    "related_node": record["n2"]
                })
    return subgraph


# То же самое, что и в мануале
def format_graph_context(subgraph):
    """Format the subgraph as a context for RAG"""
    nodes = set()
    edges = []

    for entry in subgraph:
        entity = entry["entity"]
        related = entry["related_node"]
        relationship = entry["relationship"]

        nodes.add(entity["name"])
        nodes.add(related["name"])

        edges.append(f"{entity['name']} {relationship['type']} {related['name']}")

    return {"nodes": list(nodes), "edges": edges}


# Не так как в мануале
# Предполагается некое потоковое возвращение ответа
# Пока не буду трогать
def graphRAG_run(graph_context, user_query, stream=False):
    """
    Run RAG with the graph context

    Args:
        graph_context: The graph context with nodes and edges
        user_query: The user's question
        stream: Whether to stream the response (default True)

    Returns:
        If stream=True: Iterator yielding chunks of the response
        If stream=False: Complete response message
    """
    return graphrag_query(graph_context, user_query, stream=stream)


# Нет в мануале
# Удаляет все данные из Neo4j и указанной коллекции Qdrant
def clear_data(neo4j_driver, qdrant_client, collection_name):
    """
    Clear all data from Neo4j and the specified Qdrant collection.
    """
    # Clear Neo4j data
    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

    # Clear Qdrant collection - recreate it
    try:
        qdrant_client.delete_collection(collection_name)
        print(f"Collection '{collection_name}' deleted successfully.")

        # Recreate the empty collection
        vector_dimension = VECTOR_DIMENSION  # Using dimension from environment variables
        create_collection(qdrant_client, collection_name, vector_dimension)

    except Exception as e:
        print(f"Error clearing Qdrant collection: {str(e)}")

    return True


def check_data_exists(neo4j_driver, qdrant_client, collection_name):
    """Check if data already exists in Neo4j and Qdrant."""
    # Check Neo4j
    with neo4j_driver.session() as session:
        result = session.run("MATCH (n) RETURN count(n) as count")
        neo4j_count = result.single()["count"]

    # Check Qdrant
    try:
        collection_info = qdrant_client.get_collection(collection_name)
        qdrant_count = collection_info.points_count
    except Exception:
        qdrant_count = 0

    return neo4j_count > 0 and qdrant_count > 0



if __name__ == "__main__":
    print("Скрипт запущен")

    print("Загрузка переменных окружения и инициализация клиентов...")
    load_dotenv('.env.example')
    print("Переменные среды загружены")

    print("Инициализация клиента Qdrant и драйвера Neo4j...")
    neo4j_driver, qdrant_client, collection_name = initialize_clients()
    print("Клиент и драйвер инициализированы")

    print("Создание коллекции...")
    vector_dimension = VECTOR_DIMENSION
    create_collection(qdrant_client, collection_name, vector_dimension)
    print("Коллекция создана/проверена")

    # Проверяем наличие данных
    if check_data_exists(neo4j_driver, qdrant_client, collection_name):
        print("Данные уже существуют в Neo4j и Qdrant. Пропускаем загрузку.")
    else:
        print("Данные не найдены или неполные. Загружаем из файла...")

        # data_file = os.getenv("DATA_FILE", "data.txt")
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                raw_data = f.read()
        except FileNotFoundError:
            print(f"Error: Data file '{data_file}' not found.")
            exit(1)

        print("Извлечение компонентов графа...")
        nodes, relationships = extract_graph_components(raw_data)  # передаём содержимое, а не имя файла
        print("Узлы:", nodes)
        print("Отношения:", relationships)

        print("Загрузка в Neo4j...")
        node_id_mapping = ingest_to_neo4j(neo4j_driver, nodes, relationships)
        print("Загрузка в Neo4j успешна")

        print("Загрузка в Qdrant...")
        ingest_to_qdrant(qdrant_client, collection_name, raw_data, node_id_mapping)
        print("Загрузка в Qdrant успешна")

    # Поиск и генерация ответа (выполняются всегда)
    query = "Какой ангел отвечает за 14.10?"
    print("Начало поиска с помощью Ретривера...")
    retriever_result = retriever_search(neo4j_driver, qdrant_client, collection_name, query)
    print("Результаты поиска с помощью Ретривера:", retriever_result)

    print("Извлечение id-шников сущностей (entity)...")
    # Внимание: это хрупкий способ парсинга, лучше использовать структуру результата,
    # но для примера оставим как есть
    entity_ids = [item.content.split("'id': '")[1].split("'")[0] for item in retriever_result.items]
    print("Entity IDs:", entity_ids)

    print("Получение связанного графа...")
    subgraph = fetch_related_graph(neo4j_driver, entity_ids)
    print("Подграф:", subgraph)

    print("Форматирование контекста графа...")
    graph_context = format_graph_context(subgraph)
    print("Контекст графа:", graph_context)

    print("Запуск GraphRAG...")
    answer = graphRAG_run(graph_context, query)
    print("Финальный ответ:", answer)