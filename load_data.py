import os
import sys
from dotenv import load_dotenv

from config import VECTOR_DIMENSION
from clients import initialize_clients
from graph_extraction import extract_graph_components
from ingestion import create_collection, ingest_to_neo4j, ingest_to_qdrant
from utils import check_data_exists

def main(force_reload=False):
    print("Скрипт загрузки данных запущен")
    load_dotenv('.env.example')
    neo4j_driver, qdrant_client, collection_name = initialize_clients()
    print("Клиент и драйвер инициализированы")

    create_collection(qdrant_client, collection_name, VECTOR_DIMENSION)
    print("Коллекция создана/проверена")

    if not force_reload and check_data_exists(neo4j_driver, qdrant_client, collection_name):
        print("Данные уже существуют в Neo4j и Qdrant. Для перезагрузки используйте --force")
        neo4j_driver.close()
        return

    print("Загружаем данные из файла...")
    data_file = os.getenv("DATA_FILE", "data.txt")
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            raw_data = f.read()
    except FileNotFoundError:
        print(f"Error: Data file '{data_file}' not found.")
        sys.exit(1)

    print("Извлечение компонентов графа...")
    nodes, relationships = extract_graph_components(raw_data)
    print(f"Узлов: {len(nodes)}, отношений: {len(relationships)}")

    print("Загрузка в Neo4j...")
    node_id_mapping = ingest_to_neo4j(neo4j_driver, nodes, relationships)
    print("Загрузка в Neo4j успешна")

    print("Загрузка в Qdrant...")
    ingest_to_qdrant(qdrant_client, collection_name, raw_data, node_id_mapping)
    print("Загрузка в Qdrant успешна")

    neo4j_driver.close()
    print("Загрузка завершена.")

if __name__ == "__main__":
    force = "--force" in sys.argv
    main(force_reload=force)