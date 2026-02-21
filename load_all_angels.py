import os
import glob
from dotenv import load_dotenv

from config import VECTOR_DIMENSION
from clients import initialize_clients
from graph_extraction import extract_graph_components
from ingestion import create_collection, ingest_to_neo4j, ingest_to_qdrant
from utils import check_data_exists   # можно использовать для проверки, но не обязательно

def main():
    print("Скрипт загрузки всех файлов из angels_txt")
    load_dotenv('.env.example')
    neo4j_driver, qdrant_client, collection_name = initialize_clients()
    print("Клиенты инициализированы")

    # Создаём коллекцию, если её нет
    create_collection(qdrant_client, collection_name, VECTOR_DIMENSION)
    print("Коллекция создана/проверена")

    # Собираем все .txt файлы в папке angels_txt
    file_pattern = os.path.join("angels_txt", "*.txt")
    txt_files = glob.glob(file_pattern)
    if not txt_files:
        print(f"Нет .txt файлов в {file_pattern}")
        neo4j_driver.close()
        return

    print(f"Найдено {len(txt_files)} файлов. Объединяем...")
    all_text_parts = []
    for file_path in txt_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:
                all_text_parts.append(content)

    if not all_text_parts:
        print("Все файлы пусты")
        neo4j_driver.close()
        return

    # Объединяем с разделителем (двойной перенос строки)
    raw_data = "\n\n".join(all_text_parts)
    print(f"Общий размер текста: {len(raw_data)} символов")

    # Извлекаем граф
    print("Извлечение компонентов графа...")
    nodes, relationships = extract_graph_components(raw_data)
    print(f"Узлов: {len(nodes)}, отношений: {len(relationships)}")

    # Загружаем в Neo4j
    print("Загрузка в Neo4j...")
    node_id_mapping = ingest_to_neo4j(neo4j_driver, nodes, relationships)
    print("Загрузка в Neo4j успешна")

    # Загружаем в Qdrant
    print("Загрузка в Qdrant...")
    ingest_to_qdrant(qdrant_client, collection_name, raw_data, node_id_mapping)
    print("Загрузка в Qdrant успешна")

    neo4j_driver.close()
    print("Все файлы загружены.")

if __name__ == "__main__":
    main()