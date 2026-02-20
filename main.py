import os
from dotenv import load_dotenv

from config import VECTOR_DIMENSION
from clients import initialize_clients
from graph_extraction import extract_graph_components
from ingestion import create_collection, ingest_to_neo4j, ingest_to_qdrant
from retrieval import retriever_search, fetch_related_graph, format_graph_context
from graphrag import graphRAG_run
from utils import check_data_exists

if __name__ == "__main__":
    print("Скрипт запущен")

    # Инициализация
    print("Загрузка переменных окружения и инициализация клиентов...")
    load_dotenv('.env.example')
    neo4j_driver, qdrant_client, collection_name = initialize_clients()

    # Проверка на говно
    print(f"neo4j_driver - {neo4j_driver}")
    print(f"neo4j_driver - {qdrant_client}")
    print(f"neo4j_driver - {collection_name}")

    print("Клиент и драйвер инициализированы")

    print("Создание коллекции...")
    create_collection(qdrant_client, collection_name, VECTOR_DIMENSION)

    # Проверка на говно
    print(f'create_collection - {create_collection}')

    print("Коллекция создана/проверена")

    # Проверка наличия данных
    if check_data_exists(neo4j_driver, qdrant_client, collection_name):
        print("Данные уже существуют в Neo4j и Qdrant. Пропускаем загрузку.")
    else:
        print("Данные не найдены. Загружаем из файла...")
        data_file = os.getenv("DATA_FILE", "data.txt")
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                raw_data = f.read()
        except FileNotFoundError:
            print(f"Error: Data file '{data_file}' not found.")
            exit(1)

        # Проверка на говно
        print(f'check_data_exists - {check_data_exists}')


        print("Извлечение компонентов графа...")
        nodes, relationships = extract_graph_components(raw_data)

        # Проверка на говно
        print(f'ноды -{nodes}')
        print(f'связи- {relationships}')

        print(f"Узлов: {len(nodes)}, отношений: {len(relationships)}")

        print("Загрузка в Neo4j...")
        node_id_mapping = ingest_to_neo4j(neo4j_driver, nodes, relationships)
        print("Загрузка в Neo4j успешна")

        print("Загрузка в Qdrant...")
        ingest_to_qdrant(qdrant_client, collection_name, raw_data, node_id_mapping)
        print("Загрузка в Qdrant успешна")



    # Поиск и ответ
    query = "Какие даты привязаны к каким ангелам и за что эти ангелы отвечают. Также подробно расскажи обо всем, что связано с ангелами. Кто такой розовый бегемот?"


    print("Начало поиска с помощью Ретривера...")
    retriever_result = retriever_search(neo4j_driver, qdrant_client, collection_name, query)
    print("Результаты поиска с помощью Ретривера:", retriever_result)

    # Извлечение идентификаторов сущностей (упрощённый парсинг)
    entity_ids = []
    for item in retriever_result.items:
        # Более надёжный способ: если результат содержит структуру с id, используем её
        # В текущей версии это строка, поэтому оставляем как есть
        try:
            # Пример: извлекаем id из строкового представления
            import re
            match = re.search(r"'id': '([^']+)'", item.content)
            if match:
                entity_ids.append(match.group(1))
        except:
            pass

    print("Entity IDs:", entity_ids)

    print("Получение связанного графа...")
    subgraph = fetch_related_graph(neo4j_driver, entity_ids)
    print("Подграф (количество записей):", len(subgraph))

    print("Форматирование контекста графа...")
    graph_context = format_graph_context(subgraph)
    print("Контекст графа:", graph_context)

    print("Запуск GraphRAG...")
    answer = graphRAG_run(graph_context, query)
    print("Финальный ответ:", answer)

    # Закрытие драйвера Neo4j (опционально)
    neo4j_driver.close()