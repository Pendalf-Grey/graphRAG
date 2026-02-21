import os
import sys
import re
from dotenv import load_dotenv

from config import VECTOR_DIMENSION
from clients import initialize_clients
from retrieval import retriever_search, fetch_related_graph, format_graph_context
from graphrag import graphRAG_run

def main(query=None):
    if query is None:
        if len(sys.argv) > 1:
            query = ' '.join(sys.argv[1:])
        else:
            query = input("Введите ваш вопрос: ")

    print(f"Запрос: {query}")

    load_dotenv('.env.example')
    neo4j_driver, qdrant_client, collection_name = initialize_clients()
    print("Клиенты инициализированы")

    retriever_result = retriever_search(neo4j_driver, qdrant_client, collection_name, query)
    print("Результаты поиска получены")

    entity_ids = []
    for item in retriever_result.items:
        match = re.search(r"'id': '([^']+)'", item.content)
        if match:
            entity_ids.append(match.group(1))
    print(f"Найдено ID: {entity_ids}")

    if not entity_ids:
        print("Не удалось найти информацию по запросу.")
        neo4j_driver.close()
        return

    subgraph = fetch_related_graph(neo4j_driver, entity_ids)
    print(f"Подграф содержит {len(subgraph)} записей")

    graph_context = format_graph_context(subgraph)
    print("Контекст сформирован")

    answer = graphRAG_run(graph_context, query, stream=False)
    print("\n--- Ответ ---\n")
    print(answer)

    neo4j_driver.close()

if __name__ == "__main__":
    main()