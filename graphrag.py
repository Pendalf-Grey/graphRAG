from config import graphrag_query

def graphRAG_run(graph_context, user_query, stream=False):
    """Run RAG with the graph context."""
    return graphrag_query(graph_context, user_query, stream=stream)


if __name__ == "__main__":
    # тест без реального вызова модели (просто проверка импорта)
    context = {"nodes": [], "edges": []}
    query = "тест"
    print("Вызов graphRAG_run (может занять время)...")
    answer = graphRAG_run(context, query, stream=False)
    print(answer)