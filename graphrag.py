from config import graphrag_query

def graphRAG_run(graph_context, user_query, stream=False):
    """Run RAG with the graph context."""
    return graphrag_query(graph_context, user_query, stream=stream)
