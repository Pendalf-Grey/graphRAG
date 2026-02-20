from neo4j_graphrag.retrievers import QdrantNeo4jRetriever
from config import embeddings


# Поиск
def retriever_search(neo4j_driver, qdrant_client, collection_name, query):
    """Search for relevant nodes and relationships."""
    retriever = QdrantNeo4jRetriever(
        driver=neo4j_driver,
        client=qdrant_client,
        collection_name=collection_name,
        id_property_external="id",
        id_property_neo4j="id",
    )
    results = retriever.search(query_vector=embeddings(query), top_k=10)
    return results


# Получение подграфа
def fetch_related_graph(neo4j_driver, entity_ids):
    """Fetch the related graph from Neo4j for the given entity IDs."""
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


# Форматирование контекста
def format_graph_context(subgraph):
    """Format the subgraph as a context for RAG."""
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


