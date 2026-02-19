import json
import re
import uuid
from config import processor, single, llm_parser
from ollama_utils import cached_ollama_call


# Дефолтная функция
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



# Извлечение узлов и отношений из текста
# def extract_graph_components(raw_data: str):
#     """Extract graph components from Russian text using Ollama directly."""
#     prompt = f"""Текст:
# {raw_data}
#
# Извлеки из текста все сущности (люди, организации, места, проекты и т.д.) и отношения между ними.
# Верни ТОЛЬКО валидный JSON в следующем формате (без пояснений, только JSON):
# {{"graph": [{{"node": "имя сущности", "target_node": "имя связанной сущности", "relationship": "тип отношения"}}, ...]}}
#
# Пример:
# {{"graph": [
#     {{"node": "Алиса", "target_node": "ТехКорп", "relationship": "работает в"}},
#     {{"node": "Боб", "target_node": "Алиса", "relationship": "коллега"}}
# ]}}"""
#
#     response_text = cached_ollama_call(prompt)
#
#     # Извлечение JSON из ответа
#     json_str = None
#     if "```json" in response_text:
#         match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
#         if match:
#             json_str = match.group(1)
#     if not json_str:
#         match = re.search(r'(\{.*\})', response_text, re.DOTALL)
#         if match:
#             json_str = match.group(1)
#
#     if not json_str:
#         print("Не удалось найти JSON в ответе модели.")
#         print("Ответ модели (первые 500 символов):", response_text[:500])
#         return {}, []
#
#     try:
#         data = json.loads(json_str)
#     except json.JSONDecodeError as e:
#         print(f"Ошибка парсинга JSON: {e}")
#         print("Найденный JSON (первые 200 символов):", json_str[:200])
#         return {}, []
#
#     if "graph" not in data or not isinstance(data["graph"], list):
#         print("JSON не содержит ключ 'graph' или это не список")
#         return {}, []
#
#     Single = processor["Single"]  # получаем класс Single из processor
#
#     nodes = {}
#     relationships = []
#
#     for item in data["graph"]:
#         try:
#             entry = Single(**item)
#         except Exception as e:
#             print(f"Ошибка создания Single из {item}: {e}")
#             continue
#
#         node = entry.node
#         target_node = entry.target_node
#         relationship = entry.relationship
#
#         if node and node not in nodes:
#             nodes[node] = str(uuid.uuid4())
#         if target_node and target_node not in nodes:
#             nodes[target_node] = str(uuid.uuid4())
#
#         if target_node and relationship:
#             relationships.append({
#                 "source": nodes[node],
#                 "target": nodes[target_node],
#                 "type": relationship
#             })
#
#     return nodes, relationships


if __name__ == "__main__":
    sample = """
    Иван работает в компании Рога и Копыта. 
    Пётр — его начальник. 
    Они оба находятся в Москве.
    """
    nodes, rels = extract_graph_components(sample)
    print("Узлы:", nodes)
    print("Отношения:", rels)


