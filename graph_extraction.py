import json
import re
import uuid
from config import processor  # для получения Single
from processors.ollama_processor import cached_ollama_call  # путь может отличаться


def split_text_into_chunks(text, max_tokens=3000, overlap=200):
    """
    Разбивает большой текст на чанки примерно по max_tokens токенов с перекрытием.
    Используется простая эвристика: 1 токен ≈ 1.3 слова (для русского текста).
    """
    # Заменяем переносы строк пробелами и разбиваем на предложения по точке с пробелом
    sentences = text.replace('\n', ' ').split('. ')
    chunks = []
    current_chunk = []
    current_len = 0  # длина в токенах (приблизительно)

    for sent in sentences:
        # грубая оценка количества токенов в предложении
        sent_len = len(sent.split()) * 1.3
        if current_len + sent_len > max_tokens and current_chunk:
            # сохраняем текущий чанк
            chunks.append('. '.join(current_chunk) + '.')
            # оставляем перекрытие: берём несколько последних предложений
            overlap_sentences = current_chunk[-overlap:] if overlap else []
            current_chunk = overlap_sentences
            current_len = sum(len(s.split())*1.3 for s in overlap_sentences)
        current_chunk.append(sent)
        current_len += sent_len

    # последний чанк
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
    return chunks


def extract_graph_components(raw_data: str):
    """
    Извлекает узлы и отношения из большого русского текста, разбивая его на чанки,
    обрабатывая каждый чанк отдельно и объединяя результаты.
    """
    system_prompt = """Ты — программа, извлекающая граф отношений из текста.
    Твой ответ должен быть ТОЛЬКО в формате JSON без пояснений.
    Формат: {"graph": [{"node": "...", "target_node": "...", "relationship": "..."}]}
    Если ничего не найдено, верни {"graph": []}."""

    chunks = split_text_into_chunks(raw_data, max_tokens=3000, overlap=200)
    print(f"Текст разбит на {len(chunks)} чанков")
    all_graph_items = []

    for idx, chunk in enumerate(chunks):
        print(f"Обработка чанка {idx+1}/{len(chunks)}...")
        user_prompt = f"""Текст (часть документа):
        {chunk}

        Извлеки все факты об ангелах в виде графа отношений и выведи только JSON, без других слов."""
        response_text = cached_ollama_call(
            prompt=user_prompt,
            system=system_prompt,
            temperature=0.0,
            max_tokens=8000
        )

        # ОТЛАДКА: выводим первые 1000 символов ответа модели
        print(f"Ответ модели (первые 1000 символов):\n{response_text[:1000]}\n")

        # Извлечение JSON из ответа
        json_str = None
        if "```json" in response_text:
            match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if match:
                json_str = match.group(1)
        if not json_str:
            match = re.search(r'(\{.*\})', response_text, re.DOTALL)
            if match:
                json_str = match.group(1)

        if not json_str:
            print(f"Чанк {idx+1}: не удалось найти JSON, пропускаем.")
            continue

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Чанк {idx+1}: ошибка парсинга JSON: {e}")
            continue

        if "graph" not in data or not isinstance(data["graph"], list):
            print(f"Чанк {idx+1}: JSON не содержит ключ 'graph' или это не список")
            continue

        all_graph_items.extend(data["graph"])

    # ... остальная часть функции (объединение, создание nodes/relationships) без изменений
    if not all_graph_items:
        print("Не найдено ни одного элемента графа во всех чанках.")
        return {}, []

    Single = processor["Single"]
    nodes = {}
    relationships = []

    for item in all_graph_items:
        try:
            entry = Single(**item)
        except Exception as e:
            print(f"Ошибка создания Single из {item}: {e}")
            continue

        node = entry.node
        target_node = entry.target_node
        relationship = entry.relationship

        if node and node not in nodes:
            nodes[node] = str(uuid.uuid4())
        if target_node and target_node not in nodes:
            nodes[target_node] = str(uuid.uuid4())

        if target_node and relationship:
            relationships.append({
                "source": nodes[node],
                "target": nodes[target_node],
                "type": relationship
            })

    print(f"Всего извлечено узлов: {len(nodes)}, отношений: {len(relationships)}")
    return nodes, relationships