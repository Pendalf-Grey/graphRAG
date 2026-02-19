project/
    ├── config.py               # конфигурация, переменные окружения, общие объекты
    ├── clients.py               # инициализация Neo4j и Qdrant клиентов
    ├── ollama_utils.py          # функции для работы с Ollama (вызов, эмбеддинги)
    ├── graph_extraction.py      # извлечение графа из текста (extract_graph_components)
    ├── ingestion.py             # загрузка данных в Neo4j и Qdrant
    ├── retrieval.py             # поиск, получение подграфа, форматирование контекста
    ├── graphrag.py              # вызов GraphRAG (graphRAG_run)
    ├── utils.py                 # вспомогательные функции (clear_data, check_data_exists)
    └── main.py                  # основной исполняемый скрипт