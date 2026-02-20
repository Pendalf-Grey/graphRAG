import os
from dotenv import load_dotenv
from processors.processor_factory import get_processor

# Загружаем основные переменные окружения
load_dotenv()

# Получаем объекты из processor (определены в вашем проекте)
processor = get_processor()
llm_parser = processor["llm_parser"]
embeddings = processor["embeddings"]
embeddings_batch = processor["embeddings_batch"]
graphrag_query = processor["graphrag_query"]
GraphComponents = processor["GraphComponents"]
single = processor["Single"]
MODEL_PROVIDER = processor["MODEL_PROVIDER"]
VECTOR_DIMENSION = processor["VECTOR_DIMENSION"]
LLM_MODEL = processor["LLM_MODEL"]
EMBEDDING_MODEL = processor["EMBEDDING_MODEL"]

# Настройки Ollama (могут быть переопределены в .env)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_INFERENCE_MODEL = os.getenv("OLLAMA_INFERENCE_MODEL", "qwen2.5:14b")
OLLAMA_VECTOR_DIMENSION = int(os.getenv("OLLAMA_VECTOR_DIMENSION", "768"))


