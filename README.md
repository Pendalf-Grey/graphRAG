# GraphRAG с Neo4j, Qdrant и Ollama

Проект для построения графа знаний из текстовых документов (ангелы, даты, покровительство) с последующим вопросно-ответным взаимодействием через Telegram-бота или командную строку. Используются локальные модели: **Qwen2.5:14b** для извлечения графов и генерации ответов, **nomic-embed-text** для эмбеддингов, **Neo4j** для хранения графа, **Qdrant** для векторного поиска.

## Возможности

- Загрузка большого количества текстовых файлов (в формате `.txt`) в единую базу знаний.
- Автоматическое извлечение сущностей и отношений с помощью LLM (Qwen2.5:14b) и разбиение на чанки.
- Хранение узлов и связей в Neo4j.
- Векторное индексирование текстовых чанков в Qdrant для семантического поиска.
- Гибридный поиск (векторный + графовый) с помощью `QdrantNeo4jRetriever`.
- Два режима работы:
  - **Загрузка данных** (скрипт `load_data.py` или `load_all_angels.py` для папки с файлами).
  - **Запросы** (скрипт `ask.py` или Telegram-бот `bot.py`).
- Настраиваемый системный промпт для ответов (строгое следование правилам поиска по датам и характеристикам).
- Возможность параллельной обработки чанков для ускорения.

## Требования

- Python 3.10+
- Docker и Docker Compose (для запуска Neo4j и Qdrant)
- Установленная [Ollama](https://ollama.com/) с моделями:
  - `qwen2.5:14b` (или другая совместимая)
  - `nomic-embed-text`
- (Опционально) Telegram Bot Token для запуска бота

## Быстрый старт

### 1. Клонирование репозитория

```bash
git clone <your-repo-url>
cd graphRAG
```

### 2. Настройка окружения

Создайте файл `.env` на основе `.env.example` и отредактируйте под себя:

```bash
cp .env.example .env
nano .env
```

Основные параметры:

```ini
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

QDRANT_HOST=localhost
QDRANT_PORT=6333

COLLECTION_NAME=graphRAGstoreds

DEFAULT_MODEL_PROVIDER=ollama

OLLAMA_HOST=localhost
OLLAMA_PORT=11434
OLLAMA_INFERENCE_MODEL=qwen2.5:14b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_VECTOR_DIMENSION=768

DATA_FILE=data.txt   # для одного файла
TELEGRAM_BOT_TOKEN=your_token   # если нужен бот
```

### 3. Запуск Neo4j и Qdrant через Docker Compose

Пример `docker-compose.yml`:

```yaml
version: '3.8'
services:
  neo4j:
    image: neo4j:latest
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      NEO4J_AUTH: neo4j/password
    volumes:
      - neo4j_data:/data

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  neo4j_data:
  qdrant_data:
```

Запустите:

```bash
docker-compose up -d
```

### 4. Установка Python-зависимостей

Создайте виртуальное окружение и активируйте:

```bash
python3 -m venv venv
source venv/bin/activate   # Linux/Mac
# .\venv\Scripts\activate   # Windows
```

Установите зависимости:

```bash
pip install -r requirements.txt
```

Если `requirements.txt` ещё нет, создайте его со списком:

```
neo4j
neo4j-graphrag
qdrant-client
pydantic
python-dotenv
openai
requests
python-telegram-bot
sentence-transformers  # опционально, для реранкера
```

### 5. Загрузка данных

#### Вариант А: один файл
Поместите файл с данными в корень (например, `data.txt`) и выполните:

```bash
python load_data.py
```

Для принудительной перезагрузки (если данные уже есть):

```bash
python load_data.py --force
```

#### Вариант Б: папка с множеством файлов
Создайте папку `angels_txt` и положите туда все `.txt` файлы. Затем запустите:

```bash
python load_all_angels.py
```

Этот скрипт объединит все файлы с маркерами источника и загрузит в базы.

### 6. Запросы к системе

#### Через командную строку (скрипт `ask.py`)

```bash
python ask.py "Какой ангел отвечает за 14 октября?"
```

Или запустите без аргументов для интерактивного ввода:

```bash
python ask.py
```

#### Через Telegram-бота

Запустите бота:

```bash
python bot.py
```

После этого отправьте боту команду `/start` и задавайте вопросы.

## Структура проекта

```
.
├── config.py               # конфигурация, переменные окружения
├── clients.py              # инициализация клиентов Neo4j и Qdrant
├── graph_extraction.py     # извлечение графа из текста (с чанкингом)
├── ingestion.py            # загрузка в Neo4j и Qdrant
├── retrieval.py            # поиск, получение подграфа, форматирование
├── graphrag.py             # генерация ответа с графовым контекстом
├── ollama_processor.py     # реализация для Ollama (эмбеддинги, LLM)
├── utils.py                # вспомогательные функции (очистка, проверка данных)
├── reranker.py             # опционально: реранкинг результатов
├── load_data.py            # скрипт для загрузки одного файла
├── load_all_angels.py      # скрипт для загрузки папки с файлами
├── ask.py                  # скрипт для вопросов в командной строке
├── bot.py                  # Telegram-бот
├── .env.example            # пример файла переменных окружения
├── requirements.txt        # зависимости
└── README.md               # этот файл
```

## Настройка промптов

- **Для извлечения графа** (`graph_extraction.py`) используется системный промпт, задающий роль «извлекателя отношений» и требующий JSON.
- **Для ответов на вопросы** (`ollama_processor.py` в функции `graphrag_query`) используется системный промпт, который можно настроить под свои правила (поиск по датам, исключение источников и т.д.). По умолчанию там стоит краткая инструкция; вы можете заменить её на свой расширенный вариант (см. пример в коде).

## Производительность

- При большом количестве чанков (например, > 500) рекомендуется увеличить параллелизм в `graph_extraction.py` (параметр `max_workers` в `ThreadPoolExecutor`).
- Для ускорения эмбеддингов можно настроить размер батча в `ollama_embeddings_batch` (переменная `batch_size`).
- Если используется сервер с GPU, Ollama автоматически задействует его для инференса.

## Известные ограничения

- Модель может не всегда строго следовать формату JSON при извлечении графа – требуется тщательная настройка промпта и, возможно, увеличение `max_tokens`.
- При очень больших объёмах текста может потребоваться оптимизация чанкинга (уменьшение размера чанка).
- Telegram-бот не поддерживает потоковые ответы (отправляет сразу весь текст).

## Лицензия

MIT

## Контакты

По вопросам и предложениям обращайтесь к автору проекта.