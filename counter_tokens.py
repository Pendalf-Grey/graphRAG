from transformers import AutoTokenizer

# Загружаем токенизатор для модели Qwen2.5 (подставьте нужную версию, например "Qwen/Qwen2.5-14B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")

# Ваш текст (можно прочитать из файла)
text = open("data.txt", "r", encoding="utf-8").read()

# Токенизация
tokens = tokenizer.encode(text)
print(f"Количество токенов: {len(tokens)}")