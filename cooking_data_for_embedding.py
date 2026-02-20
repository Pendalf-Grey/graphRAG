# Чтение из raw_data.txt, приведение к нижнему регистру, запись в output1.txt
with open('data.txt', 'r', encoding='utf-8') as infile, \
     open('output1.txt', 'w', encoding='utf-8') as outfile:
    for line in infile:
        outfile.write(line.lower())

# with open('raw_data.txt', 'r', encoding='utf-8') as infile, \
#      open('output2.txt', 'w', encoding='utf-8') as outfile:
#
#     lines = []
#     for line in infile:
#         # Удаляем только символ новой строки в конце (оставляем все остальные пробелы)
#         line_without_newline = line.rstrip('\n')
#         # Приводим к нижнему регистру (как в output1)
#         lower_line = line_without_newline.lower()
#         # Пропускаем строки, не содержащие печатных символов (включая строки из пробелов)
#         if lower_line.strip():
#             lines.append(lower_line)
#
#     # Объединяем содержательные строки через запятую
#     result = ', '.join(lines)
#     outfile.write(result)