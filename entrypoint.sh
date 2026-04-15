#!/bin/bash
set -e

# Проверяем, есть ли индекс. Если нет - строим
if [ ! -f "vector_index/faiss.index" ]; then
    echo "Индекс не найден. Выполняем подготовку данных и индексацию..."
    python prepare_data.py
    python build_index.py
fi

# Запускаем переданную команду (например, python main.py --query "...")
exec "$@"