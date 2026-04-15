import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import config

def build_index():
    # Загрузка чанков
    with open("chunks/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    if not chunks:
        raise RuntimeError("Нет чанков. Сначала запустите python prepare_data.py")
    
    # Модель эмбеддингов
    print(f"Загрузка модели эмбеддингов: {config.EMBEDDING_MODEL}")
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    
    # Вычисление эмбеддингов
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    
    # Создание FAISS индекса
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 расстояние
    index.add(embeddings)
    
    # Сохранение индекса
    os.makedirs("vector_index", exist_ok=True)
    faiss.write_index(index, "vector_index/faiss.index")
    
    # Сохраняем метаданные (порядок чанков соответствует индексу)
    with open("vector_index/chunks_metadata.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Индекс сохранён: {index.ntotal} векторов размерности {dimension}")

if __name__ == "__main__":
    build_index()