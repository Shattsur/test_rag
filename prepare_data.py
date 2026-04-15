import os
import json
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

# Источник: адаптированные фрагменты из Wikipedia
DOCUMENTS = {
    "rag.txt": """Retrieval-Augmented Generation (RAG) — это метод, который объединяет поиск информации и генерацию текста. RAG позволяет LLM обращаться к внешней базе знаний, что снижает галлюцинации и улучшает фактологическую точность. RAG состоит из двух основных компонентов: ретривер (поисковик) и генератор (LLM). Ретривер находит релевантные документы, а генератор создаёт ответ на основе найденного контекста.""",
    
    "faiss.txt": """FAISS (Facebook AI Similarity Search) — библиотека для эффективного поиска схожих векторов. Она разработана в Facebook Research. FAISS поддерживает различные типы индексов: Flat (точный поиск), IVF (инвертированный файл), HNSW (графовый). FAISS широко используется в RAG-системах для хранения эмбеддингов документов и быстрого поиска релевантных фрагментов.""",
    
    "transformer.txt": """Трансформер — архитектура нейронных сетей, представленная в статье "Attention is All You Need" (2017). Трансформеры используют механизм самовнимания (self-attention) для обработки последовательностей. Они стали основой для всех современных LLM: BERT, GPT, T5, Llama, Gemma. Трансформеры позволяют моделям учитывать контекст каждого слова относительно всех остальных слов в предложении.""",
    
    "bert.txt": """BERT (Bidirectional Encoder Representations from Transformers) — модель от Google, обученная на задачах маскирования слов и предсказания следующего предложения. BERT является энкодерным трансформером, который хорошо подходит для задач понимания текста. Sentence-BERT на основе BERT позволяет получать эмбеддинги предложений для задач семантического поиска.""",
    
    "sentence_transformers.txt": """Sentence-Transformers — библиотека на PyTorch, которая предоставляет предобученные модели для создания эмбеддингов предложений. Модель all-MiniLM-L6-v2 является одной из самых популярных: она компактная, быстрая и даёт качественные векторы размерности 384. Sentence-Transformers используются для кластеризации, семантического поиска и в RAG-системах.""",
    
    "llm.txt": """Большие языковые модели (LLM) — это нейронные сети с миллиардами параметров, обученные на огромных корпусах текста. Они способны генерировать связный текст, отвечать на вопросы, переводить, писать код. Примеры: GPT-4, Llama 3, Gemma, Mistral. LLM могут галлюцинировать (выдумывать факты), поэтому их часто комбинируют с RAG для доступа к актуальной информации.""",
    
    "evaluation.txt": """Оценка качества RAG-систем включает метрики для ретривера (recall@k, MRR) и для генератора (ROUGE, BLEU, BERTScore). ROUGE измеряет совпадение n-грамм между сгенерированным ответом и эталоном. BLEU ориентирован на точность совпадения фраз. BERTScore использует эмбеддинги BERT для оценки семантической близости.""",
    
    "docker.txt": """Docker — платформа для контейнеризации приложений. Docker позволяет упаковать RAG-систему со всеми зависимостями в образ, который можно запустить на любой системе. Для RAG-систем в Docker обычно копируют код, устанавливают зависимости, загружают модели (или кешируют их через тома). Важно использовать entrypoint для автоматической сборки индекса при первом запуске.""",
    
    "python.txt": """Python — основной язык для RAG-систем благодаря экосистеме библиотек: transformers, sentence-transformers, faiss, langchain, chromadb, torch. Python обеспечивает быструю разработку и интеграцию с ML-фреймворками.""",
    
    "ml.txt": """Машинное обучение (ML) — область ИИ, где модели обучаются на данных. В контексте RAG используются методы обучения без учителя (для эмбеддингов) и обучение с подкреплением (для улучшения генерации). Современные RAG-системы могут дообучать ретривер и генератор на специфических доменах."""
}

def create_raw_data():
    os.makedirs("data/raw", exist_ok=True)
    for filename, content in DOCUMENTS.items():
        with open(f"data/raw/{filename}", "w", encoding="utf-8") as f:
            f.write(content)
    print(f"✅ Создано {len(DOCUMENTS)} документов в data/raw/")

def chunk_documents():
    os.makedirs("chunks", exist_ok=True)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    all_chunks = []
    chunk_id = 0
    for filename in os.listdir("data/raw"):
        if not filename.endswith(".txt"):
            continue
        with open(f"data/raw/{filename}", "r", encoding="utf-8") as f:
            text = f.read()
        chunks = text_splitter.split_text(text)
        for chunk_text in chunks:
            if len(chunk_text.strip()) > 50:  # отсекаем слишком короткие
                all_chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "source": filename
                })
                chunk_id += 1
    with open("chunks/chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"✅ Создано {len(all_chunks)} чанков. Сохранено в chunks/chunks.json")

if __name__ == "__main__":
    create_raw_data()
    chunk_documents()