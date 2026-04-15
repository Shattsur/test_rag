import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from torchgen import context
from torchgen import context
from transformers import pipeline, AutoTokenizer
import torch
import config

class RAGPipeline:
    def __init__(self):
        # Загрузка эмбеддера
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        
        # Загрузка FAISS индекса и метаданных
        self.index = faiss.read_index("vector_index/faiss.index")
        with open("vector_index/chunks_metadata.json", "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        
        # Загрузка LLM напрямую через pipeline 
        print(f"Загрузка LLM: {config.LLM_MODEL}")
        
        # Определяем torch_dtype
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        self.generator = pipeline(
            "text-generation",
            model=config.LLM_MODEL,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            token=None  # будет None для phi-2
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL, token=None, trust_remote_code=True)
        
        # Устанавливаем pad_token, если его нет
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def retrieve(self, query: str, k: int = None):
        if k is None:
            k = config.RETRIEVAL_K
        query_emb = self.embedder.encode([query])[0].astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_emb, k)
        retrieved_chunks = [self.chunks[i] for i in indices[0] if i != -1]
        return retrieved_chunks
    
    def generate(self, query: str, return_sources=False):
        # 1. Retrieve
        retrieved = self.retrieve(query)
        context = "\n\n".join([f"[{c['source']}]: {c['text']}" for c in retrieved])
        
        # 2. Форматирование сообщений для чат-шаблона
        messages = [
            {"role": "system", "content": "Ты — полезный ассистент. Отвечай на вопросы строго на основе контекста, коротко и по существу, без повторения вопроса."},
            {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {query}\n\nОтвет:"}
        ]
        
        # Применяем шаблон чата (если модель его поддерживает)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 3. Генерация
        outputs = self.generator(
            prompt,
            max_new_tokens=300,
            temperature=0.3,
            do_sample=True,
            top_p=0.95,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        answer = outputs[0]['generated_text'][len(prompt):].strip()
        
        if return_sources:
            return answer, retrieved
        return answer

if __name__ == "__main__":
    rag = RAGPipeline()
    query = "Что такое RAG?"
    answer, sources = rag.generate(query, return_sources=True)
    print("=== Вопрос ===")
    print(query)
    print("=== Ответ ===")
    print(answer)
    print("=== Источники ===")
    for s in sources:
        print(f"- {s['source']}: {s['text'][:100]}...")