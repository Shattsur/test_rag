import json
import numpy as np
from rouge_score import rouge_scorer
from bert_score import BERTScorer
import evaluate as hf_evaluate
from rag_pipeline import RAGPipeline

# Загружаем BLEU
bleu = hf_evaluate.load("bleu")

# Тестовый набор (те же, но с проверкой на наличие релевантных источников)
TEST_SET = [
    {
        "query": "Что такое RAG?",
        "reference": "RAG — это метод, объединяющий поиск информации и генерацию текста для снижения галлюцинаций LLM.",
        "relevant_sources": ["rag.txt"]
    },
    {
        "query": "Для чего используется FAISS?",
        "reference": "FAISS используется для быстрого поиска схожих векторов и хранения эмбеддингов в RAG-системах.",
        "relevant_sources": ["faiss.txt"]
    },
    {
        "query": "Какая модель эмбеддингов популярна для RAG?",
        "reference": "all-MiniLM-L6-v2 от Sentence-Transformers популярна из-за компактности и скорости.",
        "relevant_sources": ["sentence_transformers.txt"]
    },
    {
        "query": "Что такое трансформер?",
        "reference": "Трансформер — архитектура нейронных сетей на основе механизма самовнимания.",
        "relevant_sources": ["transformer.txt"]
    },
    {
        "query": "Назовите один из недостатков LLM.",
        "reference": "LLM могут галлюцинировать, то есть выдумывать факты.",
        "relevant_sources": ["llm.txt"]
    }
]

def evaluate_retriever(rag, test_set):
    from config import RETRIEVAL_K
    recalls = []
    for test in test_set:
        # Пропускаем, если нет релевантных источников (защита от пустого списка)
        if not test.get("relevant_sources"):
            continue
        retrieved = rag.retrieve(test["query"], k=RETRIEVAL_K)
        retrieved_sources = set([c["source"] for c in retrieved])
        relevant = set(test["relevant_sources"])
        recall = len(retrieved_sources & relevant) / len(relevant)
        recalls.append(recall)
    avg_recall = np.mean(recalls) if recalls else 0.0
    print(f"📊 Retriever recall@{RETRIEVAL_K}: {avg_recall:.3f}")
    return avg_recall

def evaluate_generator(rag, test_set):
    scorer_rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    bert_scorer = BERTScorer(lang="ru", rescale_with_baseline=False)
    
    rouge_scores = []
    bleu_scores = []
    bert_scores = []
    
    for test in test_set:
        answer = rag.generate(test["query"])
        # ROUGE
        rouge = scorer_rouge.score(test["reference"], answer)
        rouge_scores.append({
            "rouge1": rouge["rouge1"].fmeasure,
            "rouge2": rouge["rouge2"].fmeasure,
            "rougeL": rouge["rougeL"].fmeasure
        })
        # BLEU
        bleu_result = bleu.compute(predictions=[answer], references=[[test["reference"]]])
        bleu_scores.append(bleu_result["bleu"])
        # BERTScore
        P, R, F1 = bert_scorer.score([answer], [test["reference"]])
        bert_scores.append(F1.item())
        
        print(f"\n--- {test['query']} ---")
        print(f"Ответ: {answer}")
        print(f"ROUGE-1: {rouge['rouge1'].fmeasure:.3f}, BLEU: {bleu_result['bleu']:.3f}, BERTScore: {F1.item():.3f}")
    
    avg_rouge1 = np.mean([s["rouge1"] for s in rouge_scores])
    avg_bleu = np.mean(bleu_scores)
    avg_bert = np.mean(bert_scores)
    
    print("\n" + "="*50)
    print(f"📈 Средняя метрика генерации:")
    print(f"   ROUGE-1: {avg_rouge1:.3f}")
    print(f"   BLEU:    {avg_bleu:.3f}")
    print(f"   BERTScore: {avg_bert:.3f}")
    return avg_rouge1, avg_bleu, avg_bert

if __name__ == "__main__":
    print("Загрузка RAG-пайплайна...")
    rag = RAGPipeline()
    print("\n=== Оценка ретривера ===")
    evaluate_retriever(rag, TEST_SET)
    print("\n=== Оценка генератора ===")
    evaluate_generator(rag, TEST_SET)