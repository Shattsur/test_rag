# -*- coding: utf-8 -*-
"""
fit_params_optuna_nsgaii_v3.py — Генетическая оптимизация RAG с SOTA функциями (NSGA-II)
Совместим с rag_gen.py (Gen Version)

"""
import csv
import json
import logging
import threading
import time
import gc
import contextlib
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import optuna
from optuna.samplers import NSGAIISampler
from concurrent.futures import ThreadPoolExecutor, as_completed

from FOR_INFO_rag_gen import Settings, SmetaRAGApp, cosine_similarity, get_cached_reranker, collect_sources, docs_to_context

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    # будет использован fallback без графиков
    logger = logging.getLogger("ga_optimizer_sota")
    logger.warning("⚠ Matplotlib/seaborn не установлены — расширенные визуализации отключены")

# ==================== ЛОГИРОВАНИЕ ====================
for logger_name in [
    "httpx", "httpcore", "urllib3", "langchain", "langchain_openai",
    "transformers", "sentence_transformers", "chromadb", "optuna"
]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("ga_optimizer_sota")

# ==================== КОНФИГУРАЦИЯ ====================
@dataclass
class GAConfig:
    """
    Конфигурация генетического алгоритма с поддержкой SOTA параметров.
    """
    # --- Диапазоны старых параметров ---
    bm25_k_range: Tuple[int, int] = (10, 500)
    vec_k_range: Tuple[int, int] = (10, 300)
    rrf_k_range: Tuple[int, int] = (10, 300)
    rerank_k_range: Tuple[int, int] = (10, 50)
    rerank_threshold_range: Tuple[float, float] = (0.005, 0.25)
    final_k_range: Tuple[int, int] = (2, 10)
    
    # --- Новые параметры: Реранкеры ---
    reranker_models: List[str] = field(default_factory=lambda: [
        # "nvidia/llama-nemotron-rerank-1b-v2",  
        "Shattsur/nemotron-smeta-4bit-adapter", # "mixedbread-ai/mxbai-rerank-xsmall-v1"
        "DiTy/cross-encoder-russian-msmarco",
        "Alibaba-NLP/gte-reranker-modernbert-base"  
        ])
    
    # Типы реранкеров для UnifiedReranker
    reranker_types: List[str] = field(default_factory=lambda: ["auto"])  # ← только auto!

    
    # --- Новые параметры: Fusion ---
    fusion_strategies: List[str] = field(default_factory=lambda: ["rrf", "weighted"])
    hybrid_alpha_range: Tuple[float, float] = (0.1, 0.9)
    
    # --- Новые параметры: Retrieval Mode ---
    retrieval_modes: List[str] = field(default_factory=lambda: ["fixed", "adaptive"])
    adaptive_threshold_range: Tuple[float, float] = (0.05, 0.25)
    
    # --- Batch Inference  ---
    batch_size: int = 3  # Кол-во вопросов, обрабатываемых за один прогон (для оптимизации времени) 
    
    # --- Параметры NSGA-II ---
    population_size: int = 28
    n_generations: int = 60
    mutation_prob: float = 0.35
    crossover_prob: float = 0.9
    
    # --- Шумоподавление ---
    num_eval_runs: int = 1 # при температуре больше 0.0 рекомендуется увеличить для усреднения метрик
    
    # --- Веса скаляризации ---
    w_context: float = 1.0
    w_faithful: float = 1.2
    w_answer: float = 1.1
    
    # --- Бейзлайн (Seed) ---
    baseline_params: Dict = field(default_factory=lambda: {
        'bm25_k': 50, 'vec_k': 50, 'rrf_k': 50,
        'rerank_k': 25, 'rerank_threshold': 0.05, 'final_k': 10,
        # Новые параметры со значениями по умолчанию (старое поведение)
        'reranker_model': "Alibaba-NLP/gte-reranker-modernbert-base",
        'reranker_type': "auto",  # ✅ НОВОЕ
        'fusion_strategy': "rrf",
        'hybrid_alpha': 0.5,  # Всегда есть значение
        'retrieval_mode': "fixed",
        'adaptive_threshold': 0.05,  # Всегда есть значение
    })
    n_baseline_seeds: int = 1
    output_dir: str = "./output/ga_optimization"
    random_seed: int = 42

      
    @property
    def n_trials(self) -> int:
        """✅ Возвращает общее количество испытаний"""
        return self.population_size * self.n_generations
    
    def validate(self):
        """✅ Валидация конфигурации"""
        assert self.n_generations >= 3, f"Минимум 3 поколения, задано {self.n_generations}"
        assert self.population_size >= 10, f"Популяция < 10, задано {self.population_size}"
        assert self.n_baseline_seeds < self.population_size, f"Seeds >= population"
        logger.info(f"✓ Конфигурация валидна: {self.population_size}×{self.n_generations} = {self.n_trials} trials")

@dataclass
class Individual:
    # Старые параметры
    bm25_k: int
    vec_k: int
    rrf_k: int
    rerank_k: int
    rerank_threshold: float
    final_k: int
    # Новые параметры
    reranker_model: str
    reranker_type: str  # ✅ НОВОЕ
    fusion_strategy: str
    hybrid_alpha: float
    retrieval_mode: str
    adaptive_threshold: float
    
    fitness: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {**asdict(self), **self.metrics}

# ==================== УТИЛИТЫ ====================
def _cache_key(params: Dict) -> Tuple:
    """Уникальный ключ кэша с учетом всех новых параметров."""
    return (
        params['bm25_k'], params['vec_k'], params['rrf_k'],
        params['rerank_k'], round(params['rerank_threshold'], 4),
        params['final_k'],
        params['reranker_model'],
        params.get('reranker_type', 'auto'),  # ✅ НОВОЕ
        params['fusion_strategy'],
        round(params['hybrid_alpha'], 2),
        params['retrieval_mode'],
        round(params['adaptive_threshold'], 3)
    )

def _scalarized_score(metrics: Dict[str, float], config: GAConfig) -> float:
    return (
        config.w_context * metrics.get('context_relevance', 0) +
        config.w_faithful * metrics.get('faithfulness', 0) +
        config.w_answer * metrics.get('answer_relevance', 0)
    )

def _generate_baseline_variations(baseline: Dict, n: int, config: GAConfig, rng: np.random.RandomState) -> List[Dict]:
    variations = [baseline.copy()]
    for _ in range(n - 1):
        v = {}
        # Вариации только для числовых параметров, категориальные оставляем как бейзлайн
        for key, val in baseline.items():
            if key in ['reranker_model', 'reranker_type', 'fusion_strategy', 'retrieval_mode']:
                v[key] = val
            elif key == 'rerank_threshold' or key == 'adaptive_threshold' or key == 'hybrid_alpha':
                # Плавающие параметры
                if key == 'rerank_threshold': lo, hi = config.rerank_threshold_range
                elif key == 'adaptive_threshold': lo, hi = config.adaptive_threshold_range
                else: lo, hi = config.hybrid_alpha_range
                delta = val * rng.uniform(-0.2, 0.2)
                v[key] = round(np.clip(val + delta, lo, hi), 4)
            else:
                # Целочисленные параметры
                ranges_map = {
                    'bm25_k': config.bm25_k_range, 'vec_k': config.vec_k_range,
                    'rrf_k': config.rrf_k_range, 'rerank_k': config.rerank_k_range,
                    'final_k': config.final_k_range,
                }
                if key in ranges_map:
                    lo, hi = ranges_map[key]
                    delta = int(val * rng.uniform(-0.2, 0.2))
                    v[key] = int(np.clip(val + delta, lo, hi))
                else:
                    v[key] = val
        variations.append(v)
    return variations

@contextlib.contextmanager
def temporary_retriever_params(retriever, new_settings: Settings, new_bm25_k: int,
                               original_settings: Settings, original_bm25_k: int):
    """ФИКС: теперь реально переключаем реранкер через кэш (Qwen, Jina, BGE будут работать!)"""
    original_reranker = retriever.reranker
    # Берём нужный реранкер из глобального кэша (грузится только один раз!)
    new_reranker = get_cached_reranker(
        new_settings.rerank_model,
        device=original_reranker.device,          # берём device из уже загруженного
        max_length=new_settings.rerank_max_length,
        reranker_type=new_settings.reranker_type
    )
    try:
        retriever.s = new_settings
        retriever.bm25.k = new_bm25_k
        retriever.reranker = new_reranker         # ← главное: теперь модель меняется!
        yield
    finally:
        retriever.s = original_settings
        retriever.bm25.k = original_bm25_k
        retriever.reranker = original_reranker

        torch.cuda.empty_cache()
        gc.collect()

def create_settings_from_individual(base: Settings, ind: Individual) -> Settings:
    """
    Создаёт Settings из Individual.
    ✅ pool_k вычисляется динамически через bm25_k + vec_k
    ✅ reranker_type передаётся явно
    """
    return Settings(
        # Поиск
        bm25_k=ind.bm25_k,
        vec_k=ind.vec_k,
        rrf_k=ind.rrf_k,
        pool_k=ind.bm25_k + ind.vec_k,  # ✅ Динамический расчет
        # Реранк
        rerank_k=ind.rerank_k,
        rerank_threshold=ind.rerank_threshold,
        final_k=ind.final_k,
        rerank_model=ind.reranker_model,
        reranker_type=ind.reranker_type,  # ✅ НОВОЕ
        # SOTA стратегии
        fusion_strategy=ind.fusion_strategy,
        hybrid_alpha=ind.hybrid_alpha,
        retrieval_mode=ind.retrieval_mode,
        adaptive_threshold=ind.adaptive_threshold,
        # Остальное (без изменений)
        chroma_dir=base.chroma_dir,
        collection=base.collection,
        chunks_jsonl=base.chunks_jsonl,
        embed_model=base.embed_model,
        embed_max_length=base.embed_max_length,
        embed_batch_size=base.embed_batch_size,
        truncate_dim=base.truncate_dim,
        rerank_max_length=base.rerank_max_length,
        rerank_batch_size=base.rerank_batch_size,
        llm_model=base.llm_model,
        llm_base_url=base.llm_base_url,
        llm_api_key=base.llm_api_key,
        max_context_chars=base.max_context_chars,
        per_doc_char_cap=base.per_doc_char_cap,
        max_tokens=base.max_tokens,
        temperature=base.temperature,
        hhem_model_name=base.hhem_model_name,
        hhem_max_length=base.hhem_max_length,
    )

# ==================== ОСНОВНОЙ ОПТИМИЗАТОР ====================
class GAOptimizer:
    def __init__(self, config: GAConfig, test_questions: List[Dict], app: SmetaRAGApp):
        config.validate()
        self.config = config
        self.test_questions = test_questions
        self.app = app
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.eval_cache: Dict[Tuple, Dict[str, float]] = {}
        self.generation_log: List[Dict] = []
        
        # === ПРОФИЛИРОВАНИЕ ===
        self.profile_lock = threading.Lock()
        self.profile_times = {
            'ask_total': [],           # app.ask (retrieval + rerank + LLM)
            'retrieval': [],          # Время, выделенное на retrieval (из meta)
            'llm': [],                # Время, выделенное на LLM (аппроксимация)
            'cr_embed': [],            # Context Relevance
            'faith': [],               # Faithfulness (HHEM или fallback)
            'ar_embed': [],            # Answer Relevance
            'total_single': []         # всё _evaluate_single
        }
        self._precompute_chunk_embeddings()
        self._precompute_embeddings()

        self.llm_cache = {}          # кэш ответов LLM
        self.llm_cache_hits = 0
        self.llm_cache_misses = 0
        
    
    def _precompute_embeddings(self):
        """Предвычисляет эмбеддинги вопросов и референсов"""
        self.question_embs = []
        self.reference_embs = []
        for q in self.test_questions:
            self.question_embs.append(np.array(self.app.embedding_fn.embed_query(q["question"])))
            ref = q.get("reference", "")
            self.reference_embs.append(
                np.array(self.app.embedding_fn.embed_query(ref)) if ref.strip() else None
            )
        logger.info(f"✅ Предвычислено {len(self.question_embs)} эмбеддингов вопросов и референсов")
    
    def _precompute_chunk_embeddings(self):
        """Предвычисление эмбеддингов всех уникальных чанков из Chroma один раз при старте."""
        logger.info("🔄 Предвычисление эмбеддингов всех чанков...")
        t0 = time.perf_counter()
        
        self.chunk_embeddings = {}
        
        try:
            # Достаём все документы + эмбеддинги
            collection_data = self.app.retriever.vs._collection.get(
                include=["documents", "embeddings"]
            )
            
            documents = collection_data.get("documents", [])
            stored_embeddings = collection_data.get("embeddings", [])
            
            if not documents:
                logger.warning("⚠️ В Chroma коллекции нет документов")
                return
            
            logger.info(f"Получено {len(documents)} чанков из Chroma")
            
            # Если embeddings уже есть в коллекции — используем их
            if len(stored_embeddings) == len(documents):
                logger.info(f"✅ Chroma уже содержит {len(stored_embeddings)} эмбеддингов — используем их")
                for doc_text, emb_list in zip(documents, stored_embeddings):
                    # emb_list — это list[float], преобразуем в np.array
                    self.chunk_embeddings[doc_text] = np.asarray(emb_list, dtype=np.float32)
            
            else:
                # Вычисляем сами
                logger.info(f"⚠️ Эмбеддинги отсутствуют/неполные — вычисляем заново")
                
                # Убираем дубликаты безопасно
                unique_texts = []
                seen = set()
                for text in documents:
                    if text not in seen:
                        unique_texts.append(text)
                        seen.add(text)
                
                batch_size = 128
                computed_emb = []
                for i in range(0, len(unique_texts), batch_size):
                    batch = unique_texts[i:i + batch_size]
                    batch_emb = self.app.embedding_fn.embed_documents(batch)
                    # Преобразуем каждый в np.array
                    computed_emb.extend([np.asarray(e, dtype=np.float32) for e in batch_emb])
                
                self.chunk_embeddings = dict(zip(unique_texts, computed_emb))
            
            t1 = time.perf_counter()
            logger.info(f"✅ Предвычислено/загружено {len(self.chunk_embeddings)} уникальных чанков за {t1-t0:.1f} сек")
        
        except Exception as e:
            logger.error(f"Ошибка предвычисления чанков: {e}")
            self.chunk_embeddings = {}
            
    def _preload_rerankers(self):
        """
        ✅ НОВОЕ: Предзагружает все реранкеры в кэш до начала оптимизации
        Экономит 20-40 секунд на каждый trial с новой моделью
        """
        logger.info("\n🔄 Предзагрузка реранкеров...")
        
        # Берём device из уже инициализированного реранкера
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"   🚀 Используется GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.warning("   ⚠️ CUDA не найдена, реранкеры будут работать на CPU (МЕДЛЕННО)!")
        
        loaded_count = 0
        failed_count = 0
        total_start = time.time()
        
        for model in self.config.reranker_models:
            for rtype in self.config.reranker_types:
                try:
                    model_short = model.split('/')[-1]
                    logger.info(f"   ⏳ Загрузка: {model_short} ({rtype})...")
                    t0 = time.time()
                    
                    # ✅ Используем ту же функцию, что и в temporary_retriever_params
                    get_cached_reranker(
                        model_name=model,
                        device=device,
                        max_length=self.app.s.rerank_max_length,
                        reranker_type=rtype
                    )
                    
                    elapsed = time.time() - t0
                    logger.info(f"   ✅ Загружено за {elapsed:.1f}с")
                    loaded_count += 1
                    
                except Exception as e:
                    logger.warning(f"   ⚠ Не удалось загрузить {model} ({rtype}): {e}")
                    failed_count += 1
        
        total_elapsed = time.time() - total_start
        logger.info(f"✅ Предзагружено {loaded_count} реранкеров за {total_elapsed:.1f}с (не удалось: {failed_count})")
    
    def _evaluate_single(self, question: str, q_emb: np.ndarray, ref_emb: Optional[np.ndarray]) -> Dict[str, float]:
        t_total = time.perf_counter()
        
        # 1. Retrieval
        t0 = time.perf_counter()
        docs, meta = self.app.retriever.retrieve(question)
        contexts = [doc.page_content for doc in docs]
        sources = collect_sources(docs)
        context = docs_to_context(docs, self.app.s.max_context_chars, self.app.s.per_doc_char_cap)
        retrieval_time = meta.get("retrieval_time", 0.0)
        t_retrieval = time.perf_counter() - t0   # время на retrieval (можно также взять из meta)
        
        # 2. LLM generation with cache
        cache_key = (question, tuple(contexts))
        if cache_key in self.llm_cache:
            answer = self.llm_cache[cache_key]
            self.llm_cache_hits += 1
            t_llm = 0.0
        else:
            t0 = time.perf_counter()
            answer = self.app.answerer.synthesize(question, context, sources)
            t_llm = time.perf_counter() - t0
            self.llm_cache[cache_key] = answer
            self.llm_cache_misses += 1
        
        # 3. Context Relevance (используем предвычисленные эмбеддинги)
        t0 = time.perf_counter()
        if contexts:
            try:
                ctx_embs_list = []
                for ctx in contexts:
                    emb = self.chunk_embeddings.get(ctx)
                    if emb is None:
                        emb = self.app.embedding_fn.embed_documents([ctx])[0]
                        self.chunk_embeddings[ctx] = emb
                    ctx_embs_list.append(emb)
                ctx_embs = np.array(ctx_embs_list)
                sims = cosine_similarity([q_emb], ctx_embs)[0]
                cr = float(np.mean(sims))
            except Exception as e:
                logger.warning(f"Ошибка предвычисленных эмбеддингов: {e}. Фоллбэк на embed_documents")
                ctx_embs = np.array(self.app.embedding_fn.embed_documents(contexts))
                sims = cosine_similarity([q_emb], ctx_embs)[0]
                cr = float(np.mean(sims))
        else:
            cr = 0.0
        t_cr = time.perf_counter() - t0
        
        # 4. Faithfulness
        t0 = time.perf_counter()
        if self.app.hhem_evaluator.is_available and contexts:
            faith = self.app.hhem_evaluator.evaluate(answer, contexts)
        elif contexts and answer.strip():
            ctx_embs = np.array(self.app.embedding_fn.embed_documents(contexts[:3]))
            ans_emb = np.array(self.app.embedding_fn.embed_query(answer))
            sims = cosine_similarity([ans_emb], ctx_embs)[0]
            faith = float(np.mean(sims))
        else:
            faith = 0.5
        t_faith = time.perf_counter() - t0
        
        # 5. Answer Relevance
        t0 = time.perf_counter()
        if ref_emb is not None and answer.strip():
            ans_emb = np.array(self.app.embedding_fn.embed_query(answer))
            ar = float(cosine_similarity([ans_emb], [ref_emb])[0][0])
        else:
            ar = 0.5
        t_ar = time.perf_counter() - t0
        
        t_single = time.perf_counter() - t_total
        
        # Сохраняем статистику
        with self.profile_lock:
            self.profile_times['retrieval'].append(t_retrieval)
            self.profile_times['llm'].append(t_llm)
            self.profile_times['ask_total'].append(t_retrieval + t_llm)
            self.profile_times['cr_embed'].append(t_cr)
            self.profile_times['faith'].append(t_faith)
            self.profile_times['ar_embed'].append(t_ar)
            self.profile_times['total_single'].append(t_single)
        
        return {'context_relevance': cr, 'faithfulness': faith, 'answer_relevance': ar}
    
    def _evaluate_params(self, params: Dict) -> Dict[str, float]:
        key = _cache_key(params)
        if key in self.eval_cache:
            return self.eval_cache[key]
        
        # Создаем Individual
        ind = Individual(
            bm25_k=params['bm25_k'],
            vec_k=params['vec_k'],
            rrf_k=params['rrf_k'],
            rerank_k=params['rerank_k'],
            rerank_threshold=params['rerank_threshold'],
            final_k=params['final_k'],
            reranker_model=params['reranker_model'],
            reranker_type=params.get('reranker_type', 'auto'),
            fusion_strategy=params['fusion_strategy'],
            hybrid_alpha=params.get('hybrid_alpha', 0.5),
            retrieval_mode=params['retrieval_mode'],
            adaptive_threshold=params.get('adaptive_threshold', 0.05),
        )
        
        original_settings = self.app.s
        new_settings = create_settings_from_individual(original_settings, ind)
        
        all_cr, all_fa, all_ar = [], [], []
        
        for run_idx in range(self.config.num_eval_runs):
            with temporary_retriever_params(
                self.app.retriever, new_settings, ind.bm25_k,
                original_settings, original_settings.bm25_k
            ):
                # === BATCH INFERENCE ===
                def evaluate_one(i: int):
                    try:
                        tc = self.test_questions[i]
                        return self._evaluate_single(
                            tc["question"], self.question_embs[i], self.reference_embs[i]
                        )
                    except Exception as e:
                        logger.warning(f"⚠️ Error in thread Q#{i} (trial {key}): {e}")
                        return {'context_relevance': 0.0, 'faithfulness': 0.0, 'answer_relevance': 0.0}
                
                with ThreadPoolExecutor(max_workers=self.config.batch_size) as executor:
                    futures = [executor.submit(evaluate_one, i) 
                              for i in range(len(self.test_questions))]
                    
                    for future in as_completed(futures):
                        m = future.result()
                        all_cr.append(m['context_relevance'])
                        all_fa.append(m['faithfulness'])
                        all_ar.append(m['answer_relevance'])
        
        metrics = {
            'context_relevance': float(np.mean(all_cr)),
            'faithfulness': float(np.mean(all_fa)),
            'answer_relevance': float(np.mean(all_ar)),
        }
        self.eval_cache[key] = metrics
        return metrics
    
    def _objective(self, trial: optuna.Trial) -> Tuple[float, float, float]:
        # 1. Старые параметры
        params = {
            'bm25_k': trial.suggest_int("bm25_k", *self.config.bm25_k_range),
            'vec_k': trial.suggest_int("vec_k", *self.config.vec_k_range),
            'rrf_k': trial.suggest_int("rrf_k", *self.config.rrf_k_range),
            'rerank_k': trial.suggest_int("rerank_k", *self.config.rerank_k_range),
            'rerank_threshold': trial.suggest_float("rerank_threshold", *self.config.rerank_threshold_range),
            'final_k': trial.suggest_int("final_k", *self.config.final_k_range),
            # 2. Новые параметры SOTA
            'reranker_model': trial.suggest_categorical("reranker_model", self.config.reranker_models),
            'reranker_type': trial.suggest_categorical("reranker_type", self.config.reranker_types),  # ✅ НОВОЕ
            'fusion_strategy': trial.suggest_categorical("fusion_strategy", self.config.fusion_strategies),
            'retrieval_mode': trial.suggest_categorical("retrieval_mode", self.config.retrieval_modes),
        }
        
        # ✅ ИСПРАВЛЕНИЕ: Условные параметры ВСЕГДА создаются (не через if/else)
        # Это предотвращает KeyError при чтении из best_trials
        params['hybrid_alpha'] = trial.suggest_float("hybrid_alpha", *self.config.hybrid_alpha_range)
        params['adaptive_threshold'] = trial.suggest_float("adaptive_threshold", *self.config.adaptive_threshold_range)
        
        # Примечание: Значения будут игнорироваться, если стратегия не используется,
        # но параметр всегда существует в trial.params
        
        metrics = self._evaluate_params(params)
        
        # Логирование
        gen = trial.number // self.config.population_size
        idx_in_gen = trial.number % self.config.population_size
        score = _scalarized_score(metrics, self.config)
        
        if idx_in_gen == 0:
            logger.info(f"\n{'='*70}\n🧬 Поколение {gen + 1}/{self.config.n_generations}\n{'='*70}")
        
        logger.info(
            f"  [{trial.number+1:3d}] CR={metrics['context_relevance']:.3f} "
            f"F={metrics['faithfulness']:.3f} AR={metrics['answer_relevance']:.3f} |  "
            f"W={score:.3f} | Model={params['reranker_model'].split('/')[-1][:10]}... "
            f"Fusion={params['fusion_strategy']} Mode={params['retrieval_mode']}"
        )
        
        return (metrics['context_relevance'], metrics['faithfulness'], metrics['answer_relevance'])
    
    def optimize(self) -> Tuple[Individual, optuna.study.Study]:
        rng = np.random.RandomState(self.config.random_seed)
        sampler = NSGAIISampler(
            population_size=self.config.population_size,
            mutation_prob=self.config.mutation_prob,
            crossover_prob=self.config.crossover_prob,
            seed=self.config.random_seed,
        )
        study = optuna.create_study(
            directions=["maximize", "maximize", "maximize"],
            sampler=sampler,
            study_name="RAG_NSGA2_SOTA"
        )
        
        # Seed бейзлайна
        seeds = _generate_baseline_variations(
            self.config.baseline_params, self.config.n_baseline_seeds, self.config, rng
        )
        for seed_params in seeds:
            study.enqueue_trial(seed_params)
        
        logger.info(f"\n🚀 Запуск оптимизации SOTA RAG (NSGA-II)")
        logger.info(f"   Популяция: {self.config.population_size}, Поколений: {self.config.n_generations}")
        logger.info(f"   Реранкеры: {len(self.config.reranker_models)}, Стратегии: {self.config.fusion_strategies}")
        
        # ✅ НОВОЕ: Предзагрузка моделей ПЕРЕД оптимизацией
        logger.info("\n📦 Этап 1/2: Предзагрузка моделей...")
        self._preload_rerankers()
        
        # Прогрев embedding модели
        logger.info("📦 Этап 2/2: Прогрев embedding...")
        self.app.embedding_fn.embed_query("test")
        logger.info("✅ Все модели готовы к работе\n")
        
        t_start = time.time()
        study.optimize(self._objective, n_trials=self.config.n_trials, n_jobs=1, show_progress_bar=False)
        elapsed = time.time() - t_start
        
        self._collect_generation_stats(study)
        best_individual = self._select_best(study)
        best_individual = self._baseline_guard(best_individual, study)
        self.save_results(best_individual, study, elapsed)
        self._print_profile_stats()
        self.print_report(best_individual, study, elapsed)
        
        return best_individual, study
    
    def _collect_generation_stats(self, study: optuna.study.Study):
        pop = self.config.population_size
        trials = sorted(study.trials, key=lambda t: t.number)
        for gen_idx in range(self.config.n_generations):
            gen_trials = trials[gen_idx * pop : (gen_idx + 1) * pop]
            if not gen_trials: break
            scores = []
            for t in gen_trials:
                if t.values and len(t.values) == 3:
                    m = {'context_relevance': t.values[0], 'faithfulness': t.values[1], 'answer_relevance': t.values[2]}
                    scores.append(_scalarized_score(m, self.config))
            if scores:
                self.generation_log.append({
                    'generation': gen_idx + 1, 'best_score': max(scores),
                    'mean_score': np.mean(scores), 'worst_score': min(scores), 'std_score': np.std(scores)
                })
    
    def _select_best(self, study: optuna.study.Study) -> Individual:
        best_trial, best_score = None, float('-inf')
        for t in study.best_trials:
            if t.values and len(t.values) == 3:
                m = {'context_relevance': t.values[0], 'faithfulness': t.values[1], 'answer_relevance': t.values[2]}
                score = _scalarized_score(m, self.config)
                if score > best_score:
                    best_score = score
                    best_trial = t
        
        if best_trial is None: raise RuntimeError("Нет валидных trials")
        
        p = best_trial.params
        # ✅ ИСПРАВЛЕНИЕ: .get() с значениями по умолчанию для условных параметров
        return Individual(
            bm25_k=p['bm25_k'],
            vec_k=p['vec_k'],
            rrf_k=p['rrf_k'],
            rerank_k=p['rerank_k'],
            rerank_threshold=round(p['rerank_threshold'], 4),
            final_k=p['final_k'],
            reranker_model=p['reranker_model'],
            reranker_type=p.get('reranker_type', 'auto'),  # ✅ НОВОЕ
            fusion_strategy=p['fusion_strategy'],
            hybrid_alpha=p.get('hybrid_alpha', 0.5),  # ✅ Защита от KeyError
            retrieval_mode=p['retrieval_mode'],
            adaptive_threshold=p.get('adaptive_threshold', 0.05),  # ✅ Защита от KeyError
            fitness=best_score,
            metrics=self.eval_cache.get(_cache_key(p), {})
        )
    
    def _baseline_guard(self, best: Individual, study: optuna.study.Study) -> Individual:
            bl_key = _cache_key(self.config.baseline_params)
            if bl_key not in self.eval_cache:
                logger.warning("⚠ Бейзлайн не оценён")
                return best
            
            bl_metrics = self.eval_cache[bl_key]
            bl_score = _scalarized_score(bl_metrics, self.config)
            
            logger.info(f"\n🛡️ Baseline Guard: Бейзлайн W={bl_score:.4f} vs ГА W={best.fitness:.4f}")
            
            if bl_score == 0:
                logger.warning("   ⚠ Бейзлайн = 0, пропускаем сравнение")
                return best
            
            if best.fitness >= bl_score:
                logger.info(f"   ✅ ГА улучшил бейзлайн на {((best.fitness - bl_score)/bl_score)*100:.2f}%")
                return best
            else:
                logger.warning(f"   ⚠ ГА хуже бейзлайна. Возвращаем бейзлайн.")
                bp = self.config.baseline_params
                return Individual(
                    bm25_k=bp['bm25_k'],
                    vec_k=bp['vec_k'],
                    rrf_k=bp['rrf_k'],
                    rerank_k=bp['rerank_k'],
                    rerank_threshold=bp['rerank_threshold'],
                    final_k=bp['final_k'],
                    reranker_model=bp['reranker_model'],
                    reranker_type=bp.get('reranker_type', 'auto'),
                    fusion_strategy=bp['fusion_strategy'],
                    hybrid_alpha=bp['hybrid_alpha'],
                    retrieval_mode=bp['retrieval_mode'],
                    adaptive_threshold=bp['adaptive_threshold'],
                    fitness=bl_score,
                    metrics=bl_metrics
                )
    
    def save_results(self, best: Individual, study: optuna.study.Study, elapsed: float):
        bl_key = _cache_key(self.config.baseline_params)
        bl_metrics = self.eval_cache.get(bl_key, {})
        bl_score = _scalarized_score(bl_metrics, self.config) if bl_metrics else 0
        
        results = {
            'algorithm': 'NSGA-II (SOTA Extended)',
            'optimized_params': {k: getattr(best, k) for k in [
                'bm25_k', 'vec_k', 'rrf_k', 'rerank_k', 'rerank_threshold', 'final_k',
                'reranker_model', 'reranker_type', 'fusion_strategy', 'hybrid_alpha',
                'retrieval_mode', 'adaptive_threshold'
            ]},
            'best_metrics': best.metrics,
            'best_weighted_score': best.fitness,
            'baseline_weighted_score': bl_score,
            'improvement_pct': ((best.fitness - bl_score) / bl_score * 100) if bl_score > 0 else 0,
            'ga_config': asdict(self.config),
            'pareto_front_size': len(study.best_trials),
            'generation_stats': self.generation_log,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        with open(self.output_dir / "optimization_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # CSV История
        with open(self.output_dir / "trial_history.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'trial', 'gen', 'CR', 'Faith', 'AR', 'W-Score',
                'bm25_k', 'vec_k', 'rrf_k', 'rerank_k', 'thresh', 'fk',
                'reranker', 'reranker_type', 'fusion', 'alpha', 'mode', 'adapt_thresh'
            ])
            pop = self.config.population_size
            for t in sorted(study.trials, key=lambda x: x.number):
                if not t.values or len(t.values) < 3: continue
                cr, fa, ar = t.values[:3]
                ws = _scalarized_score({'context_relevance': cr, 'faithfulness': fa, 'answer_relevance': ar}, self.config)
                writer.writerow([
                    t.number, t.number // pop + 1, f"{cr:.4f}", f"{fa:.4f}", f"{ar:.4f}", f"{ws:.4f}",
                    t.params['bm25_k'], t.params['vec_k'], t.params['rrf_k'], t.params['rerank_k'],
                    f"{t.params['rerank_threshold']:.4f}", t.params['final_k'],
                    t.params['reranker_model'],
                    t.params.get('reranker_type', 'auto'),  # ✅ НОВОЕ
                    t.params['fusion_strategy'],
                    f"{t.params.get('hybrid_alpha', 0.5):.2f}",  # ✅ Защита
                    t.params['retrieval_mode'],
                    f"{t.params.get('adaptive_threshold', 0.05):.3f}"  # ✅ Защита
                ])
        
        # Python файл с параметрами
        code = f"""# Оптимальные параметры SOTA RAG
OPTIMIZED_PARAMS = {{
    'bm25_k': {best.bm25_k}, 'vec_k': {best.vec_k}, 'rrf_k': {best.rrf_k},
    'rerank_k': {best.rerank_k}, 'rerank_threshold': {best.rerank_threshold}, 'final_k': {best.final_k},
    'reranker_model': "{best.reranker_model}",
    'reranker_type': "{best.reranker_type}",
    'fusion_strategy': "{best.fusion_strategy}",
    'hybrid_alpha': {best.hybrid_alpha},
    'retrieval_mode': "{best.retrieval_mode}",
    'adaptive_threshold': {best.adaptive_threshold},
}}
METRICS = {{'cr': {best.metrics.get('context_relevance', 0):.4f}, 'faith': {best.metrics.get('faithfulness', 0):.4f}, 'ar': {best.metrics.get('answer_relevance', 0):.4f}}}
SCORE = {best.fitness:.4f} (Baseline: {bl_score:.4f})
"""
        (self.output_dir / "optimized_params.py").write_text(code, encoding="utf-8")
        
        # Генерация графиков
        self._plot_results(study, best)
        self._plot_reranker_comparison(study)
        self._plot_pareto_evolution(study)
        self._plot_metrics_evolution(study)          # новое
        self._plot_correlation_matrix(study)         # новое
        
        logger.info(f"💾 Результаты сохранены в {self.output_dir}/")
    
    def _print_profile_stats(self):
        print("\n" + "="*60)
        print("📊 ПРОФИЛИРОВАНИЕ ВРЕМЕНИ (среднее на 1 вопрос)")
        print("="*60)
        for stage in ['retrieval', 'llm', 'cr_embed', 'faith', 'ar_embed', 'ask_total', 'total_single']:
            times = self.profile_times.get(stage, [])
            if times:
                mean = np.mean(times)
                p95 = np.percentile(times, 95)
                print(f"{stage:18s} → {mean*1000:6.1f} ms (95% = {p95*1000:6.1f} ms)")
        print("="*60)
        print(f"\n💾 LLM Cache: hits={self.llm_cache_hits}, misses={self.llm_cache_misses}, hit_rate={self.llm_cache_hits/(self.llm_cache_hits+self.llm_cache_misses+1e-9)*100:.1f}%")
    
    # ----------------------------------------------------------------------
    # ГРАФИКИ
    # ----------------------------------------------------------------------
    def _setup_academic_style(self):
        """Настройка академического стиля для matplotlib."""
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 13
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'

    def _plot_results(self, study: optuna.study.Study, best: Individual):
        """Парето-фронт и сходимость взвешенного скора."""
        if not MATPLOTLIB_AVAILABLE:
            return
        try:
            self._setup_academic_style()
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # --- Pareto Front (CR vs Faithfulness) ---
            ax = axes[0]
            all_cr = [t.values[0] for t in study.trials if t.values and len(t.values)==3]
            all_fa = [t.values[1] for t in study.trials if t.values and len(t.values)==3]
            ax.scatter(all_cr, all_fa, c='lightgray', s=30, alpha=0.5, label='Все trials', edgecolors='none')
            
            pareto_cr = [t.values[0] for t in study.best_trials if len(t.values)==3]
            pareto_fa = [t.values[1] for t in study.best_trials if len(t.values)==3]
            ax.scatter(pareto_cr, pareto_fa, c='red', s=60, edgecolors='black', linewidth=0.5, label='Парето-фронт', zorder=3)
            
            # Baseline and best
            bl_key = _cache_key(self.config.baseline_params)
            if bl_key in self.eval_cache:
                bl = self.eval_cache[bl_key]
                ax.scatter(bl['context_relevance'], bl['faithfulness'], c='blue', s=150, marker='*', 
                           edgecolors='black', linewidth=0.8, label='Базовый набор', zorder=4)
            ax.scatter(best.metrics.get('context_relevance',0), best.metrics.get('faithfulness',0),
                      c='green', s=120, marker='D', edgecolors='black', linewidth=0.8, label='Лучшее решение', zorder=4)
            
            ax.set_xlabel('Context Relevance', fontweight='bold')
            ax.set_ylabel('Faithfulness', fontweight='bold')
            ax.set_title('Парето-фронт: релевантность контекста vs фактичность', fontweight='bold')
            ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black')
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
            # --- Convergence of Weighted Score ---
            ax = axes[1]
            if self.generation_log:
                gens = [g['generation'] for g in self.generation_log]
                bests = [g['best_score'] for g in self.generation_log]
                means = [g['mean_score'] for g in self.generation_log]
                ax.plot(gens, bests, 'g-o', linewidth=1.5, markersize=4, label='Лучший в поколении')
                ax.plot(gens, means, 'b--', linewidth=1.5, markersize=4, label='Средний по поколению')
                if bl_key in self.eval_cache:
                    bl_score = _scalarized_score(self.eval_cache[bl_key], self.config)
                    ax.axhline(y=bl_score, color='red', linestyle=':', linewidth=1.5, label=f'Базовый набор ({bl_score:.3f})')
                ax.set_xlabel('Поколение', fontweight='bold')
                ax.set_ylabel('Взвешенный Score', fontweight='bold')
                ax.set_title('Сходимость алгоритма', fontweight='bold')
                ax.legend(loc='lower right', frameon=True, fancybox=False)
                ax.grid(True, linestyle='--', alpha=0.4)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "sota_results.png", dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / "sota_results.pdf", format='pdf', bbox_inches='tight')
            plt.close()
            logger.info("📊 График результатов сохранён (PNG + PDF)")
        except Exception as e:
            logger.warning(f"Не удалось построить график результатов: {e}")

    def _plot_reranker_comparison(self, study: optuna.study.Study):
        """Горизонтальный boxplot + swarmplot с удалёнными выбросами."""
        if not MATPLOTLIB_AVAILABLE:
            return
        try:
            self._setup_academic_style()
            import pandas as pd
            if 'seaborn' in globals():
                sns.set_theme(style="whitegrid", font_scale=1.1)

            # Собираем данные
            records = []
            for t in study.trials:
                if t.values and len(t.values) == 3:
                    model = t.params.get('reranker_model', 'unknown').split('/')[-1]
                    score = _scalarized_score({
                        'context_relevance': t.values[0],
                        'faithfulness': t.values[1],
                        'answer_relevance': t.values[2]
                    }, self.config)
                    records.append({'model': model, 'score': score})
            if not records:
                return
            df = pd.DataFrame(records)

            # --- ОБРЕЗКА ВЫБРОСОВ по IQR для каждой модели ---
            def filter_outliers(group):
                q1 = group['score'].quantile(0.25)
                q3 = group['score'].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                return group[(group['score'] >= lower) & (group['score'] <= upper)]

            df = df.groupby('model', observed=True).apply(filter_outliers).reset_index(drop=True)

            # Сортировка по медиане (после удаления выбросов)
            median_order = df.groupby('model', observed=True)['score'].median().sort_values(ascending=False).index.tolist()
            df['model'] = pd.Categorical(df['model'], categories=median_order, ordered=True)

            # Количество trial'ов после фильтрации
            counts = df.groupby('model', observed=True).size().to_dict()
            labels_with_n = [f"{model}\n(n={counts[model]})" for model in median_order]

            fig, ax = plt.subplots(figsize=(14, 8))

            # Boxplot (выбросы уже удалены, showfliers можно оставить False)
            sns.boxplot(
                data=df, x='score', y='model',
                palette="Blues", width=0.6, linewidth=1.5,
                boxprops=dict(alpha=0.85),
                showfliers=False,   # выбросов нет, но оставляем для ясности
                ax=ax
            )

            # Swarmplot без выбросов
            sns.swarmplot(
                data=df, x='score', y='model',
                color="#2C3E50", size=4.5, alpha=0.85, ax=ax
            )

            # Короткие подписи
            short_labels = [lbl[:37] + '...' if len(lbl) > 40 else lbl for lbl in labels_with_n]
            ax.set_yticks(range(len(short_labels)))
            ax.set_yticklabels(short_labels, fontsize=9, rotation=0, ha='right')
            plt.subplots_adjust(left=0.35)

            ax.set_xlabel('Взвешенный Score', fontweight='bold')
            ax.set_ylabel('Модель реранкера', fontweight='bold')
            ax.set_title('Сравнение эффективности реранкеров', fontweight='bold', pad=20)
            ax.legend(['Медиана + IQR', 'Отдельные trials'], loc='lower right', frameon=True, fancybox=False)
            ax.grid(axis='x', linestyle='--', alpha=0.5)

            plt.tight_layout()
            plt.savefig(self.output_dir / "reranker_comparison.png", dpi=400, bbox_inches='tight')
            plt.savefig(self.output_dir / "reranker_comparison.pdf", format='pdf', bbox_inches='tight')
            plt.close()

            logger.info("📊 График сравнения реранкеров сохранён (выбросы удалены)")
        except Exception as e:
            logger.warning(f"Не удалось построить сравнение реранкеров: {e}")

    def _plot_pareto_evolution(self, study: optuna.study.Study):
        """Эволюция качества по поколениям: разброс CR/Faith, динамика AR, влияние final_k."""
        if not MATPLOTLIB_AVAILABLE:
            return
        try:
            self._setup_academic_style()
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            pop = self.config.population_size
            generations = {}
            
            for t in study.trials:
                if t.values and len(t.values) == 3:
                    gen = t.number // pop
                    if gen not in generations:
                        generations[gen] = []
                    generations[gen].append(t)
            
            # 1. CR vs Faithfulness по поколениям (цветом показаны поколения)
            ax = axes[0]
            colors = plt.cm.viridis(np.linspace(0, 1, len(generations)))
            for idx, (gen, trials) in enumerate(sorted(generations.items())):
                cr = [t.values[0] for t in trials]
                fa = [t.values[1] for t in trials]
                ax.scatter(cr, fa, c=[colors[idx]]*len(cr), s=30, alpha=0.6,
                           label=f'Поколение {gen+1}', edgecolors='black', linewidth=0.3)
            ax.set_xlabel('Context Relevance', fontweight='bold')
            ax.set_ylabel('Faithfulness', fontweight='bold')
            ax.set_title('Эволюция качества: CR vs Faithfulness', fontweight='bold')
            ax.legend(loc='lower right', fontsize=9, ncol=2, frameon=True, fancybox=False)
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
            # 2. Динамика Answer Relevance (среднее и разброс)
            ax = axes[1]
            gen_nums, ar_means, ar_stds, ar_best = [], [], [], []
            for gen, trials in sorted(generations.items()):
                ar = [t.values[2] for t in trials]
                gen_nums.append(gen + 1)
                ar_means.append(np.mean(ar))
                ar_stds.append(np.std(ar))
                ar_best.append(np.max(ar))
            
            ax.errorbar(gen_nums, ar_means, yerr=ar_stds, marker='o', capsize=4,
                        linewidth=1.5, markersize=5, color='blue', alpha=0.7, label='Среднее ± std')
            ax.plot(gen_nums, ar_best, 's-', color='red', linewidth=1.5, markersize=4, label='Лучшее в поколении')
            ax.set_xlabel('Поколение', fontweight='bold')
            ax.set_ylabel('Answer Relevance', fontweight='bold')
            ax.set_title('Стабильность и улучшение ответа', fontweight='bold')
            ax.legend(loc='lower right', frameon=True, fancybox=False)
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.set_ylim(0, 1)
            
            # 3. Влияние final_k (boxplot)
            ax = axes[2]
            final_k_vals = sorted(set(t.params.get('final_k', 5) for t in study.trials
                                     if t.values and 'final_k' in t.params))
            scores_by_k = {}
            for k in final_k_vals:
                scores = [_scalarized_score({
                    'context_relevance': t.values[0],
                    'faithfulness': t.values[1],
                    'answer_relevance': t.values[2]
                }, self.config) for t in study.trials
                         if t.values and t.params.get('final_k') == k]
                scores_by_k[k] = scores
            
            data = [scores_by_k[k] for k in final_k_vals]
            ax.boxplot(data, tick_labels=final_k_vals, patch_artist=True,
                       boxprops=dict(facecolor='lightcoral', color='darkred', alpha=0.7, linewidth=1.2),
                       medianprops=dict(color='darkred', linewidth=2),
                       whiskerprops=dict(color='darkred', linewidth=1),
                       capprops=dict(color='darkred', linewidth=1))
            ax.set_xlabel('final_k', fontweight='bold')
            ax.set_ylabel('Взвешенный Score', fontweight='bold')
            ax.set_title('Влияние количества итоговых контекстов', fontweight='bold')
            ax.grid(axis='y', linestyle='--', alpha=0.4)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "pareto_evolution.png", dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / "pareto_evolution.pdf", format='pdf', bbox_inches='tight')
            plt.close()
            logger.info("📊 График эволюции Парето-фронта сохранён (PNG + PDF)")
        except Exception as e:
            logger.warning(f"Не удалось построить эволюцию: {e}")

    def _plot_metrics_evolution(self, study: optuna.study.Study):
        """График эволюции всех трёх метрик (средние и лучшие по поколениям)."""
        if not MATPLOTLIB_AVAILABLE:
            return
        try:
            self._setup_academic_style()
            pop = self.config.population_size
            generations = {}
            for t in study.trials:
                if t.values and len(t.values) == 3:
                    gen = t.number // pop
                    if gen not in generations:
                        generations[gen] = []
                    generations[gen].append(t)
            
            if not generations:
                return
            
            gen_nums = sorted(generations.keys())
            cr_means, fa_means, ar_means = [], [], []
            cr_best, fa_best, ar_best = [], [], []
            
            for gen in gen_nums:
                trials = generations[gen]
                cr_vals = [t.values[0] for t in trials]
                fa_vals = [t.values[1] for t in trials]
                ar_vals = [t.values[2] for t in trials]
                cr_means.append(np.mean(cr_vals))
                fa_means.append(np.mean(fa_vals))
                ar_means.append(np.mean(ar_vals))
                cr_best.append(np.max(cr_vals))
                fa_best.append(np.max(fa_vals))
                ar_best.append(np.max(ar_vals))
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Средние значения
            ax = axes[0]
            ax.plot(gen_nums, cr_means, 'o-', label='CR (средн.)', color='blue', linewidth=1.5, markersize=4)
            ax.plot(gen_nums, fa_means, 's-', label='Faithfulness (средн.)', color='green', linewidth=1.5, markersize=4)
            ax.plot(gen_nums, ar_means, 'd-', label='Answer Relevance (средн.)', color='red', linewidth=1.5, markersize=4)
            ax.set_xlabel('Поколение', fontweight='bold')
            ax.set_ylabel('Значение метрики', fontweight='bold')
            ax.set_title('Эволюция средних значений метрик', fontweight='bold')
            ax.legend(loc='lower right')
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.set_ylim(0, 1)
            
            # Лучшие значения
            ax = axes[1]
            ax.plot(gen_nums, cr_best, 'o-', label='CR (лучш.)', color='blue', linewidth=1.5, markersize=4)
            ax.plot(gen_nums, fa_best, 's-', label='Faithfulness (лучш.)', color='green', linewidth=1.5, markersize=4)
            ax.plot(gen_nums, ar_best, 'd-', label='Answer Relevance (лучш.)', color='red', linewidth=1.5, markersize=4)
            ax.set_xlabel('Поколение', fontweight='bold')
            ax.set_ylabel('Значение метрики', fontweight='bold')
            ax.set_title('Эволюция лучших значений метрик', fontweight='bold')
            ax.legend(loc='lower right')
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.set_ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "metrics_evolution.png", dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / "metrics_evolution.pdf", format='pdf', bbox_inches='tight')
            plt.close()
            logger.info("📊 График эволюции метрик сохранён (PNG + PDF)")
        except Exception as e:
            logger.warning(f"Не удалось построить эволюцию метрик: {e}")

    def _plot_correlation_matrix(self, study: optuna.study.Study):
        """Матрица корреляций между числовыми параметрами и метриками."""
        if not MATPLOTLIB_AVAILABLE:
            return
        try:
            # Собираем данные
            data = []
            for t in study.trials:
                if t.values and len(t.values) == 3:
                    row = {
                        'bm25_k': t.params.get('bm25_k', 0),
                        'vec_k': t.params.get('vec_k', 0),
                        'rrf_k': t.params.get('rrf_k', 0),
                        'rerank_k': t.params.get('rerank_k', 0),
                        'rerank_threshold': t.params.get('rerank_threshold', 0),
                        'final_k': t.params.get('final_k', 0),
                        'hybrid_alpha': t.params.get('hybrid_alpha', 0.5),
                        'adaptive_threshold': t.params.get('adaptive_threshold', 0.05),
                        'CR': t.values[0],
                        'Faithfulness': t.values[1],
                        'Answer_Relevance': t.values[2],
                        'Weighted_Score': _scalarized_score({
                            'context_relevance': t.values[0],
                            'faithfulness': t.values[1],
                            'answer_relevance': t.values[2]
                        }, self.config)
                    }
                    data.append(row)
            
            if not data:
                return
            
            import pandas as pd
            df = pd.DataFrame(data)
            # Выбираем только числовые колонки
            corr = df.select_dtypes(include=[np.number]).corr()
            
            self._setup_academic_style()
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                        square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
            ax.set_title('Корреляционная матрица параметров и метрик', fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.output_dir / "correlation_matrix.png", dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / "correlation_matrix.pdf", format='pdf', bbox_inches='tight')
            plt.close()
            logger.info("📊 Матрица корреляций сохранена (PNG + PDF)")
        except Exception as e:
            logger.warning(f"Не удалось построить матрицу корреляций: {e}")
    
    def print_report(self, best: Individual, study: optuna.study.Study, elapsed: float):
        bl_key = _cache_key(self.config.baseline_params)
        bl_metrics = self.eval_cache.get(bl_key, {})
        bl_score = _scalarized_score(bl_metrics, self.config) if bl_metrics else 0
        impr = ((best.fitness - bl_score) / bl_score * 100) if bl_score > 0 else 0
        
        print("\n" + "="*80)
        print("  🏆 ОПТИМИЗАЦИЯ SOTA RAG ЗАВЕРШЕНА")
        print("="*80)
        print(f"⏱ Время: {elapsed/60:.1f} мин | Trials: {len(study.trials)}")
        print(f"🎯 Лучший Score: {best.fitness:.4f} (Бейзлайн: {bl_score:.4f}, {'+' if impr >=0 else ''}{impr:.2f}%)")
        print(f"\n🔧 Лучшие параметры:")
        print(f"   Реранкер:      {best.reranker_model} (тип: {best.reranker_type})")
        print(f"   Слияние:       {best.fusion_strategy} (Alpha={best.hybrid_alpha:.2f})")
        print(f"   Режим поиска:  {best.retrieval_mode} (Threshold={best.adaptive_threshold:.3f})")
        print(f"   K-параметры:   BM25={best.bm25_k}, Vec={best.vec_k}, Rerank={best.rerank_k}, Final={best.final_k}")
        print(f"\n📊 Метрики:")
        print(f"   CR: {best.metrics.get('context_relevance',0):.3f}, Faith: {best.metrics.get('faithfulness',0):.3f}, AR: {best.metrics.get('answer_relevance',0):.3f}")
        print("="*80)

# ==================== ТОЧКА ВХОДА ====================
def load_test_questions() -> List[Dict]:
    """
    РАСШИРЕННЫЙ ТЕСТОВЫЙ НАБОР ДЛЯ ОЦЕНКИ СИСТЕМЫ
    ДОПОЛНЕННОГО ПОИСКА (19 вопросов)
    
    Каждый вопрос специально подобран для проверки различных
    аспектов качества работы системы: точности поиска по ключевым
    словам, понимания структуры нормативных документов, работы
    со сложной логикой, условными правилами и извлечения информации
    из таблиц и приложений.
    """
    return [
        # 🔹 Тест 1: Точный поиск по коду расценки (проверка точного совпадения по нормативным кодам)
        {
            "question": "Что такое 01-01-009-11?",
            "reference": "Разработка грунта в траншеях экскаватором \"обратная лопата\" с ковшом вместимостью 0,65 м³, группа грунтов: 5"
        },
        # 🔹 Тест 2: Структурный вопрос по документу (проверка поиска информации, требующей анализа нескольких разделов)
        {
            "question": "Сколько глав в сводном сметном расчете? Что находится в 3 главе?",
            "reference": "12 глав, Глава 3 включает объекты подсобного и обслуживающего назначения, такие как склады, столовые, медпункты, административные здания и другие вспомогательные сооружения, необходимые для обеспечения нормальной работы основных объектов строительства."
        },
        # 🔹 Тест 3: Нормативные коэффициенты (проверка извлечения общих правил из методических документов)
        {
            "question": "Какие коэффициенты применяются при работе в зимних условиях?",
            "reference": "НДЗ являются среднегодовыми и применяются на весь период строительства."
        },
        # 🔹 Тест 4: Поиск по описанию работ (проверка поиска по смысловому описанию)
        {
            "question": "Какую расценку применить для подвесных потолков Armstrong в офисном здании?",
            "reference": "Наиболее подходящей является расценка ФЕР15-01-047-15 - установка подвесных потолков из минераловатных плит типа \"Армстронг\"."
        },
        # 🔹 Тест 5: Сложное разъяснение коэффициентов (проверка понимания условных правил и исключений)
        {
            "question": "Разъясните возможность применения коэффициента 1,5 «Производство ремонтно-строительных работ осуществляется в жилых помещениях без расселения» ...",
            "reference": "Коэффициент 1,5 применяется при капитальном ремонте многоквартирных домов (МКД) без расселения к нормам затрат труда рабочих строителей и затратам на эксплуатацию машин только для работ, выполняемых непосредственно в жилых помещениях (например, замена полов). При производстве работ по замене инженерных сетей, проходящих через квартиры МКД (стояки ХВС, ГВС и иных), данный коэффициент не применим. Однако при выполнении таких работ следует учитывать другие коэффициенты условий труда согласно МДС 81-35.2004, если имеет место пересечение людских потоков в местах общего пользования здания."
        },
        # 🔹 Тест 6: Проектные работы со встроенными помещениями (проверка работы со сложными корректирующими правилами)
        {
            "question": "Как расценивается проектирование встраиваемых помещений?",
            "reference": "Условия проектирования объединенных или сблокированных зданий и сооружений, а также зданий со встроенными помещениями другого назначения при определении стоимости проектных работ по ценам НЗ на проектные работы учитываются применением в расчете соответствующих корректирующих коэффициентов на сокращенный объем работ с учетом следующих положений, если иное не установлено в НЗ на проектные работы: а) стоимость проектных работ для основного здания (сооружения) определяется с применением цен на проектные работы исходя из натурального показателя, установленного для этого объекта, с ценообразующим коэффициентом 1; б) стоимость проектных работ для встраиваемых помещений (сооружений) в основное здание (сооружение) определяется с применением цен НЗ на проектные работы исходя из натурального показателя, установленного для встраиваемого объекта, с ценообразующим коэффициентом в размере до 0,5; в) стоимость подготовки проектной и рабочей документации зданий (сооружений) сблокированных с основным определяется по ценам НЗ на проектные работы исходя из натурального показателя, установленного для сблокированного объекта, с корректирующим коэффициентом в размере до 0,8. (в ред. Приказа Минстроя РФ от 08.06.2023 N 409/пр)"
        },
        # 🔹 Тест 7: Методика исчисления объёма работ (проверка технической точности извлечения правил)
        {
            "question": "Как исчисляется объем работ на прокладку кабельной продукции?",
            "reference": "Объем работ по прокладке электрического кабеля следует определять по всей проектной длине трассы или линии (кабельной линии), исчисляемой как длина в плане (промеренная по чертежам методом суммирования) с учетом изгибов и поворотов."
        },
        # 🔹 Тест 8: Детальный разбор пункта нормативной таблицы (проверка извлечения из узкоспецифичных разделов)
        {
            "question": "Согласно пункту 1.20.22 раздела I «Общие положения» сборника 20 «Вентиляция и кондиционирование воздуха» ГЭСН 81-02-20-2022 в нормах таблиц 20-06-018, 20-06-019 учтены затраты на прокладку каждого типа коммуникационных трасс (медные трубки, дренаж, питающий кабель) до 10 м. Какие медные трубки и какой длины имеются ввиду?",
            "reference": "В нормах таблиц 20-06-018, 20-06-019 сборника 20 «Вентиляция и кондиционирование воздуха» ГЭСН 81-02-20-2022 учтены затраты на прокладку каждого типа коммуникационных трасс: медные трубки для транспортировки хладагента: две трубки (до 10 м каждая); дренаж (до 10 м); питающий кабель (до 10 м)."
        },
        # 🔹 Тест 9: Нормативное определение (проверка извлечения концептуальных определений)
        {
            "question": "Что такое сметная стоимость строительства, реконструкции, капитального ремонта, сноса объектов капитального строительства, работ по сохранению объектов культурного наследия согласно Методике 421/пр?",
            "reference": "Согласно пункту 30 статьи 1 Градостроительного кодекса Российской Федерации сметная стоимость строительства, реконструкции, капитального ремонта, сноса объектов капитального строительства, работ по сохранению объектов культурного наследия (далее — сметная стоимость строительства) — расчетная стоимость строительства, реконструкции, капитального ремонта, сноса объектов капитального строительства, работ по сохранению объектов культурного наследия, подлежащая определению на этапе архитектурно-строительного проектирования, подготовки сметы на снос объекта капитального строительства и применению в соответствии со статьей 8.3 Градостроительного кодекса Российской Федерации."
        },
        # 🔹 Тест 10: Условия применения метода (проверка извлечения условных правил)
        {
            "question": "В каких случаях при определении сметной стоимости применяется конъюнктурный анализ?",
            "reference": "При отсутствии в ФГИС ЦС данных о сметных ценах в базисном или текущем уровне цен на отдельные материальные ресурсы и оборудование, а также сметных нормативов на отдельные виды работ и услуг допускается определение их сметной стоимости по наиболее экономичному варианту, определенному на основании сбора информации о текущих ценах (далее — конъюнктурный анализ)."
        },
        # 🔹 Тест 11: Коэффициенты в стеснённых условиях (проверка сложных условий применения)
        {
            "question": "Какой коэффициент применяется к затратам труда рабочих при производстве работ в стесненных условиях в населенных пунктах?",
            "reference": "Коэффициент, учитывающий стесненные условия в населенных пунктах, определяется наличием трех из перечисленных факторов: интенсивное движение городского транспорта и пешеходов в непосредственной близости (в пределах 50 метров) от зоны производства работ; сети подземных коммуникаций, подлежащие перекладке или подвеске; расположение объектов капитального строительства и сохраняемых зеленых насаждений в непосредственной близости (в пределах 50 метров) от зоны производства работ; стесненные условия или невозможность складирования материалов; ограничение поворота стрелы грузоподъемного крана."
        },
        # 🔹 Тест 12: Мета-вопрос о методике (проверка понимания общих положений)
        {
            "question": "Что устанавливает Методика определения стоимости работ по подготовке проектной документации?",
            "reference": "Методика определения стоимости работ по подготовке проектной документации (далее - Методика) устанавливает порядок определения сметной стоимости работ по подготовке проектной и (или) рабочей документации для строительства, реконструкции, капитального ремонта, сноса объектов капитального строительства, работ по сохранению объектов культурного наследия на основании сметных нормативов на работы по подготовке проектной и (или) рабочей документации, нормативных затрат на работы по подготовке проектной документации (далее - НЗ на проектные работы), а также порядок разработки НЗ на проектные работы."
        },
        # 🔹 Тест 13: Формула расчёта цены (проверка извлечения математических формул)
        {
            "question": "По какой формуле рассчитывается цена проектных работ при использовании параметров цены в зависимости от натуральных показателей?",
            "reference": "Цена проектных работ (Ц) рассчитывается по формуле: Ц = a + b × X, где: Ц — цена проектных работ, тыс. руб.; a — параметр цены проектных работ: постоянная величина, выраженная в тыс. руб.; b — параметр цены проектных работ: постоянная величина, имеющая размерность тыс. руб. на единицу натурального показателя; X — величина натурального показателя проектируемого объекта."
        },
        # 🔹 Тест 14: Информационное моделирование (проверка современных нормативов)
        {
            "question": "Как определяется стоимость работ по подготовке проектной документации, содержащей материалы в форме информационной модели?",
            "reference": "Стоимость работ по подготовке проектной документации, содержащей материалы в форме информационной модели, рассчитывается по формуле: СИМП = СП × (ΔИМП × КИМ + ΔТГП) × КПД, где СП - цена разработки проектной и рабочей документации, определяемая по справочникам базовых цен; ΔИМП - сумма долей относительных стоимостей разработки разделов ПЗУ, ППО, АР, КР, ТКР, ИЛО, ПОС, СМ, подразделов ЭО, ВС, ВО, ОВ, СС, ГС, ТХ раздела ИОС, включаемых в трехмерную модель; КИМ - корректирующий коэффициент в зависимости от вида объекта (1,09-1,31); ΔТГП - сумма долей разработки разделов ПЗ, ООС, МПОБ, МОДИ, ЭЭФ, размещаемых в среде общих данных; КПД - доля стоимости работ по подготовке проектной документации (40-60%)."
        },
        # 🔹 Тест 15: Корректирующие коэффициенты для объектов (проверка извлечения списков значений)
        {
            "question": "Какие корректирующие коэффициенты применяются для объектов жилищно-гражданского строительства при определении стоимости работ с применением информационного моделирования?",
            "reference": "Для объектов жилищно-гражданского строительства применяются следующие корректирующие коэффициенты: кирпичный многоквартирный дом - 1,17; крупнопанельный, монолитный многоквартирный дом - 1,18; офисное здание - 1,21; здание банка - 1,22; здание концертного зала - 1,15; здание театра, цирка - 1,16; гипермаркет, универсам - 1,18; здание гостиницы - 1,17; бассейн крытый - 1,13; здание больницы общего профиля - 1,17; здание поликлиники - 1,19; здание общежития - 1,14; здание стоянки закрытого типа отапливаемой - 1,11."
        },
        # 🔹 Тест 16: Численность технического заказчика (проверка работы с таблицами диапазонов)
        {
            "question": "Как определяется численность работников технического заказчика?",
            "reference": "Численность работников технического заказчика определяется согласно показателям, приведенным в Приложении № 2 к Методике, исходя из величины сметной стоимости строительства объекта по итогам глав 1-9 и 12 ССРСС, определенной в уровне цен по состоянию на 1 января 2023 г. с округлением до целого числа, млн. рублей. При сметной стоимости строительства до 600 млн. рублей численность составляет 1 человек; от 600 до 1200 млн. рублей - 3 человека; от 1200 до 1500 - 4; от 1500 до 1800 - 5; от 1800 до 2350 - 6; от 2350 до 3500 - 8; от 3500 до 4700 - 9; от 4700 до 5900 - 10; от 5900 до 7000 - 11; от 7000 до 8800 - 13; от 8800 до 10600 - 15 человек. При сметной стоимости свыше 10600 млн. рублей численность увеличивается на одного человека на каждые 1100 млн. рублей сверх указанной суммы."
        },
        # 🔹 Тест 17: Затраты на функции технического заказчика (проверка извлечения сложных перечислений)
        {
            "question": "Какие затраты включаются в расчет на осуществление функций технического заказчика?",
            "reference": "Затраты на осуществление функций технического заказчика включают: затраты на оплату труда (определяются исходя из численности работников и размера средней заработной платы, не превышающего произведение среднемесячного размера оплаты труда рабочего четвертого разряда на коэффициент 1,4); страховые взносы; налог на имущество, транспортный, земельный налоги; амортизацию основных средств; командировочные расходы; арендную плату и содержание служебного автотранспорта; арендную плату и содержание зданий и помещений (площадь определяется произведением численности работников, величины площади рабочего места (согласно СанПиН) и коэффициента 1,2); повышение квалификации кадров; приобретение оргтехники, мебели, канцелярских товаров, программного обеспечения, спецодежды; охрану зданий и помещений; услуги связи и интернет; прочие расходы (не более 5% от суммы указанных расходов); премирование за досрочный ввод объекта; надбавку за секретность (при наличии); расходы на организацию геодезических работ; сметную прибыль в размере 10% от суммы всех затрат."
        },
        # 🔹 Тест 18: Точный поиск по коду ремонтных работ (проверка поиска в сборниках ремонтно-строительных работ)
        {
            "question": "Что означает код 65-01-001-01?",
            "reference": "Код 65-01-001-01 соответствует нормативу на разборку трубопроводов из водогазопроводных труб диаметром до 25 мм. Единица измерения — 100 м трубопровода. В состав работ входит: снятие труб и креплений с отборкой годных труб, арматуры, фасонных и крепежных частей; свертывание арматуры; правка и очистка труб от накипи; складирование труб и фасонных частей. Норматив относится к сборнику ГЭСНр (ремонтно-строительные работы), таблица 65-01-001."
        },
        # 🔹 Тест 19: Пусконаладочные работы (проверка поиска в специализированных сборниках)
        {
            "question": "Какой код соответствует пусконаладочным работам по регулировке сети вентиляции с количеством сечений до 5?",
            "reference": "Для пусконаладочных работ по регулировке сети систем вентиляции и кондиционирования воздуха при количестве сечений до 5 применяется код 03-01-022-01. Единица измерения — сеть. Работы включают: подготовительные работы, снятие с натуры схем вентиляционных систем, аэродинамические испытания и сопоставление с проектом объемов воздуха, регулировку сети для достижения проектных показателей, комплексное опробование и обеспечение воздушного баланса. Норматив относится к сборнику ГЭСНп (пусконаладочные работы), таблица 03-01-022."
        }
    ]

def main():
    import argparse
    parser = argparse.ArgumentParser(description="SOTA Оптимизация RAG (NSGA-II)")
    parser.add_argument("--pop", type=int, default=28)
    parser.add_argument("--gens", type=int, default=60)
    parser.add_argument("--eval-runs", type=int, default=1)
    parser.add_argument("--output", type=str, default="./output/ga_optimization")
    args = parser.parse_args()
    
    config = GAConfig(
        population_size=args.pop,
        n_generations=args.gens,
        num_eval_runs=args.eval_runs,
        output_dir=args.output
    )
    
    test_questions = load_test_questions()
    settings = Settings()  # Использует дефолты из rag_gen
    app = SmetaRAGApp(settings)
    
    optimizer = GAOptimizer(config, test_questions, app)
    best, study = optimizer.optimize()

if __name__ == "__main__":
    main() 