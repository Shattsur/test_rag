# -*- coding: utf-8 -*-
"""
SmetaGPT RAG (Gen Version): Hybrid retrieval + rerank + SOTA features.
Исправления:
1. ✅ UnifiedReranker — поддержка CrossEncoder и transformers-моделей (Qwen, Jina)
2. ✅ pool_k передаётся динамически без хаков с s_eff
3. ✅ Поддержка загрузки оптимизированных параметров из optimized_params.py (по умолчанию)
4. ✅ Вывод используемых параметров при запуске
"""
import argparse
import gc
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional
import torch
import torch.nn.functional as F
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from peft import PeftModel


# ==================== ЛОГИРОВАНИЕ ====================
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("smeta_rag_gen")

# ==================== КОНСТАНТЫ ====================
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
_CHROMA_ALLOWED = (str, int, float, bool)

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================
CODE_RE = re.compile(r"\b\d{2}-\d{2}-\d{3}-\d{2}\b")
PART_TRASH_RE = re.compile(r"\s*(част[ьи]\s*\d+)\s*$", re.IGNORECASE)
SOURCES_BLOCK_RE = re.compile(r"\n\s*[{0,2}\s]источники\s*[{0,2}\s]:?\s*\n", re.IGNORECASE)
QUESTION_LINE_RE = re.compile(r"(?im)^\s*вопрос\s*:\s*.*$", re.IGNORECASE)
_SENT_SPLIT_RE = re.compile(r'(?<!\w.\w.)(?<![A-Z][a-z].)(?<=\.|\?|!)\s')

def st(x: Any) -> str:
    return x.strip() if isinstance(x, str) else ""

def uniq(xs: List[str]) -> List[str]:
    seen = set()
    return [cleaned for x in xs if (cleaned := st(x)) and cleaned not in seen and not seen.add(cleaned)]

def strip_part_suffix(title: str) -> str:
    return PART_TRASH_RE.sub("", st(title)).strip()

def doc_sig(d: Document) -> str:
    md = d.metadata if isinstance(d.metadata, dict) else {}
    cid = st(md.get("chunk_id"))
    return f"cid:{cid}" if cid else f"txt:{st(d.page_content)[:220]}"

def dedup_docs(docs: List[Document]) -> List[Document]:
    seen = set()
    return [d for d in docs if (s := doc_sig(d)) and s not in seen and not seen.add(s)]

def truncate_sentence_aware(text: str, cap: int) -> str:
    text = st(text)
    if cap <= 0 or len(text) <= cap:
        return text
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text) if p.strip()]
    out, total = [], 0
    for p in parts:
        add = p if not out else " " + p
        if total + len(add) > cap:
            break
        out.append(add.strip())
        total += len(add)
    s = "".join(out).strip() if out else text[:cap]
    if len(s) < len(text):
        s += "\n...[обрезано]"
    return s

def docs_to_context(docs: List[Document], max_chars: int, per_doc_cap: int) -> str:
    parts, total = [], 0
    for d in docs:
        md = d.metadata if isinstance(d.metadata, dict) else {}
        wc = st(md.get("work_code"))
        title = strip_part_suffix(st(md.get("title")))
        src = st(md.get("source") or md.get("source_file") or md.get("url"))
        head = [f"[{wc}] {title}".strip() if wc or title else ""]
        if src:
            head.append(f"Документ: {src}")
        head = [h for h in head if h]
        text = truncate_sentence_aware(st(d.page_content), per_doc_cap)
        block = ("\n".join(head) + "\n" + text).strip() if head else text.strip()
        if not block:
            continue
        add_len = len(block) + (5 if parts else 0)
        if total + add_len > max_chars:
            break
        parts.append(block)
        total += add_len
    return "\n\n---\n\n".join(parts).strip()

def normalize_source(md: Dict[str, Any]) -> Dict[str, str]:
    wc = st(md.get("work_code"))
    title = strip_part_suffix(st(md.get("title")))
    src = st(md.get("source") or md.get("source_file") or md.get("url"))
    clause = st(md.get("clause") or md.get("section") or md.get("paragraph"))
    page = st(md.get("page"))
    anchors = [f"п. {clause}" if clause else "", f"стр. {page}" if page else ""]
    anchor = f" ({', '.join(a for a in anchors if a)})" if any(anchors) else ""
    return {"work_code": wc, "title": title, "source_file": src, "anchor": anchor}

def collect_sources(docs: List[Document], limit: int = 25) -> List[Dict[str, str]]:
    seen = set()
    out = []
    for d in docs:
        md = d.metadata if isinstance(d.metadata, dict) else {}
        item = normalize_source(md)
        key = (item["work_code"], item["title"], item["source_file"], item["anchor"])
        if any(item.values()) and key not in seen:
            seen.add(key)
            out.append(item)
        if len(out) >= limit:
            break
    return out

def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<!\w.\w.)(?<![A-Z][a-z].)(?<=\.|\?|!)\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 15 and not re.match(r'^[\W\d]+$', s.strip())]

def _coerce_metadata_value(v: Any) -> Optional[Any]:
    if v is None:
        return None
    if isinstance(v, _CHROMA_ALLOWED):
        return v
    if isinstance(v, (list, dict)):
        s = json.dumps(v, ensure_ascii=False)
        return s if s else None
    try:
        s = str(v).strip()
        return s if s else None
    except Exception:
        return None

def normalize_chunk_metadata(obj: Dict[str, Any], chunk_id: str) -> Dict[str, Any]:
    raw_md: Dict[str, Any] = obj.get("metadata", {})
    text: str = obj.get("text", "")
    md: Dict[str, Any] = {}
    md["chunk_id"] = chunk_id
    md["text_length"] = len(text)
    source = st(obj.get("source") or raw_md.get("source") or raw_md.get("source_file") or "")
    if source:
        md["source"] = source
    doc_type = st(raw_md.get("doc_type") or raw_md.get("type") or "unknown")
    md["doc_type"] = doc_type
    _optional_str_fields = [
        ("work_code", ["work_code"]), ("question", ["question"]), ("answer", ["answer"]),
        ("title", ["title"]), ("clause", ["clause"]), ("subclause", ["subclause"]),
        ("clause_hierarchy", ["clause_hierarchy"]), ("clause_title", ["clause_title"]),
        ("extraction", ["extraction"]), ("section", ["section"]), ("subsection", ["subsection"]),
    ]
    for target_key, source_keys in _optional_str_fields:
        for sk in source_keys:
            val = st(raw_md.get(sk) or obj.get(sk) or "")
            if val:
                md[target_key] = val
                break
    for key in ("page_start", "page_end", "chunk_index", "total_chunks"):
        raw_val = raw_md.get(key)
        if raw_val is not None:
            try:
                md[key] = int(raw_val)
            except (ValueError, TypeError):
                pass
    clean_md: Dict[str, Any] = {}
    for k, v in md.items():
        coerced = _coerce_metadata_value(v)
        if coerced is None:
            continue
        if isinstance(coerced, str) and not coerced:
            continue
        clean_md[k] = coerced
    return clean_md

# ==================== КЭШИРОВАНИЕ РЕРАНКЕРОВ ====================
_RERANKER_CACHE: Dict[str, Any] = {}

def get_cached_reranker(model_name: str, device: str, max_length: int, reranker_type: str = "auto") -> Any:
    """Глобальный кэш реранкеров для ускорения оптимизации."""
    cache_key = f"{model_name}|{max_length}|{device}|{reranker_type}"
    if cache_key not in _RERANKER_CACHE:
        logger.info(f"🔄 Загрузка реранкера в кэш: {model_name} (тип: {reranker_type})...")
        t0 = time.time()
        _RERANKER_CACHE[cache_key] = UnifiedReranker(
            model_name, device=device, max_length=max_length, reranker_type=reranker_type
        )
        t1 = time.time()
        logger.info(f"✅ Реранкер {model_name} загружен за {t1-t0:.2f} сек.")
    return _RERANKER_CACHE[cache_key]

# ==================== UNIFIED RERANKER ====================
class UnifiedReranker:
    """
    Унифицированный враппер для реранкеров.
    Поддерживает: CrossEncoder, SequenceClassification, CausalLM, Jina v3.
    """
    
    TRANSFORMERS_ONLY_MODELS = [
        "qwen", "jina", "jinaai", "Qwen", "Jina",
        "E2Rank", "Alibaba", "Alibaba-NLP", "bge-reranker",
        "nemotron", "Nemotron"
    ]
    
    def __init__(self, model_name: str, device: str, max_length: int, reranker_type: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.reranker_type = reranker_type
        self.model = None
        self.tokenizer = None
        self.is_cross_encoder = False
        self.is_causal_lm = False
        self.is_jina_v3 = "jina-reranker-v3" in model_name.lower()
        
        if reranker_type == "auto":
            self.is_cross_encoder = self._detect_cross_encoder(model_name)
            self.is_causal_lm = any(p in model_name.lower() for p in ["qwen", "qwen3"])
        elif reranker_type == "cross_encoder":
            self.is_cross_encoder = True
        elif reranker_type == "causal_lm":
            self.is_causal_lm = True
        
        self._load_model()
    
    def _detect_cross_encoder(self, model_name: str) -> bool:
        for pattern in self.TRANSFORMERS_ONLY_MODELS:
            if pattern.lower() in model_name.lower():
                return False
        return True
    
    def _load_model(self):
        """
        Загружает модель с поддержкой:
        ✅ Локальные адаптеры + локальная база
        ✅ Локальные адаптеры + база с HF (фоллбэк)
        ✅ Адаптеры с HF + база с HF
        ✅ Merged локальные модели
        ✅ Стандартные HF модели
        ✅ 4-bit квантование на лету при загрузке с HF
        """
        
        # ===================== CROSS-ENCODER =====================
        if self.is_cross_encoder:
            self.model = CrossEncoder(
                self.model_name, device=self.device,
                max_length=self.max_length, trust_remote_code=True
            )
            self.tokenizer = None
            logger.info(f"✅ CrossEncoder загружен: {self.model_name}")
            return

        # ===================== 4-BIT КОНФИГУРАЦИЯ =====================
        # 🔥 Квантуем ВСЕГДА, если это не CrossEncoder
        # Это позволяет грузить базу с HF в 4-bit "на лету"
        quant_config = None
        try:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            logger.info(f"🔧 Используем 4‑битное квантование для {self.model_name}")
        except ImportError:
            logger.warning("⚠️ bitsandbytes не установлен — загрузка без квантования")
            quant_config = None

        # ===================== ТОКЕНИЗАТОР =====================
        tokenizer_kwargs = {"trust_remote_code": True, "use_fast": True}
        if any(x in self.model_name.lower() for x in ["qwen", "mistral", "jina"]):
            tokenizer_kwargs["fix_mistral_regex"] = True

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tokenizer_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ===================== ПРОВЕРКА: ЭТО АДАПТЕР? =====================
        model_path = Path(self.model_name)
        is_local = os.path.isdir(self.model_name)
        
        # Проверяем наличие adapter_config.json (локально или на HF)
        has_adapter = False
        if is_local:
            has_adapter = (model_path / "adapter_config.json").exists()
        else:
            # Для HF: пробуем скачать конфиг адаптера
            try:
                from huggingface_hub import hf_hub_download
                hf_hub_download(
                    repo_id=self.model_name,
                    filename="adapter_config.json",
                    local_dir=model_path if is_local else None,
                    force_download=False
                )
                has_adapter = True
                logger.info(f"🌐 Обнаружен PEFT-адаптер на HF: {self.model_name}")
            except Exception:
                has_adapter = False

        # ===================== ОПРЕДЕЛЕНИЕ ПУТИ К БАЗЕ =====================
        if has_adapter:
            # Это адаптер — нужна база
            if is_local:
                # Локальный адаптер: ищем базу локально
                base_candidates = [
                    model_path.parent.parent / "Nemotron-Rerank-1B-4bit",
                    model_path / "base_model",
                ]
                base_model_path = None
                for cand in base_candidates:
                    if cand.exists() and (cand / "config.json").exists():
                        base_model_path = str(cand)
                        logger.info(f"📁 Найдена локальная база: {base_model_path}")
                        break
                
                if base_model_path is None:
                    # 🔥 Фоллбэк: читаем base_model_name_or_path из локального конфига
                    import json
                    cfg_path = model_path / "adapter_config.json"
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    base_model_path = cfg.get("base_model_name_or_path")
                    if base_model_path:
                        logger.info(f"🌐 Локальная база не найдена, используем базу с HF: {base_model_path}")
                    else:
                        raise FileNotFoundError(
                            f"Не найдена база для адаптера {self.model_name}. "
                            f"Укажите base_model_name_or_path в adapter_config.json"
                        )
            else:
                # 🔥 HF-адаптер: читаем базу из конфига на HF
                from peft import PeftConfig
                peft_cfg = PeftConfig.from_pretrained(self.model_name)
                base_model_path = peft_cfg.base_model_name_or_path
                logger.info(f"🌐 База с HF: {base_model_path}")
        else:
            # Обычная модель (не адаптер)
            base_model_path = self.model_name

        # ===================== ЗАГРУЗКА: АДАПТЕР (база + LoRA) =====================
        if has_adapter:
            logger.info(f"🔄 Загружаем базу + LoRA адаптер: {self.model_name}")
            from peft import PeftModel
            
            # 1. Загружаем БАЗУ (с квантованием, если указано)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_path,
                quantization_config=quant_config,  # 🔥 Квантуем базу при загрузке с HF!
                device_map=self.device if self.device != "cpu" else None,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                num_labels=1,
                problem_type="regression",
                low_cpu_mem_usage=False,
            )
            
            # 2. Заменяем голову (как при обучении)
            hidden_size = base_model.config.hidden_size
            device = next(base_model.parameters()).device
            for attr in ["score", "classifier", "lm_head"]:
                if hasattr(base_model, attr):
                    delattr(base_model, attr)
            base_model.score = torch.nn.Linear(hidden_size, 1, dtype=torch.float32).to(device)
            base_model.config.num_labels = 1
            base_model.config.problem_type = "regression"
            
            # 3. Отключаем градиенты для инференса
            for name, param in base_model.named_parameters():
                if param.dtype == torch.float32:
                    param.requires_grad_(False)
            
            # 4. Навешиваем адаптер (локально или с HF)
            model = PeftModel.from_pretrained(base_model, self.model_name, is_trainable=False)
            model.eval()
            
            self.model = model
            
            # Токенизатор уже загружен выше из self.model_name (адаптера)
            logger.info(f"✅ Модель с адаптером загружена: {self.model_name}")
            return

        # ===================== ЗАГРУЗКА: ОБЫЧНАЯ / MERGED МОДЕЛЬ =====================
        # Токенизатор уже загружен выше
        
        if self.is_causal_lm:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quant_config,  # 🔥 Квантуем и causal LM
                device_map=self.device if self.device == "cuda" else None,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            logger.info(f"✅ Causal LM загружен: {self.model_name}")
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                quantization_config=quant_config,  # 🔥 Квантуем и SequenceClassification
                device_map=self.device if self.device == "cuda" else None,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                num_labels=1,
                problem_type="regression",
            )
            logger.info(f"✅ SequenceClassification загружен: {self.model_name}")

        # Перенос на устройство (если не использовался device_map)
        if self.device != "cpu" and not hasattr(self.model, "hf_device_map"):
            self.model.to(self.device)

        self.model.eval()

    def rerank(self, query: str, docs: List[Document], batch_size: int = 32) -> List[Tuple[Document, float]]:
        """Главный метод ранжирования."""
        if not docs:
            return []

        if self.is_jina_v3:
            return self._jina_rerank(query, docs)

        pairs = [[query, doc.page_content] for doc in docs]
        try:
            if self.is_cross_encoder:
                scores = self._cross_encoder_rerank(pairs, batch_size)
            elif self.is_causal_lm:
                scores = self._causal_lm_rerank(pairs, batch_size)
            else:
                scores = self._transformers_rerank(pairs, batch_size)

            scored = sorted([(d, float(s)) for d, s in zip(docs, scores)], key=lambda x: x[1], reverse=True)
            return scored
        except Exception as e:
            logger.warning(f"Rerank failed: {e}")
            return [(d, 0.0) for d in docs]

    def _jina_rerank(self, query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
        """Специальный метод для Jina v3 (использует встроенный .rerank)."""
        try:
            documents = [doc.page_content for doc in docs]
            results = self.model.rerank(query, documents)
            scored = [(docs[res['index']], float(res['score'])) for res in results]
            torch.cuda.empty_cache()
            gc.collect()
            return sorted(scored, key=lambda x: x[1], reverse=True)
        except Exception as e:
            logger.warning(f"Jina rerank failed: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            return [(d, 0.0) for d in docs]

    # (твои методы _cross_encoder_rerank, _causal_lm_rerank, _transformers_rerank оставь без изменений)
       
    # (все три приватных метода остаются без изменений)
    def _cross_encoder_rerank(self, pairs: List[List[str]], batch_size: int) -> List[float]:
        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
            else:
                scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
        return scores.tolist() if isinstance(scores, np.ndarray) else scores
    
    def _causal_lm_rerank(self, pairs: List[List[str]], batch_size: int) -> List[float]:
            """
            ОФИЦИАЛЬНЫЙ скоринг Qwen3-Reranker (из model card).
            """
            scores = []

            system_prompt = (
                "Judge whether the Document meets the requirements based on the Query "
                "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
            )

            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                batch_texts = []

                for query, doc in batch_pairs:
                    instruction = "Given a web search query, retrieve relevant passages that answer the query."
                    user_content = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ]

                    text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    batch_texts.append(text)

                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    last_logits = outputs.logits[:, -1, :]

                    yes_id = self.tokenizer.encode("yes", add_special_tokens=False)[-1]
                    no_id  = self.tokenizer.encode("no", add_special_tokens=False)[-1]

                    yes_logits = last_logits[:, yes_id]
                    no_logits  = last_logits[:, no_id]

                    probs = torch.softmax(torch.stack([no_logits, yes_logits], dim=1), dim=1)
                    batch_scores = probs[:, 1].cpu().tolist()   # вероятность "yes"

                scores.extend(batch_scores)

            return scores

    
    def _transformers_rerank(self, pairs: List[List[str]], batch_size: int) -> List[float]:
        """
        Скоринг для моделей SequenceClassification (BGE, GTE, etc.)
        """
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i+batch_size]
            queries = [p[0] for p in batch_pairs]
            documents = [p[1] for p in batch_pairs]
            
            inputs = self.tokenizer(
                queries,
                documents,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                return_token_type_ids=False
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)
                
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits.squeeze(-1).cpu().float()
                    batch_scores = logits.flatten().tolist() if logits.dim() > 0 else logits.tolist()
                else:
                    batch_scores = [0.0] * len(batch_pairs)
            
            if isinstance(batch_scores, list) and batch_scores and isinstance(batch_scores[0], list):
                batch_scores = [float(item[0]) if isinstance(item, list) else float(item) for item in batch_scores]
            
            scores.extend(batch_scores)
        
        return scores

# ==================== НАСТРОЙКИ (SOTA расширенные) ====================
@dataclass(frozen=True)
class Settings:
    # --- Базовые пути и модели ---
    chroma_dir: str = os.environ.get("CHROMA_PATH", "./chroma_db_base")
    collection: str = os.environ.get("CHROMA_COLLECTION", "smeta_collection")
    chunks_jsonl: str = os.environ.get("CHUNKS_JSONL", "./data/chunks/all_chunks.jsonl")
    embed_model: str = os.environ.get("EMBED_MODEL", "deepvk/USER2-base")
    embed_max_length: int = int(os.environ.get("EMBED_MAX_LENGTH", "512"))
    embed_batch_size: int = int(os.environ.get("EMBED_BATCH_SIZE", "16"))
    truncate_dim: Optional[int] = int(os.environ.get("TRUNCATE_DIM", "0")) or None
    
    # --- Реранкер (теперь динамический) ---
    rerank_model: str = os.environ.get("RERANK_MODEL", "Alibaba-NLP/gte-reranker-modernbert-base")
    rerank_max_length: int = int(os.environ.get("RERANK_MAX_LENGTH", "512"))
    rerank_batch_size: int = int(os.environ.get("RERANK_BATCH_SIZE", "16"))
    reranker_type: str = os.environ.get("RERANKER_TYPE", "auto")  # "auto", "cross_encoder", "transformers"
    
    # --- LLM ---
    llm_model: str = os.environ.get("LLM_MODEL", "saiga_yandexgpt_8b_gguf")
    llm_base_url: str = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
    llm_api_key: str = os.environ.get("LLM_API_KEY", "any")
    
    # --- Параметры поиска (Бейзлайн значения) ---
    bm25_k: int = 300
    vec_k: int = 200
    rrf_k: int = 150
    pool_k: int = 0  # 0 = вычисляется динамически как bm25_k + vec_k
    rerank_k: int = 60
    rerank_threshold: float = 0.05
    final_k: int = 5
    
    # --- НОВЫЕ ПАРАМЕТРЫ SOTA (со значениями по умолчанию = старый режим) ---
    fusion_strategy: str = "rrf"  # "rrf" или "weighted"
    hybrid_alpha: float = 0.5  # Вес векторного поиска (0.0 - только BM25, 1.0 - только Vector)
    retrieval_mode: str = "fixed"  # "fixed" или "adaptive"
    adaptive_threshold: float = 0.05  # Порог отсечения для адаптивного режима
    
    # --- Прочее ---
    max_context_chars: int = 12000
    per_doc_char_cap: int = 5000
    max_tokens: int = 1024
    temperature: float = 0.0
    hhem_model_name: str = "vectara/hallucination_evaluation_model"
    hhem_max_length: int = 512
    
    def get_pool_k(self) -> int:
        """Вычисляет pool_k динамически, если не задан явно."""
        if self.pool_k > 0:
            return self.pool_k
        return self.bm25_k + self.vec_k

# ==================== КЛАССЫ ====================
class USER2Embedder(Embeddings):
    def __init__(self, model_name: str, device: str, max_length: int, batch_size: int,
                 dtype: torch.dtype = None, truncate_dim: Optional[int] = None):
        dtype = dtype or (torch.float16 if device == "cuda" else torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name, dtype=dtype).eval().to(device)
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.truncate_dim = truncate_dim
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    @torch.no_grad()
    def _encode(self, texts: List[str]) -> List[List[float]]:
        out = []
        for i in range(0, len(texts), self.batch_size):
            bt = texts[i:i + self.batch_size]
            batch = self.tokenizer(bt, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            emb = self.mean_pooling(outputs, batch["attention_mask"])
            if self.truncate_dim:
                emb = emb[:, :self.truncate_dim]
            emb = F.normalize(emb, p=2, dim=1)
            out.extend(emb.float().cpu().tolist())
        return out
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = [DOC_PREFIX + t for t in texts]
        return self._encode(prefixed)
    
    def embed_query(self, text: str) -> List[float]:
        prefixed = QUERY_PREFIX + text
        return self._encode([prefixed])[0]

@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    source: str
    metadata: Dict[str, Any]

def iter_chunks(jsonl_path: str) -> Iterable[ChunkRecord]:
    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Bad JSON at line {ln} (skipped)")
                continue
            chunk_id = st(obj.get("chunk_id"))
            text = st(obj.get("text"))
            source = st(obj.get("source"))
            if not chunk_id or not text:
                continue
            md = normalize_chunk_metadata(obj, chunk_id)
            yield ChunkRecord(chunk_id=chunk_id, text=text, source=source, metadata=md)

def build_bm25_documents(jsonl_path: str) -> List[Document]:
    return [Document(page_content=ch.text, metadata=ch.metadata) for ch in iter_chunks(jsonl_path)]

def _chroma_query(vs: Chroma, query_emb: List[float], k: int) -> List[Document]:
    raw = vs._collection.query(query_embeddings=[query_emb], n_results=k, include=["documents", "metadatas", "distances"])
    docs0, mds0 = (raw.get("documents") or [[]])[0], (raw.get("metadatas") or [[]])[0]
    out = []
    for doc, md in zip(docs0, mds0):
        md = md or {}
        md["title"] = strip_part_suffix(st(md.get("title")))
        out.append(Document(page_content=st(doc), metadata=md))
    return [d for d in out if st(d.page_content)]

def _bm25_invoke(bm25: BM25Retriever, q: str) -> List[Document]:
    return bm25.invoke(q) or []

def rrf_fusion(ranked_lists: List[List[Document]], rrf_k: int) -> Dict[str, float]:
    scores = {}
    for lst in ranked_lists:
        for r, d in enumerate(lst, 1):
            if s := doc_sig(d):
                scores[s] = scores.get(s, 0.0) + 1.0 / (rrf_k + r)
    return scores

def weighted_fusion(bm25_docs: List[Document], vec_docs: List[Document], alpha: float) -> Dict[str, float]:
    """
    Weighted Sum Fusion: Score = alpha * VecScore + (1-alpha) * BM25Score.
    Scores нормализуются рангом (1/rank) для сопоставимости.
    """
    scores = {}
    all_docs = dedup_docs(bm25_docs + vec_docs)
    # Создаем мапы рангов
    bm25_ranks = {doc_sig(d): i+1 for i, d in enumerate(bm25_docs)}
    vec_ranks = {doc_sig(d): i+1 for i, d in enumerate(vec_docs)}
    
    for d in all_docs:
        sig = doc_sig(d)
        # Нормализованный скор через ранг (чем меньше ранг, тем больше скор)
        b_score = 1.0 / bm25_ranks.get(sig, 1000)
        v_score = 1.0 / vec_ranks.get(sig, 1000)
        scores[sig] = (alpha * v_score) + ((1.0 - alpha) * b_score)
    
    return scores

class Retriever:
    def __init__(self, s: Settings, vs: Chroma, bm25: BM25Retriever, embedding_fn: Embeddings, reranker: UnifiedReranker):
        self.s = s
        self.vs = vs
        self.bm25 = bm25
        self.embedding_fn = embedding_fn
        self.reranker = reranker
    
    def _vector(self, query: str, k: int) -> List[Document]:
        emb = self.embedding_fn.embed_query(query)
        return _chroma_query(self.vs, emb, k)
    
    def _hybrid_retrieve(self, question: str) -> Tuple[List[Document], Dict[str, Any]]:
        meta = {"mode": "hybrid"}
        t0 = time.time()
        
        # 1. Получаем кандидаты
        bm_ranked = dedup_docs(_bm25_invoke(self.bm25, question)[:self.s.bm25_k])
        vec_ranked = dedup_docs(self._vector(question, self.s.vec_k))
        
        # 2. Fusion (RRF или Weighted)
        if self.s.fusion_strategy == "weighted":
            scores_fusion = weighted_fusion(bm_ranked, vec_ranked, self.s.hybrid_alpha)
            meta["fusion"] = "weighted"
            meta["alpha"] = self.s.hybrid_alpha
        else:
            scores_fusion = rrf_fusion([bm_ranked, vec_ranked], self.s.rrf_k)
            meta["fusion"] = "rrf"
        
        cand = dedup_docs(bm_ranked + vec_ranked)
        scored_fusion = sorted([(d, scores_fusion.get(doc_sig(d), 0.0)) for d in cand], key=lambda x: x[1], reverse=True)
        
        # 3. Pool для реранкера (ИСПРАВЛЕНИЕ БЛОКЕРА №2: динамический pool_k)
        pool_k = self.s.get_pool_k()
        pool = [d for d, _ in scored_fusion][:pool_k]
        t1 = time.time()  # Время до реранкера
        
        # 4. Реранкинг
        scored = self.reranker.rerank(question, pool[:self.s.rerank_k], batch_size=self.s.rerank_batch_size)
        
        # 5. Финальный отбор (Fixed или Adaptive)
        final = []
        if self.s.retrieval_mode == "adaptive":
            # Адаптивная остановка по порогу
            for d, s in scored:
                if s >= self.s.adaptive_threshold:
                    final.append(d)
                else:
                    break  # Останавливаемся, как только score упал ниже порога
            # Ограничиваем сверху final_k
            final = final[:self.s.final_k]
            meta["retrieval_mode"] = "adaptive"
            meta["adaptive_threshold"] = self.s.adaptive_threshold
        else:
            # Старый режим: фильтр по порогу, потом top-K
            scored_filtered = [(d, s) for d, s in scored if s > self.s.rerank_threshold]
            final = [d for d, _ in scored_filtered[:self.s.final_k]]
            meta["retrieval_mode"] = "fixed"
        
        fs = [float(s) for _, s in scored[:len(final)]]
        meta.update({
            "bm25": len(bm_ranked),
            "vec": len(vec_ranked),
            "pool_k": pool_k,
            "pool": len(pool),
            "rerank_input": len(scored),
            "final": len(final),
            "top_scores": [round(x, 4) for x in fs[:min(8, len(fs))]],
            "retrieval_time": t1 - t0,
        })
        return final, meta
    
    def retrieve(self, question: str) -> Tuple[List[Document], Dict[str, Any]]:
        code_match = CODE_RE.search(question)
        if code_match:
            target_code = code_match.group(0)
            logger.debug(f"Обнаружен код расценки: {target_code}")
            try:
                raw = self.vs._collection.get(where={"work_code": target_code}, include=["documents", "metadatas"])
                if raw["ids"]:
                    docs = [Document(page_content=doc, metadata=md or {}) for doc, md in zip(raw["documents"], raw["metadatas"])]
                    meta = {"mode": "exact_code_match", "code": target_code, "exact_hits": len(docs)}
                    logger.info(f"Точный матч по коду {target_code} (документов: {len(docs)})")
                    if len(docs) < self.s.final_k:
                        hybrid_docs, _ = self._hybrid_retrieve(question)
                        docs = dedup_docs(docs + hybrid_docs[:2])
                    return docs[:self.s.final_k], meta
                else:
                    logger.warning(f"Код {target_code} не найден. Переход на гибридный поиск.")
            except Exception as e:
                logger.warning(f"Ошибка точного поиска по коду {target_code}: {e}")
        return self._hybrid_retrieve(question)

# ==================== PROMPTS ====================
SYSTEM_PROMPT = (
    "Ты — специалист по ценообразованию и сметному нормированию в строительстве и проектировании на территории Российской федерации.\n"
    "Твоя задача — ответить на вопрос пользователя, используя только предоставленный контекст из нормативных документов (ГЭСН, ФСНБ, official_normative и т.п.).\n"
    "Отвечай строго по предоставленному контексту (без выдумывания фактов). "
    "Если вопрос про расценку — постарайся найти код или максимально похожий по наименованию норматив, если используешь не точный код — уточни, что это не точное совпадение.\n"
    "Иерархия источников:\n"
    "- official_normative, fsnb, ГЭСН — первоисточник.\n"
    "- QA — разъяснение, НЕ заменяет норматив.\n\n"
    "Правила:\n"
    "- Нельзя добавлять факты вне контекста.\n"
    "- Если данных недостаточно — скажи 'нет данных в контексте' и уточни, чего не хватает.\n"
)

STYLE_PROMPT = (
    "Сформируй ответ СТРОГО в формате:\n"
    "Ответ: <текст>\n\n"
    "КРИТИЧЕСКИЕ ОГРАНИЧЕНИЯ:\n"
    "- Максимум 4 предложения.\n"
    "- Без воды, вступлений и подробных разъяснений.\n"
    "- Если не знаешь точно: 'Ответ: нет данных в контексте'.\n"
)

REFLECTION_PROMPT = (
    "Ты — рецензент ответа по сметному нормированию.\n"
    "Проверь draft на точность, полноту и соответствие контексту.\n"
    "Верни улучшенную версию в формате 'Ответ: <текст>', без объяснений.\n"
    "Если draft хорош — верни его без изменений.\n"
)

class Answerer:
    def __init__(self, s: Settings, llm: ChatOpenAI):
        self.s = s
        self.llm = llm
    
    def synthesize(self, question: str, context: str, sources: List[Dict[str, str]]) -> str:
        if not context.strip():
            return "Ответ: нет данных в контексте: база не нашла подходящие фрагменты. Уточните запрос."
        
        user = (
            f"Контекст:\n{context}\n\n"
            f"Вопрос пользователя: {st(question)}\n\n"
            f"{STYLE_PROMPT}\n"
            "Ответ: "
        )
        try:
            resp = self.llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user)])
            draft = SOURCES_BLOCK_RE.sub("", st(resp.content))
            draft = QUESTION_LINE_RE.sub("", re.sub(r"\n{3,}", "\n\n", draft)).strip()
            if draft and not draft.lower().startswith("ответ"):
                draft = "Ответ: " + draft
            
            reflection_user = (
                f"Контекст:\n{context}\n\n"
                f"Вопрос: {st(question)}\n\n"
                f"Draft-ответ: {draft}\n\n"
                f"{REFLECTION_PROMPT}"
            )
            reflection_resp = self.llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=reflection_user)])
            revised = st(reflection_resp.content).strip()
            if revised and not revised.lower().startswith("ответ"):
                revised = "Ответ: " + revised
            draft = revised
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return "Ответ: ошибка генерации ответа."
        
        lines = []
        for s0 in sources:
            wc = st(s0.get("work_code"))
            title = st(s0.get("title"))
            doc = st(s0.get("source_file"))
            anchor = st(s0.get("anchor"))
            label = f"[{wc}] " if wc else ""
            main = (label + title).strip() or "Документ"
            tail = f" — {doc}{anchor}".strip() if doc else anchor
            line = f"- {main}{tail}".strip()
            if line != "-":
                lines.append(line)
        lines = uniq(lines)[:20]
        if lines:
            draft = draft.rstrip() + "\n\nИсточники:\n" + "\n".join(lines)
        return draft

class HHEMEvaluator:
    def __init__(self, model_name: str, device: str, max_length: int = 512):
        self.device = device
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.is_available = False
        try:
            logger.info(f"Загрузка Vectara HHEM модели: {model_name}")
            from transformers import T5Tokenizer, AutoModelForSequenceClassification
            
            self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base", legacy=False, use_fast=False)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, trust_remote_code=True,
                dtype=torch.float16 if device == "cuda" else torch.float32,
                ignore_mismatched_sizes=True
            ).to(device).eval()
            
            # Тест
            test_input = "The sky is blue [CONTEXT] The sky is blue during daytime"
            inputs = self.tokenizer(test_input, return_tensors="pt", truncation=True, max_length=self.max_length, padding="max_length").to(device)
            with torch.no_grad():
                outputs = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                hallucination_prob = torch.sigmoid(outputs.logits[0, 1]).item()
                logger.debug(f"HHEM тест: prob={hallucination_prob:.3f}")
            
            self.is_available = True
            logger.info("✓ Vectara HHEM успешно загружен")
        except Exception as e:
            logger.error(f"✗ Ошибка загрузки Vectara HHEM: {e}")
            logger.warning("→ Faithfulness будет оцениваться через cosine similarity (fallback)")
    
    @torch.no_grad()
    def evaluate(self, answer: str, contexts: List[str]) -> float:
        if not self.is_available or not answer.strip() or not contexts:
            return 0.0
        
        context_str = " ".join(contexts)[:4000]
        sentences = split_into_sentences(answer)
        if not sentences:
            return 0.0
        
        hallucination_scores = []
        for sent in sentences:
            input_text = f"{sent} [CONTEXT] {context_str}"
            try:
                inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=self.max_length, padding="max_length").to(self.device)
                outputs = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                hallucination_prob = torch.sigmoid(outputs.logits[0, 1]).item()
                hallucination_scores.append(hallucination_prob)
            except Exception as e:
                logger.warning(f"HHEM ошибка на предложении: {e}")
                hallucination_scores.append(1.0)
        
        avg_hallucination = np.mean(hallucination_scores)
        return float(np.clip(1.0 - avg_hallucination, 0.0, 1.0))

class SmetaRAGApp:
    def __init__(self, s: Settings):
        self.s = s
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Устройство: {self.device}")
        
        emb_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.embedding_fn = USER2Embedder(
            self.s.embed_model, self.device, self.s.embed_max_length,
            self.s.embed_batch_size, emb_dtype, truncate_dim=self.s.truncate_dim
        )
        
        self.vectorstore = Chroma(
            persist_directory=self.s.chroma_dir,
            collection_name=self.s.collection,
            embedding_function=self.embedding_fn
        )
        
        # Проверка совместимости
        try:
            stored_meta = self.vectorstore._collection.metadata
            stored_model = stored_meta.get("embedding_model", "")
            if stored_model and stored_model != self.s.embed_model:
                logger.warning(f"⚠️ Несовпадение модели эмбеддингов! В коллекции: {stored_model}, Текущая: {self.s.embed_model}")
        except Exception:
            pass
        
        self.llm = ChatOpenAI(
            model=self.s.llm_model, base_url=self.s.llm_base_url, api_key=self.s.llm_api_key,
            temperature=self.s.temperature, max_tokens=self.s.max_tokens
        )
        
        logger.info("Загрузка чанков для BM25...")
        bm25_docs = build_bm25_documents(self.s.chunks_jsonl)
        logger.info(f"BM25 документов: {len(bm25_docs)}")
        
        bm25 = BM25Retriever.from_documents(bm25_docs)
        bm25.k = self.s.bm25_k
        
        # ИСПРАВЛЕНИЕ БЛОКЕРА №1: UnifiedReranker вместо CrossEncoder
        logger.info(f"Загрузка реранкера: {self.s.rerank_model} (тип: {self.s.reranker_type})")
        reranker = get_cached_reranker(
            self.s.rerank_model,
            self.device,
            self.s.rerank_max_length,
            reranker_type=self.s.reranker_type
        )
        
        # ИСПРАВЛЕНИЕ БЛОКЕРА №2: pool_k вычисляется динамически через s.get_pool_k()
        # Больше нет хака с s_eff - Settings теперь имеет метод get_pool_k()
        self.retriever = Retriever(s, self.vectorstore, bm25, self.embedding_fn, reranker)
        self.answerer = Answerer(s, self.llm)
        
        self.hhem_evaluator = HHEMEvaluator(
            model_name=s.hhem_model_name,
            device=self.device,
            max_length=s.hhem_max_length
        )
    
    def ask(self, question: str, debug: bool = False) -> Dict[str, Any]:
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
        
        docs, meta = self.retriever.retrieve(question)
        sources = collect_sources(docs)
        context = docs_to_context(docs, self.s.max_context_chars, self.s.per_doc_char_cap)
        answer = self.answerer.synthesize(question, context, sources)
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "docs_count": len(docs),
            "meta": meta,
            "contexts": [doc.page_content for doc in docs],
            "retrieval_time": meta.get("retrieval_time", 0.0),
        }

# ==================== ЗАПУСК  ====================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SmetaRAG Gen — RAG с поддержкой SOTA оптимизации")
    p.add_argument("--ask", type=str, default="", help="Один вопрос для теста")
    p.add_argument("--debug", action="store_true", help="Включить debug-логи")
    p.add_argument("--print-meta", action="store_true", help="Вывести мета-информацию поиска")
    # Аргументы для ручного тестирования новых фич
    p.add_argument("--fusion", type=str, default=None, choices=["rrf", "weighted"], help="Стратегия слияния (переопределяет оптимизированные)")
    p.add_argument("--alpha", type=float, default=None, help="Alpha для weighted fusion")
    p.add_argument("--mode", type=str, default=None, choices=["fixed", "adaptive"], help="Режим поиска")
    p.add_argument("--thresh", type=float, default=None, help="Порог для adaptive mode")
    p.add_argument("--reranker", type=str, default=None, help="Модель реранкера")
    p.add_argument("--reranker-type", type=str, default=None, choices=["auto", "cross_encoder", "transformers"], 
                   help="Тип реранкера (auto определяет автоматически)")
    p.add_argument("--pool-k", type=int, default=None, help="Размер пула для реранкера (0 = bm25_k + vec_k)")
    # Дополнительные параметры поиска
    p.add_argument("--bm25-k", type=int, default=None, help="Количество документов из BM25")
    p.add_argument("--vec-k", type=int, default=None, help="Количество документов из векторного поиска")
    p.add_argument("--rrf-k", type=int, default=None, help="Параметр сглаживания RRF")
    p.add_argument("--rerank-k", type=int, default=None, help="Количество документов на реранкинг")
    p.add_argument("--rerank-threshold", type=float, default=None, help="Порог отсечения реранкера")
    p.add_argument("--final-k", type=int, default=None, help="Итоговое количество документов")
    # Путь к файлу с оптимизированными параметрами
    p.add_argument("--optimized-params", type=str, default="./output/ga_optimization/optimized_params.py",
                   help="Путь к файлу с оптимизированными параметрами (по умолчанию загружается)")
    return p.parse_args()

def load_optimized_params(path: str) -> Dict[str, Any]:
    """Загружает OPTIMIZED_PARAMS из файла, игнорируя синтаксические ошибки в других строках."""
    params = {}
    if not os.path.exists(path):
        logger.info(f"Файл оптимизированных параметров не найден: {path}. Используются значения по умолчанию.")
        return params

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Ищем строку, где начинается OPTIMIZED_PARAMS = {
        import re
        match = re.search(r'OPTIMIZED_PARAMS\s*=\s*(\{.*?\})', content, re.DOTALL)
        if match:
            dict_str = match.group(1)
            # Безопасно вычисляем словарь
            params = eval(dict_str)
            logger.info(f"✅ Загружены оптимизированные параметры из {path}")
        else:
            logger.warning("В файле не найдена переменная OPTIMIZED_PARAMS")
    except Exception as e:
        logger.error(f"Ошибка загрузки оптимизированных параметров: {e}")

    return params

def print_parameters(params: Dict[str, Any]):
    """Выводит используемые параметры в читаемом виде."""
    print("\n" + "=" * 60)
    print("🔧 ИСПОЛЬЗУЕМЫЕ ПАРАМЕТРЫ")
    print("=" * 60)
    print(f"  bm25_k:              {params.get('bm25_k')}")
    print(f"  vec_k:               {params.get('vec_k')}")
    print(f"  rrf_k:               {params.get('rrf_k')}")
    print(f"  rerank_k:            {params.get('rerank_k')}")
    print(f"  rerank_threshold:    {params.get('rerank_threshold')}")
    print(f"  final_k:             {params.get('final_k')}")
    print(f"  fusion_strategy:     {params.get('fusion_strategy')}")
    print(f"  hybrid_alpha:        {params.get('hybrid_alpha'):.3f}")
    print(f"  retrieval_mode:      {params.get('retrieval_mode')}")
    print(f"  adaptive_threshold:  {params.get('adaptive_threshold'):.3f}")
    print(f"  rerank_model:        {params.get('rerank_model')}")
    print(f"  reranker_type:       {params.get('reranker_type')}")
    print(f"  pool_k:              {params.get('pool_k')}")
    print("=" * 60 + "\n")


def evaluate_with_triad(app: SmetaRAGApp, test_questions: List[str]):
    logger.info("Запуск полной RAG Triad оценки ")
    reference_answers = {
        "Что такое 01-01-009-11?": "Разработка грунта в траншеях экскаватором \"обратная лопата\" с ковшом вместимостью 0,65 м³, группа грунтов: 5",
        "Сколько глав в сводном сметном расчете? Что находится в 3 главе?": "12 глав, Глава 3 включает объекты подсобного и обслуживающего назначения, такие как склады, столовые, медпункты, административные здания и другие вспомогательные сооружения, необходимые для обеспечения нормальной работы основных объектов строительства.",
        "Какие коэффициенты применяются при работе в зимних условиях?": "НДЗ являются среднегодовыми и применяются на весь период строительства.",
        "Какую расценку применить для подвесных потолков Armstrong в офисном здании?": "Наиболее подходящей является расценка ФЕР15-01-047-15 - установка подвесных потолков из минераловатных плит типа \"Армстронг\".",
        "Разъясните возможность применения коэффициента 1,5 «Производство ремонтно-строительных работ осуществляется в жилых помещениях без расселения» ...": "Коэффициент 1,5 применяется при капитальном ремонте многоквартирных домов (МКД) без расселения к нормам затрат труда рабочих строителей и затратам на эксплуатацию машин только для работ, выполняемых непосредственно в жилых помещениях (например, замена полов). При производстве работ по замене инженерных сетей, проходящих через квартиры МКД (стояки ХВС, ГВС и иных), данный коэффициент не применим. Однако при выполнении таких работ следует учитывать другие коэффициенты условий труда согласно МДС 81-35.2004, если имеет место пересечение людских потоков в местах общего пользования здания.",
        "Как расценивается проектирование встраиваемых помещений?": "Условия проектирования объединенных или сблокированных зданий и сооружений, а также зданий со встроенными помещениями другого назначения при определении стоимости проектных работ по ценам НЗ на проектные работы учитываются применением в расчете соответствующих корректирующих коэффициентов на сокращенный объем работ с учетом следующих положений, если иное не установлено в НЗ на проектные работы: а) стоимость проектных работ для основного здания (сооружения) определяется с применением цен на проектные работы исходя из натурального показателя, установленного для этого объекта, с ценообразующим коэффициентом 1; б) стоимость проектных работ для встраиваемых помещений (сооружений) в основное здание (сооружение) определяется с применением цен НЗ на проектные работы исходя из натурального показателя, установленного для встраиваемого объекта, с ценообразующим коэффициентом в размере до 0,5; в) стоимость подготовки проектной и рабочей документации зданий (сооружений) сблокированных с основным определяется по ценам НЗ на проектные работы исходя из натурального показателя, установленного для сблокированного объекта, с корректирующим коэффициентом в размере до 0,8. (в ред. Приказа Минстроя РФ от 08.06.2023 N 409/пр)",
        "Как исчисляется объем работ на прокладку кабельной продукции?": "Объем работ по прокладке электрического кабеля следует определять по всей проектной длине трассы или линии (кабельной линии), исчисляемой как длина в плане (промеренная по чертежам методом суммирования) с учетом изгибов и поворотов.",
        "Согласно пункту 1.20.22 раздела I «Общие положения» сборника 20 «Вентиляция и кондиционирование воздуха» ГЭСН 81-02-20-2022 в нормах таблиц 20-06-018, 20-06-019 учтены затраты на прокладку каждого типа коммуникационных трасс (медные трубки, дренаж, питающий кабель) до 10 м. Какие медные трубки и какой длины имеются ввиду?": "В нормах таблиц 20-06-018, 20-06-019 сборника 20 «Вентиляция и кондиционирование воздуха» ГЭСН 81-02-20-2022 учтены затраты на прокладку каждого типа коммуникационных трасс: медные трубки для транспортировки хладагента: две трубки (до 10 м каждая); дренаж (до 10 м); питающий кабель (до 10 м).",
        "Что такое сметная стоимость строительства, реконструкции, капитального ремонта, сноса объектов капитального строительства, работ по сохранению объектов культурного наследия согласно Методике 421/пр?": "Согласно пункту 30 статьи 1 Градостроительного кодекса Российской Федерации сметная стоимость строительства, реконструкции, капитального ремонта, сноса объектов капитального строительства, работ по сохранению объектов культурного наследия (далее — сметная стоимость строительства) — расчетная стоимость строительства, реконструкции, капитального ремонта, сноса объектов капитального строительства, работ по сохранению объектов культурного наследия, подлежащая определению на этапе архитектурно-строительного проектирования, подготовки сметы на снос объекта капитального строительства и применению в соответствии со статьей 8.3 Градостроительного кодекса Российской Федерации.",
        "В каких случаях при определении сметной стоимости применяется конъюнктурный анализ?": "При отсутствии в ФГИС ЦС данных о сметных ценах в базисном или текущем уровне цен на отдельные материальные ресурсы и оборудование, а также сметных нормативов на отдельные виды работ и услуг допускается определение их сметной стоимости по наиболее экономичному варианту, определенному на основании сбора информации о текущих ценах (далее — конъюнктурный анализ).",
        "Какой коэффициент применяется к затратам труда рабочих при производстве работ в стесненных условиях в населенных пунктах?": "Коэффициент, учитывающий стесненные условия в населенных пунктах, определяется наличием трех из перечисленных факторов: интенсивное движение городского транспорта и пешеходов в непосредственной близости (в пределах 50 метров) от зоны производства работ; сети подземных коммуникаций, подлежащие перекладке или подвеске; расположение объектов капитального строительства и сохраняемых зеленых насаждений в непосредственной близости (в пределах 50 метров) от зоны производства работ; стесненные условия или невозможность складирования материалов; ограничение поворота стрелы грузоподъемного крана.",
        "Что устанавливает Методика определения стоимости работ по подготовке проектной документации?": "Методика определения стоимости работ по подготовке проектной документации (далее - Методика) устанавливает порядок определения сметной стоимости работ по подготовке проектной и (или) рабочей документации для строительства, реконструкции, капитального ремонта, сноса объектов капитального строительства, работ по сохранению объектов культурного наследия на основании сметных нормативов на работы по подготовке проектной и (или) рабочей документации, нормативных затрат на работы по подготовке проектной документации (далее - НЗ на проектные работы), а также порядок разработки НЗ на проектные работы.",
        "По какой формуле рассчитывается цена проектных работ при использовании параметров цены в зависимости от натуральных показателей?": "Цена проектных работ (Ц) рассчитывается по формуле: Ц = a + b × X, где: Ц — цена проектных работ, тыс. руб.; a — параметр цены проектных работ: постоянная величина, выраженная в тыс. руб.; b — параметр цены проектных работ: постоянная величина, имеющая размерность тыс. руб. на единицу натурального показателя; X — величина натурального показателя проектируемого объекта.",
        "Как определяется стоимость работ по подготовке проектной документации, содержащей материалы в форме информационной модели?": "Стоимость работ по подготовке проектной документации, содержащей материалы в форме информационной модели, рассчитывается по формуле: СИМП = СП × (ΔИМП × КИМ + ΔТГП) × КПД, где СП - цена разработки проектной и рабочей документации, определяемая по справочникам базовых цен; ΔИМП - сумма долей относительных стоимостей разработки разделов ПЗУ, ППО, АР, КР, ТКР, ИЛО, ПОС, СМ, подразделов ЭО, ВС, ВО, ОВ, СС, ГС, ТХ раздела ИОС, включаемых в трехмерную модель; КИМ - корректирующий коэффициент в зависимости от вида объекта (1,09-1,31); ΔТГП - сумма долей разработки разделов ПЗ, ООС, МПОБ, МОДИ, ЭЭФ, размещаемых в среде общих данных; КПД - доля стоимости работ по подготовке проектной документации (40-60%).",
        "Какие корректирующие коэффициенты применяются для объектов жилищно-гражданского строительства при определении стоимости работ с применением информационного моделирования?": "Для объектов жилищно-гражданского строительства применяются следующие корректирующие коэффициенты: кирпичный многоквартирный дом - 1,17; крупнопанельный, монолитный многоквартирный дом - 1,18; офисное здание - 1,21; здание банка - 1,22; здание концертного зала - 1,15; здание театра, цирка - 1,16; гипермаркет, универсам - 1,18; здание гостиницы - 1,17; бассейн крытый - 1,13; здание больницы общего профиля - 1,17; здание поликлиники - 1,19; здание общежития - 1,14; здание стоянки закрытого типа отапливаемой - 1,11.",
        "Как определяется численность работников технического заказчика?": "Численность работников технического заказчика определяется согласно показателям, приведенным в Приложении № 2 к Методике, исходя из величины сметной стоимости строительства объекта по итогам глав 1-9 и 12 ССРСС, определенной в уровне цен по состоянию на 1 января 2023 г. с округлением до целого числа, млн. рублей. При сметной стоимости строительства до 600 млн. рублей численность составляет 1 человек; от 600 до 1200 млн. рублей - 3 человека; от 1200 до 1500 - 4; от 1500 до 1800 - 5; от 1800 до 2350 - 6; от 2350 до 3500 - 8; от 3500 до 4700 - 9; от 4700 до 5900 - 10; от 5900 до 7000 - 11; от 7000 до 8800 - 13; от 8800 до 10600 - 15 человек. При сметной стоимости свыше 10600 млн. рублей численность увеличивается на одного человека на каждые 1100 млн. рублей сверх указанной суммы.",
        "Какие затраты включаются в расчет на осуществление функций технического заказчика?": "Затраты на осуществление функций технического заказчика включают: затраты на оплату труда (определяются исходя из численности работников и размера средней заработной платы, не превышающего произведение среднемесячного размера оплаты труда рабочего четвертого разряда на коэффициент 1,4); страховые взносы; налог на имущество, транспортный, земельный налоги; амортизацию основных средств; командировочные расходы; арендную плату и содержание служебного автотранспорта; арендную плату и содержание зданий и помещений (площадь определяется произведением численности работников, величины площади рабочего места (согласно СанПиН) и коэффициента 1,2); повышение квалификации кадров; приобретение оргтехники, мебели, канцелярских товаров, программного обеспечения, спецодежды; охрану зданий и помещений; услуги связи и интернет; прочие расходы (не более 5% от суммы указанных расходов); премирование за досрочный ввод объекта; надбавку за секретность (при наличии); расходы на организацию геодезических работ; сметную прибыль в размере 10% от суммы всех затрат.",
        "Что означает код 65-01-001-01?": "Код 65-01-001-01 соответствует нормативу на разборку трубопроводов из водогазопроводных труб диаметром до 25 мм. Единица измерения — 100 м трубопровода. В состав работ входит: снятие труб и креплений с отборкой годных труб, арматуры, фасонных и крепежных частей; свертывание арматуры; правка и очистка труб от накипи; складирование труб и фасонных частей. Норматив относится к сборнику ГЭСНр (ремонтно-строительные работы), таблица 65-01-001.",
        "Какой код соответствует пусконаладочным работам по регулировке сети вентиляции с количеством сечений до 5?": "Для пусконаладочных работ по регулировке сети систем вентиляции и кондиционирования воздуха при количестве сечений до 5 применяется код 03-01-022-01. Единица измерения — сеть. Работы включают: подготовительные работы, снятие с натуры схем вентиляционных систем, аэродинамические испытания и сопоставление с проектом объемов воздуха, регулировку сети для достижения проектных показателей, комплексное опробование и обеспечение воздушного баланса. Норматив относится к сборнику ГЭСНп (пусконаладочные работы), таблица 03-01-022."
    }
    
    results = []
    for q in test_questions:
        res = app.ask(q)
        results.append(res)
        print("\n" + "=" * 100)
        print(f"ВОПРОС: {q}")
        print(res["answer"])
        print("=" * 100)
    
    print("\n" + "=" * 80)
    print("Оцениваемые метрики (полная RAG Triad):")
    print("-" * 80)
    print(f"Context Relevance → Средняя семантическая близость чанков к вопросу (cosine)")
    print(f"Faithfulness      → Верность контексту (Vectara HHEM{' ✅' if app.hhem_evaluator.is_available else ' ❌ fallback на cosine'})")
    print(f"Answer Relevance  → Семантическая близость ответа к reference_answer (cosine)")
    print(f"Retrieval Time    → Время на retrieval ДО реранкера (секунды)")
    print("-" * 80)
    print(f"Всего тестовых кейсов: {len(test_questions)}")
    print("=" * 80 + "\n")
    
    triad_results = []
    for i, q in enumerate(test_questions):
        res = results[i]
        answer = res["answer"]
        contexts = res["contexts"]
        ref = reference_answers.get(q, "нет данных")
        
        print(f"\n--- Кейс {i+1}/{len(test_questions)}: {q[:80]}{'...' if len(q) > 80 else ''}")
        
        # 1. Context Relevance (cosine)
        if contexts:
            query_emb = np.array(app.embedding_fn.embed_query(q))
            context_embs = np.array(app.embedding_fn.embed_documents(contexts))
            sims = cosine_similarity([query_emb], context_embs)[0]
            context_relevance = np.mean(sims)
            print(f"  Context Relevance: {context_relevance:.3f}")
        else:
            context_relevance = 0.0
            print("  Context Relevance: 0.000 (нет чанков)")
        
        # 2. Faithfulness (HHEM или cosine fallback)
        if app.hhem_evaluator.is_available:
            faithfulness_score = app.hhem_evaluator.evaluate(answer, contexts)
            print(f"  Faithfulness (HHEM): {faithfulness_score:.3f}")
        else:
            # Fallback на косинусную близость
            if contexts and answer.strip():
                context_embs = np.array(app.embedding_fn.embed_documents(contexts[:3]))
                answer_emb = np.array(app.embedding_fn.embed_query(answer))
                sims = cosine_similarity([answer_emb], context_embs)[0]
                faithfulness_score = np.mean(sims)
                print(f"  Faithfulness (cosine fallback): {faithfulness_score:.3f}")
            else:
                faithfulness_score = 0.0
                print("  Faithfulness: 0.000 (нет данных для оценки)")
        
        # 3. Answer Relevance (cosine)
        if ref.strip():
            answer_emb = np.array(app.embedding_fn.embed_query(answer))
            ref_emb = np.array(app.embedding_fn.embed_query(ref))
            answer_relevance = cosine_similarity([answer_emb], [ref_emb])[0][0]
            print(f"  Answer Relevance: {answer_relevance:.3f}")
        else:
            answer_relevance = 0.0
            print("  Answer Relevance: 0.000 (нет reference)")
        
        # 4. Retrieval Time (до реранкера)
        retrieval_time = res["retrieval_time"]
        print(f"  Retrieval Time (до rerank): {retrieval_time:.3f} sec")
        
        triad_results.append({
            "question": q,
            "context_relevance": context_relevance,
            "faithfulness": faithfulness_score,
            "answer_relevance": answer_relevance,
            "retrieval_time": retrieval_time
        })
        
        time.sleep(0.3)
    
    # Средние значения
    if triad_results:
        print("\n" + "=" * 100)
        print("Итоговые средние по RAG Triad:")
        print("=" * 100)
        avg_context = np.mean([r["context_relevance"] for r in triad_results])
        avg_faithful = np.mean([r["faithfulness"] for r in triad_results])
        avg_answer = np.mean([r["answer_relevance"] for r in triad_results])
        avg_retrieval = np.mean([r["retrieval_time"] for r in triad_results])
        print(f"Средний Context Relevance: {avg_context:.3f}")
        print(f"Средний Faithfulness:      {avg_faithful:.3f}")
        print(f"Средний Answer Relevance:  {avg_answer:.3f}")
        print(f"Средний Retrieval Time (до rerank): {avg_retrieval:.3f} sec")
        print("=" * 100)
        
        # Дополнительная статистика
        print("\nДетальная статистика:")
        print("-" * 100)
        for i, r in enumerate(triad_results, 1):
            q_short = r["question"][:60] + "..." if len(r["question"]) > 60 else r["question"]
            print(f"{i:2d}. {q_short:60s} | CR: {r['context_relevance']:.3f} | F: {r['faithfulness']:.3f} | AR: {r['answer_relevance']:.3f} | RT: {r['retrieval_time']:.3f}s")


def main():
    args = parse_args()
    
    # ---- Загрузка оптимизированных параметров (всегда, если файл существует) ----
    opt_params = load_optimized_params(args.optimized_params)
    # Преобразование имён: reranker_model -> rerank_model
    if "reranker_model" in opt_params:
        opt_params["rerank_model"] = opt_params.pop("reranker_model")
    
    # ---- Формирование итоговых параметров: значения по умолчанию + опт. параметры + аргументы CLI ----
    # Словарь значений по умолчанию из класса Settings (через asdict)
    default_settings = asdict(Settings())
    
    # Объединяем: сначала оптимизированные, затем CLI (CLI переопределяет)
    final_params = {**default_settings, **opt_params}
    
    # Переопределяем из аргументов командной строки (если указаны)
    cli_overrides = {}
    if args.fusion is not None:
        cli_overrides["fusion_strategy"] = args.fusion
    if args.alpha is not None:
        cli_overrides["hybrid_alpha"] = args.alpha
    if args.mode is not None:
        cli_overrides["retrieval_mode"] = args.mode
    if args.thresh is not None:
        cli_overrides["adaptive_threshold"] = args.thresh
    if args.reranker is not None:
        cli_overrides["rerank_model"] = args.reranker
    if args.reranker_type is not None:
        cli_overrides["reranker_type"] = args.reranker_type
    if args.pool_k is not None:
        cli_overrides["pool_k"] = args.pool_k
    if args.bm25_k is not None:
        cli_overrides["bm25_k"] = args.bm25_k
    if args.vec_k is not None:
        cli_overrides["vec_k"] = args.vec_k
    if args.rrf_k is not None:
        cli_overrides["rrf_k"] = args.rrf_k
    if args.rerank_k is not None:
        cli_overrides["rerank_k"] = args.rerank_k
    if args.rerank_threshold is not None:
        cli_overrides["rerank_threshold"] = args.rerank_threshold
    if args.final_k is not None:
        cli_overrides["final_k"] = args.final_k
    
    final_params.update(cli_overrides)
    
    # Вывод используемых параметров
    print_parameters(final_params)
    
    # Создаём объект Settings с итоговыми параметрами
    settings = Settings(
        fusion_strategy=final_params["fusion_strategy"],
        hybrid_alpha=final_params["hybrid_alpha"],
        retrieval_mode=final_params["retrieval_mode"],
        adaptive_threshold=final_params["adaptive_threshold"],
        rerank_model=final_params["rerank_model"],
        reranker_type=final_params["reranker_type"],
        pool_k=final_params["pool_k"],
        bm25_k=final_params["bm25_k"],
        vec_k=final_params["vec_k"],
        rrf_k=final_params["rrf_k"],
        rerank_k=final_params["rerank_k"],
        rerank_threshold=final_params["rerank_threshold"],
        final_k=final_params["final_k"],
    )
    
    app = SmetaRAGApp(settings)
    
    test_questions = [
        "Что такое 01-01-009-11?",
        "Сколько глав в сводном сметном расчете? Что находится в 3 главе?",
        "Какие коэффициенты применяются при работе в зимних условиях?",
        "Какую расценку применить для подвесных потолков Armstrong в офисном здании?",
        "Разъясните возможность применения коэффициента 1,5 «Производство ремонтно-строительных работ осуществляется в жилых помещениях без расселения» ...",
        "Как расценивается проектирование встраиваемых помещений?",
        "Как исчисляется объем работ на прокладку кабельной продукции?",
        "Согласно пункту 1.20.22 раздела I «Общие положения» сборника 20 «Вентиляция и кондиционирование воздуха» ГЭСН 81-02-20-2022 в нормах таблиц 20-06-018, 20-06-019 учтены затраты на прокладку каждого типа коммуникационных трасс (медные трубки, дренаж, питающий кабель) до 10 м. Какие медные трубки и какой длины имеются ввиду?",
        "Что такое сметная стоимость строительства, реконструкции, капитального ремонта, сноса объектов капитального строительства, работ по сохранению объектов культурного наследия согласно Методике 421/пр?",
        "В каких случаях при определении сметной стоимости применяется конъюнктурный анализ?",
        "Какой коэффициент применяется к затратам труда рабочих при производстве работ в стесненных условиях в населенных пунктах?",
        "Что устанавливает Методика определения стоимости работ по подготовке проектной документации?",
        "По какой формуле рассчитывается цена проектных работ при использовании параметров цены в зависимости от натуральных показателей?",
        "Как определяется стоимость работ по подготовке проектной документации, содержащей материалы в форме информационной модели?",
        "Какие корректирующие коэффициенты применяются для объектов жилищно-гражданского строительства при определении стоимости работ с применением информационного моделирования?",
        "Как определяется численность работников технического заказчика?",
        "Какие затраты включаются в расчет на осуществление функций технического заказчика?",
        "Что означает код 65-01-001-01?",
        "Какой код соответствует пусконаладочным работам по регулировке сети вентиляции с количеством сечений до 5?"
    ]
    
    if args.ask:
        result = app.ask(args.ask, debug=args.debug)
        print(result["answer"])
        if args.print_meta:
            print("\n--- META ---")
            print(json.dumps(result["meta"], ensure_ascii=False, indent=2))
        return
    
    evaluate_with_triad(app, test_questions)

if __name__ == "__main__":
    main()