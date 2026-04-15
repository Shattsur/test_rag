# train_reranker_nemotron.py
from pathlib import Path
import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import ndcg_score
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import logging
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch_dtype.*")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ===================== ПУТИ =====================
BASE_DIR = Path(r"C:\Files\AI\SmetaGPT\smeta_rag_project")
TRAIN_PATH = BASE_DIR / "data" / "rerank" / "train_pairs.jsonl"
EVAL_PATH  = BASE_DIR / "data" / "rerank" / "eval_pairs.jsonl"
MODEL_PATH = BASE_DIR / "models" / "Nemotron-Rerank-1B-4bit"
OUTPUT_DIR = BASE_DIR / "models" / "Nemotron-Rerank-1B-4bit-finetuned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===================== 4-БИТНАЯ КОНФИГУРАЦИЯ =====================
logger.info("🔧 Настройка 4-битной квантовки...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ===================== ТОКЕНИЗАТОР =====================
logger.info("🔄 Загрузка токенизатора...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    padding_side="right",
    fix_mistral_regex=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.sep_token is None:
    tokenizer.sep_token = tokenizer.eos_token

# ===================== ЗАГРУЗКА МОДЕЛИ В 4-БИТ =====================
logger.info("🔄 Загрузка модели в 4-бит...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    num_labels=1,
    problem_type="regression",
    attn_implementation="sdpa" if torch.cuda.is_available() else None,
)

# ===================== ЗАМЕНА КЛАССИФИКАЦИОННОЙ ГОЛОВЫ =====================
logger.info("🔧 Замена классификационной головы...")
hidden_size = model.config.hidden_size
device = next(model.parameters()).device
logger.info(f"   Модель на устройстве: {device}")

for attr in ["score", "classifier", "lm_head"]:
    if hasattr(model, attr):
        delattr(model, attr)

# 🔥 Создаём голову в float32 и сразу на GPU
new_head = torch.nn.Linear(hidden_size, 1, dtype=torch.float32).to(device)
model.score = new_head
model.config.num_labels = 1
model.config.problem_type = "regression"

# ===================== ПОДГОТОВКА К 4-БИТНОМУ ОБУЧЕНИЮ =====================
logger.info("🔧 Подготовка модели для k-bit training...")
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

# ===================== LoRA КОНФИГУРАЦИЯ =====================
logger.info("🔧 Настройка LoRA...")
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    inference_mode=False,
    modules_to_save=["score"],  # 🔥 Тренируем и новую голову
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ===================== ПРОВЕРКА НАСТРОЙКИ =====================
try:
    score_layer = model.base_model.model.score
    logger.info(f"✓ score dtype: {score_layer.weight.dtype}")
    logger.info(f"✓ score requires_grad: {score_layer.weight.requires_grad}")
    logger.info(f"✓ score device: {score_layer.weight.device}")
    
    for name, module in model.named_modules():
        if "q_proj" in name and hasattr(module, "weight"):
            weight_type = type(module.weight).__name__
            logger.info(f"✓ Base weight type: {weight_type}")
            break
            
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"✓ Trainable parameters: {trainable:,}")
except Exception as e:
    logger.warning(f"⚠️ Не удалось проверить слои: {e}")

# ===================== ДАТАСЕТЫ =====================
logger.info("📊 Загрузка датасетов...")
train_dataset = load_dataset("json", data_files=str(TRAIN_PATH), split="train")
eval_dataset_original = load_dataset("json", data_files=str(EVAL_PATH), split="train")
logger.info(f"   Train: {len(train_dataset):,}, Eval: {len(eval_dataset_original):,}")

# ===================== ТОКЕНИЗАЦИЯ =====================
def tokenize_function(examples):
    texts = [f"{q} {tokenizer.sep_token} {d}" for q, d in zip(examples["query"], examples["document"])]
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_attention_mask=True,
    )
    tokenized["labels"] = [float(x) for x in examples["label"]]  # 🔥 float для регрессии
    return tokenized

logger.info("🔄 Токенизация...")
train_dataset = train_dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=["query", "document", "label"],
    desc="Tokenizing train"
)
eval_dataset = eval_dataset_original.map(
    tokenize_function, 
    batched=True, 
    remove_columns=["query", "document", "label"],
    desc="Tokenizing eval"
)

# ===================== NDCG@10 CALLBACK =====================
class NDCGEvalCallback(TrainerCallback):
    """Callback для вычисления NDCG@10 и ручного сохранения лучшей модели"""
    
    def __init__(self, eval_dataset_original, model, tokenizer, output_dir, 
                 k=10, eval_steps=50, max_queries=500):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir) / "ndcg_best"
        self.k = k
        self.eval_steps = eval_steps
        self.best_ndcg = -1.0
        
        # Группируем по запросам
        self.groups = defaultdict(lambda: {"docs": [], "labels": []})
        for row in eval_dataset_original:
            q = row["query"]
            self.groups[q]["docs"].append(row["document"])
            self.groups[q]["labels"].append(row["label"])
        
        # Подвыборка для скорости
        self.queries = list(self.groups.keys())
        if len(self.queries) > max_queries:
            import random
            random.seed(42)
            self.queries = random.sample(self.queries, max_queries)
            logger.info(f"🎲 NDCG evaluator: подвыборка {len(self.queries)} из {len(self.groups)} запросов")
        else:
            logger.info(f"📊 NDCG evaluator: {len(self.queries)} уникальных запросов")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Вычисляем NDCG@k и сохраняем лучшую модель вручную"""
        logger.info(f"🔍 Вычисление NDCG@{self.k}...")
        self.model.eval()
        
        ndcg_scores = []
        
        with torch.no_grad():
            for query in self.queries:
                group = self.groups[query]
                docs, labels = group["docs"], group["labels"]
                
                # Формируем пары для инференса
                texts = [f"{query} {self.tokenizer.sep_token} {d}" for d in docs]
                inputs = self.tokenizer(
                    texts,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.model.device)
                
                # Предсказания
                outputs = self.model(**inputs)
                scores = outputs.logits.squeeze(-1).cpu().numpy()
                
                # NDCG для этого запроса
                if len(scores) >= 2:
                    k = min(self.k, len(scores))
                    y_true = np.array(labels).reshape(1, -1)
                    y_score = scores.reshape(1, -1)
                    ndcg = ndcg_score(y_true, y_score, k=k)
                    ndcg_scores.append(ndcg)
        
        if ndcg_scores:
            avg_ndcg = float(np.mean(ndcg_scores))
            logger.info(f"📈 NDCG@{self.k}: {avg_ndcg:.5f} (best: {self.best_ndcg:.5f})")
            
            # 🔥 РУЧНОЕ СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ ПО NDCG
            if avg_ndcg > self.best_ndcg:
                self.best_ndcg = avg_ndcg
                logger.info(f"✨ Новая лучшая модель! NDCG@{self.k}={avg_ndcg:.5f}")
                logger.info(f"💾 Сохраняем в: {self.output_dir}")
                
                try:
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                    # Сохраняем модель с адаптерами (быстро)
                    self.model.save_pretrained(str(self.output_dir))
                    self.tokenizer.save_pretrained(str(self.output_dir))
                    # Сохраняем метрику для справки
                    with open(self.output_dir / "ndcg_metric.txt", "w") as f:
                        f.write(f"NDCG@{self.k}: {avg_ndcg:.5f}\n")
                        f.write(f"Step: {state.global_step}\n")
                    logger.info(f"✅ Модель сохранена!")
                except Exception as e:
                    logger.error(f"❌ Ошибка сохранения: {e}")
        
        return control

# ===================== TRAINING ARGS =====================
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,
    learning_rate=3e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    max_grad_norm=0.3,
    bf16=True,
    gradient_checkpointing=False,  # 🔥 Отключено для стабильности на Windows
    gradient_checkpointing_kwargs={"use_reentrant": False},
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=1,
    load_best_model_at_end=True,
    # 🔥 Используем eval_loss для встроенной логики Trainer
    metric_for_best_model="eval_loss",
    greater_is_better=False,  # loss: чем меньше, тем лучше
    report_to="none",
    logging_steps=20,
    seed=42,
    dataloader_num_workers=0,  # Windows: избегаем проблем с multiprocessing
    dataloader_pin_memory=False,
)

# ===================== TRAINER + CALLBACK =====================
ndcg_callback = NDCGEvalCallback(
    eval_dataset_original=eval_dataset_original,
    model=model,
    tokenizer=tokenizer,
    output_dir=str(OUTPUT_DIR),  # 🔥 Путь для сохранения лучшей модели
    k=10,
    eval_steps=50,
    max_queries=500
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3),
        ndcg_callback,  # 🔥 Наш callback с NDCG
    ],
)

# ===================== ЗАПУСК =====================
logger.info("🚀 Запуск обучения...")
try:
    trainer.train()
except KeyboardInterrupt:
    logger.warning("⚠️ Прервано пользователем")
    raise
except Exception as e:
    logger.error(f"❌ Ошибка: {e}")
    raise

# ===================== СОХРАНЕНИЕ =====================
logger.info("💾 Финальное сохранение модели...")
try:
    # 🔥 Приоритет: лучшая модель по NDCG (если она есть)
    ndcg_best_path = OUTPUT_DIR / "ndcg_best"
    if ndcg_callback.best_ndcg > 0 and ndcg_best_path.exists():
        logger.info(f"🏆 Используем лучшую модель по NDCG@10: {ndcg_callback.best_ndcg:.5f}")
        logger.info(f"✅ Лучшая модель уже сохранена в: {ndcg_best_path}")
        
        # Опционально: создать слитую версию для инференса
        merge_path = OUTPUT_DIR / "final_merged"
        logger.info(f"🔄 Создаём слитую версию в: {merge_path}")
        try:
            # Загружаем лучшую модель и сливаем адаптеры
            from transformers import AutoModelForSequenceClassification
            best_model = AutoModelForSequenceClassification.from_pretrained(
                str(ndcg_best_path),
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                num_labels=1,
            )
            # Если модель уже с адаптерами — сливаем
            if hasattr(best_model, "merge_and_unload"):
                merged = best_model.merge_and_unload()
                merged.save_pretrained(str(merge_path), safe_serialization=True)
                tokenizer.save_pretrained(str(merge_path))
                logger.info(f"✅ Слитая модель сохранена в: {merge_path}")
        except Exception as merge_err:
            logger.warning(f"⚠️ Не удалось создать слитую версию: {merge_err}")
            logger.info(f"📁 Используйте модель с адаптерами из: {ndcg_best_path}")
    else:
        # Фоллбэк: стандартное сохранение
        logger.info("⚠️ NDCG-модель не найдена, используем стандартное сохранение...")
        merged = model.merge_and_unload()
        final_dir = OUTPUT_DIR / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(final_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(final_dir))
        training_args.save_to_json(str(final_dir / "training_args.json"))
        logger.info(f"✅ Готово! Модель в: {final_dir}")
    
except Exception as e:
    logger.error(f"❌ Ошибка сохранения: {e}")
    fallback_dir = OUTPUT_DIR / "fallback"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(fallback_dir))
    tokenizer.save_pretrained(str(fallback_dir))
    logger.info(f"🔄 Сохранено в: {fallback_dir}")

logger.info("=" * 60)
logger.info("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
if ndcg_callback.best_ndcg > 0:
    logger.info(f"🏆 Лучший NDCG@10: {ndcg_callback.best_ndcg:.5f}")
logger.info("=" * 60)