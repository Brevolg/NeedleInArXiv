#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Загрузка сохранённого индекса SPLADE и вычисление метрик на уровне документов.
Предполагается, что индекс сохранён с отсортированными chunk_ids.
Восстанавливает исходные ID для корректной оценки.
"""
import os
import gc
import pickle
import warnings
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from scipy.sparse import csr_matrix, load_npz

warnings.filterwarnings('ignore')

# ============================================================
# Настройки
# ============================================================
MODEL_NAME = "naver/splade-cocondenser-ensembledistil"
MAX_LENGTH = 374
BATCH_SIZE = 32          # для кодирования запросов
OMP_NUM_THREADS = 8
os.environ["OMP_NUM_THREADS"] = str(OMP_NUM_THREADS)
torch.set_num_threads(OMP_NUM_THREADS)
torch.backends.cudnn.benchmark = True

# ============================================================
# Загрузка модели SPLADE (для кодирования запросов)
# ============================================================
print(f"Loading {MODEL_NAME} in FP16 on GPU...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
model = model.half().cuda()
model.eval()
print(f"Model loaded on {next(model.parameters()).device} (FP16)")

# ============================================================
# Функция кодирования (для запросов) – копия из основного скрипта
# ============================================================
def encode_splade_all(texts, tokenizer, model,
                      batch_size=8, max_length=374, threshold=0.01,
                      show_progress=True):
    # Сортируем по длине
    lengths = [len(t.split()) for t in texts]
    order = np.argsort(lengths)
    sorted_texts = [texts[i] for i in order]
    del lengths, texts
    gc.collect()

    all_indices, all_values, all_counts = [], [], []
    total_batches = (len(sorted_texts) + batch_size - 1) // batch_size
    iterator = range(0, len(sorted_texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=total_batches, desc="Encoding queries")

    with torch.inference_mode():
        for start_idx in iterator:
            batch_texts = sorted_texts[start_idx:start_idx + batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True,
                               max_length=max_length, return_tensors='pt')
            inputs = {k: v.cuda() for k, v in inputs.items()}

            logits = model(**inputs).logits
            max_logits, _ = torch.max(logits, dim=1)
            del logits

            weights = torch.log1p(torch.relu(max_logits))
            del max_logits

            mask = weights > threshold
            rows, cols = torch.where(mask)
            vals = weights[mask]

            if vals.numel() > 0:
                counts = torch.bincount(rows, minlength=len(batch_texts))
                all_counts.append(counts.cpu().numpy())
                all_indices.append(cols.cpu().numpy())
                all_values.append(vals.cpu().numpy())
            else:
                all_counts.append(np.zeros(len(batch_texts), dtype=np.int64))

            del weights, mask, rows, cols, vals, inputs
            torch.cuda.empty_cache()
            gc.collect()

    print("Building CSR matrix for queries...")
    all_counts = np.concatenate(all_counts)
    indptr = np.concatenate([[0], np.cumsum(all_counts)])
    if all_indices:
        indices = np.concatenate(all_indices)
        values = np.concatenate(all_values)
    else:
        indices = np.array([], dtype=np.int64)
        values = np.array([], dtype=np.float32)

    sparse_matrix = csr_matrix((values, indices, indptr),
                               shape=(len(sorted_texts), tokenizer.vocab_size))
    # возвращаем матрицу и порядок сортировки (для восстановления)
    return sparse_matrix, order

# ============================================================
# 1. Загрузка сохранённого индекса и данных
# ============================================================
print("\nLoading saved index...")
chunk_embeddings = load_npz("splade_index.npz")
chunk_ids_sorted = np.load("splade_chunk_ids.npy")  # отсортированные ID
print(f"Index shape: {chunk_embeddings.shape}")
print(f"Number of chunks: {len(chunk_ids_sorted)}")

# Загружаем исходные данные для восстановления порядка и ground truth
print("\nLoading chunk data for ground truth...")
df_chunks = pd.read_parquet("chunks_fixed_v1.parquet")
df_questions = pd.read_parquet("questions_for_sample.parquet")

# Строим маппинг: отсортированный chunk_id -> исходный chunk_id
chunk_ids_original = df_chunks['chunk_id'].values
chunk_texts = df_chunks['chunk_text'].tolist()
lengths = [len(t.split()) for t in chunk_texts]
sort_order = np.argsort(lengths)
sorted_chunk_ids_original = chunk_ids_original[sort_order]
sorted_to_original = {sorted_id: orig_id for sorted_id, orig_id in zip(sorted_chunk_ids_original, chunk_ids_original[sort_order])}

# Проверяем, что загруженные ID совпадают с восстановленными (для уверенности)
assert np.array_equal(chunk_ids_sorted, sorted_chunk_ids_original), "Mismatch in sorted chunk IDs!"

# ============================================================
# 2. Подготовка ground truth на уровне документов
# ============================================================
chunk_to_doc = dict(zip(df_chunks['chunk_id'], df_chunks['doc_id']))
expected_doc_ids_by_qid = {
    row['question_id']: set(row['expected_doc_ids'])
    for _, row in df_questions.iterrows()
}

queries = []
for _, row in df_questions.iterrows():
    queries.append({
        'qid': row['question_id'],
        'text': row['question']
    })
print(f"Number of queries: {len(queries)}")

# ============================================================
# 3. Кодирование запросов
# ============================================================
print("\nEncoding queries...")
query_texts = [q['text'] for q in queries]
query_embeddings, _ = encode_splade_all(
    texts=query_texts,
    tokenizer=tokenizer,
    model=model,
    batch_size=BATCH_SIZE,
    max_length=MAX_LENGTH,
    threshold=0.01,
    show_progress=True
)

# ============================================================
# 4. Поиск (результаты в отсортированных ID)
# ============================================================
print("\nSearching...")
similarities = query_embeddings @ chunk_embeddings.T
top_k = 1000
splade_results_sorted = {}
for i, query in enumerate(tqdm(queries, desc="Extracting top-K")):
    row = similarities[i].toarray().flatten()
    if top_k < len(row):
        top_k_idx = np.argpartition(row, -top_k)[-top_k:]
        top_k_idx = top_k_idx[np.argsort(row[top_k_idx])[::-1]]
    else:
        top_k_idx = np.argsort(row)[::-1][:top_k]
    # Результаты в отсортированных ID
    splade_results_sorted[query['qid']] = [chunk_ids_sorted[idx] for idx in top_k_idx]

# ============================================================
# 5. Преобразование в исходные ID для оценки
# ============================================================
splade_results_original = {}
for qid, sorted_ids in splade_results_sorted.items():
    splade_results_original[qid] = [sorted_to_original[sid] for sid in sorted_ids]

print("Search completed.")

# Сохраняем результаты в исходных ID
with open("splade_results_from_index.pkl", "wb") as f:
    pickle.dump(splade_results_original, f)
print("Results saved to splade_results_from_index.pkl")

# ============================================================
# 6. Оценка на уровне документов
# ============================================================
def compute_metrics_doc_level(retrieved_lists, chunk_to_doc, expected_doc_ids,
                              k_values=[1, 5, 10, 100, 1000]):
    results = []
    for k in k_values:
        recalls, ndcgs, mrrs = [], [], []
        for qid, expected_docs in tqdm(expected_doc_ids.items(), desc=f"Doc metrics@k={k}", leave=False):
            if qid not in retrieved_lists or not expected_docs:
                continue
            retrieved_chunks = retrieved_lists[qid][:k]
            seen_docs, seen_set = [], set()
            for chunk_id in retrieved_chunks:
                doc_id = chunk_to_doc.get(chunk_id)
                if doc_id is not None and doc_id not in seen_set:
                    seen_set.add(doc_id)
                    seen_docs.append(doc_id)
            n_expected = len(expected_docs)
            n_found = len(seen_set & expected_docs)
            recalls.append(n_found / n_expected)
            dcg = 0.0
            for i, doc_id in enumerate(seen_docs):
                rel = 1.0 if doc_id in expected_docs else 0.0
                dcg += rel / np.log2(i + 2)
            ideal_rels = [1.0] * min(n_expected, k)
            idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
            rr = 0.0
            for i, doc_id in enumerate(seen_docs):
                if doc_id in expected_docs:
                    rr = 1.0 / (i + 1)
                    break
            mrrs.append(rr)
        results.append({
            'k': k,
            'Recall@k': np.mean(recalls) if recalls else 0.0,
            'NDCG@k': np.mean(ndcgs) if ndcgs else 0.0,
            'MRR@k': np.mean(mrrs) if mrrs else 0.0,
            'n_queries': len(recalls),
        })
    return pd.DataFrame(results)

def print_metrics_table(results_df, title=""):
    if title:
        print(f"\n{'='*60}\n  {title}\n{'='*60}")
    print(results_df.round(4).to_string(index=False))
    print()

print("\n" + "="*60)
doc_metrics = compute_metrics_doc_level(
    retrieved_lists=splade_results_original,
    chunk_to_doc=chunk_to_doc,
    expected_doc_ids=expected_doc_ids_by_qid,
    k_values=[1, 5, 10, 100, 1000]
)
print_metrics_table(doc_metrics, "SPLADE (Doc‑level) Results")
print("✅ Done.")
