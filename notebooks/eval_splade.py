#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Загрузка сохранённого индекса SPLADE и вычисление метрик на уровне документов.
Индекс (splade_index.npz + splade_chunk_ids.npy) уже в правильном порядке —
пересчитывать его не нужно, баг был только в кодировании запросов.
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
# Функция кодирования – ВАЖНО: теперь возвращает order корректно
# и есть возможность отключить сортировку по длине.
# ============================================================
def encode_splade_all(texts, tokenizer, model,
                      batch_size=8, max_length=374, threshold=0.01,
                      show_progress=True, sort_by_length=True):
    """
    Кодирует тексты в CSR-матрицу.

    Возвращает (sparse_matrix, order), где order — перестановка такая,
    что sparse_matrix[i] соответствует ИСХОДНОМУ тексту texts[order[i]].

    Если sort_by_length=False, order == arange(len(texts)) (тривиальный,
    без сортировки) — используйте это для запросов, чтобы не было риска
    перепутать порядок при извлечении результатов.
    """
    n = len(texts)
    if sort_by_length:
        lengths = [len(t.split()) for t in texts]
        order = np.argsort(lengths)          # order[i] = исходный индекс текста, который встал на позицию i
    else:
        order = np.arange(n)

    sorted_texts = [texts[i] for i in order]
    gc.collect()

    all_indices, all_values, all_counts = [], [], []
    total_batches = (len(sorted_texts) + batch_size - 1) // batch_size
    iterator = range(0, len(sorted_texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=total_batches, desc="Encoding")

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

    print("Building CSR matrix...")
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
    return sparse_matrix, order

# ============================================================
# 1. Загрузка сохранённого индекса и данных
# ============================================================
print("\nLoading saved index...")
chunk_embeddings = load_npz("splade_index.npz")
chunk_ids_sorted = np.load("splade_chunk_ids.npy")  # уже в правильном порядке, ничего не пересчитываем
print(f"Index shape: {chunk_embeddings.shape}")
print(f"Number of chunks: {len(chunk_ids_sorted)}")

print("\nLoading question data...")
df_questions = pd.read_parquet("questions_for_sample.parquet")
df_chunks = pd.read_parquet("chunks_fixed_v1.parquet")

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
# 2. Кодирование запросов
# sort_by_length=False -> order тривиален, строка i == queries[i].
# Запросов немного, так что потеря скорости от отсутствия сортировки
# по длине незначительна, а риск рассинхрона убран полностью.
# ============================================================
print("\nEncoding queries...")
query_texts = [q['text'] for q in queries]
query_embeddings, q_order = encode_splade_all(
    texts=query_texts,
    tokenizer=tokenizer,
    model=model,
    batch_size=BATCH_SIZE,
    max_length=MAX_LENGTH,
    threshold=0.01,
    show_progress=True,
    sort_by_length=False,
)
assert np.array_equal(q_order, np.arange(len(queries))), "order должен быть тривиальным при sort_by_length=False"

# ============================================================
# 3. Поиск – chunk_ids_sorted уже соответствует строкам chunk_embeddings,
# так что результат сразу в исходных chunk_id, отдельного маппинга не нужно.
# ============================================================
print("\nSearching...")
similarities = query_embeddings @ chunk_embeddings.T
top_k = 1000
splade_results = {}
for i, query in enumerate(tqdm(queries, desc="Extracting top-K")):
    row = similarities[i].toarray().flatten()
    if top_k < len(row):
        top_k_idx = np.argpartition(row, -top_k)[-top_k:]
        top_k_idx = top_k_idx[np.argsort(row[top_k_idx])[::-1]]
    else:
        top_k_idx = np.argsort(row)[::-1][:top_k]
    splade_results[query['qid']] = [chunk_ids_sorted[idx] for idx in top_k_idx]

print("Search completed.")

with open("splade_results_from_index.pkl", "wb") as f:
    pickle.dump(splade_results, f)
print("Results saved to splade_results_from_index.pkl")

# ============================================================
# 4. Оценка на уровне документов
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
    retrieved_lists=splade_results,
    chunk_to_doc=chunk_to_doc,
    expected_doc_ids=expected_doc_ids_by_qid,
    k_values=[1, 5, 10, 100, 1000]
)
print_metrics_table(doc_metrics, "SPLADE (Doc-level) Results")
print("Done.")