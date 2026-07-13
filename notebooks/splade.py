#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
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
from scipy.sparse import csr_matrix, save_npz

warnings.filterwarnings('ignore')

# ============================================================
# 1.  Загрузка данных
# ============================================================
print("Loading data...")
df_chunks = pd.read_parquet("chunks_fixed_v1.parquet")
df_questions = pd.read_parquet("questions_for_sample.parquet")
print(f"Chunks: {len(df_chunks)} rows, {df_chunks['doc_id'].nunique()} unique docs")
print(f"Questions: {len(df_questions)}")

# ============================================================
# 2.  Ground Truth (для оценки)
# ============================================================
def prepare_ground_truth_chunks(df_questions, df_chunks):
    doc_to_chunks = defaultdict(set)
    for _, row in tqdm(df_chunks.iterrows(), total=len(df_chunks), desc="Mapping doc→chunks"):
        doc_to_chunks[row['doc_id']].add(row['chunk_id'])

    gt = {}
    for _, row in tqdm(df_questions.iterrows(), total=len(df_questions), desc="Building ground truth (chunks)"):
        qid = row['question_id']
        expected_chunks = set()
        for doc_id in row['expected_doc_ids']:
            expected_chunks.update(doc_to_chunks.get(doc_id, set()))
        gt[qid] = expected_chunks
    return gt

ground_truth_chunks = prepare_ground_truth_chunks(df_questions, df_chunks)
chunk_to_doc = dict(zip(df_chunks['chunk_id'], df_chunks['doc_id']))
expected_doc_ids_by_qid = {
    row['question_id']: set(row['expected_doc_ids'])
    for _, row in df_questions.iterrows()
}

queries = []
for _, row in tqdm(df_questions.iterrows(), total=len(df_questions), desc="Preparing queries"):
    queries.append({
        'qid': row['question_id'],
        'text': row['question'],
        'expected_chunks': ground_truth_chunks[row['question_id']]
    })
print(f"Queries for evaluation: {len(queries)}")

# ============================================================
# 3.  Функции оценки (doc‑level)
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

# ============================================================
# 4.  SPLADE – загрузка модели и кодирование (один проход)
# ============================================================
BATCH_SIZE = 96
MAX_LENGTH = 374
OMP_NUM_THREADS = 8
os.environ["OMP_NUM_THREADS"] = str(OMP_NUM_THREADS)
torch.set_num_threads(OMP_NUM_THREADS)
torch.backends.cudnn.benchmark = True

MODEL_NAME = "naver/splade-cocondenser-ensembledistil"
print(f"\nLoading {MODEL_NAME} in FP16 on GPU...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
model = model.half().cuda()
model.eval()
print(f"Model loaded on {next(model.parameters()).device} (FP16)")

def encode_splade_all(texts, tokenizer, model,
                      batch_size=32, max_length=374, threshold=0.01,
                      show_progress=True):
    """
    Кодирует все тексты в CSR-матрицу (без сохранения на диск).
    Возвращает матрицу в порядке сортировки по длине (для эффективности)
    и массив порядковых индексов, чтобы восстановить исходный порядок, если нужно.
    """
    # Сортируем тексты по длине для эффективного batching
    lengths = [len(t.split()) for t in texts]
    order = np.argsort(lengths)          # индексы в порядке возрастания длины
    sorted_texts = [texts[i] for i in order]
    del lengths, texts
    gc.collect()

    all_indices, all_values, all_counts = [], [], []
    total_batches = (len(sorted_texts) + batch_size - 1) // batch_size
    iterator = range(0, len(sorted_texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=total_batches, desc="SPLADE encoding")

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

    # Сборка CSR
    print("Building CSR matrix...")
    t0 = time.time()
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
    print(f"CSR built in {time.time()-t0:.1f}s")

    # Возвращаем матрицу (в отсортированном порядке) и массив order
    return sparse_matrix, order

# ============================================================
# 5.  Кодирование чанков (один проход)
# ============================================================
print("\nEncoding all chunks in one pass...")
t_start = time.time()
splade_chunk_texts = df_chunks['chunk_text'].tolist()
splade_chunk_ids = np.array(df_chunks['chunk_id'].tolist())  # исходный порядок

splade_chunk_embeddings, sort_order = encode_splade_all(
    texts=splade_chunk_texts,
    tokenizer=tokenizer,
    model=model,
    batch_size=BATCH_SIZE,
    max_length=MAX_LENGTH,
    threshold=0.01,
    show_progress=True,
)

# Сортируем chunk_ids в том же порядке, что и строки матрицы
sorted_chunk_ids = splade_chunk_ids[sort_order]

t_end = time.time()
print(f"\nAll chunks encoded in {t_end-t_start:.1f}s")
print(f"Shape: {splade_chunk_embeddings.shape}")
print(f"Sparsity: {1 - splade_chunk_embeddings.nnz / (splade_chunk_embeddings.shape[0] * splade_chunk_embeddings.shape[1]):.4f}")
print(f"Avg non-zero per chunk: {splade_chunk_embeddings.nnz / splade_chunk_embeddings.shape[0]:.0f}")

# Сохраняем индекс (матрицу и отсортированные chunk_ids)
print("\nSaving index for sharing...")
save_npz("splade_index.npz", splade_chunk_embeddings)
np.save("splade_chunk_ids.npy", sorted_chunk_ids)
print("Index saved: splade_index.npz, splade_chunk_ids.npy")

# ============================================================
# 6.  Поиск SPLADE (используем отсортированные ID)
# ============================================================
def search_splade(queries, tokenizer, model, chunk_embeddings, chunk_ids, top_k=1000):
    query_texts = [q['text'] for q in queries]
    # Кодируем запросы – не важно, в каком порядке, возвращаем матрицу и порядок (не используем)
    query_embeddings, _ = encode_splade_all(
        texts=query_texts,
        tokenizer=tokenizer,
        model=model,
        batch_size=32,
        max_length=MAX_LENGTH,
        threshold=0.01,
        show_progress=False
    )
    similarities = query_embeddings @ chunk_embeddings.T
    results = {}
    for i, query in enumerate(tqdm(queries, desc="SPLADE search")):
        row = similarities[i].toarray().flatten()
        if top_k < len(row):
            top_k_idx = np.argpartition(row, -top_k)[-top_k:]
            top_k_idx = top_k_idx[np.argsort(row[top_k_idx])[::-1]]
        else:
            top_k_idx = np.argsort(row)[::-1][:top_k]
        # Используем отсортированные chunk_ids
        results[query['qid']] = [chunk_ids[idx] for idx in top_k_idx]
    return results

print("\nSearching with SPLADE...")
t0 = time.time()
splade_results = search_splade(
    queries=queries,
    tokenizer=tokenizer,
    model=model,
    chunk_embeddings=splade_chunk_embeddings,
    chunk_ids=sorted_chunk_ids,           # важный момент
    top_k=1000
)
t1 = time.time()
print(f"SPLADE search completed in {t1-t0:.1f}s ({(t1-t0)/len(queries):.3f}s per query)")

# Сохраняем результаты поиска для коллег
with open("splade_results.pkl", "wb") as f:
    pickle.dump(splade_results, f)
print("Search results saved: splade_results.pkl")

# ============================================================
# 7.  Оценка на уровне документов (опционально, для себя)
# ============================================================
doc_metrics = compute_metrics_doc_level(
    retrieved_lists=splade_results,
    chunk_to_doc=chunk_to_doc,
    expected_doc_ids=expected_doc_ids_by_qid,
    k_values=[1, 5, 10, 100, 1000]
)
print_metrics_table(doc_metrics, "SPLADE (Doc‑level) Results")

print("\nDone. Index and results are saved for sharing.")
