# Статус на момент передачи

## Проверено на приложенных данных

- mapping читается: 511 961 строк, 511 957 уникальных `doc_id`;
- найдено 8 строк, участвующих в четырёх парах повторных `doc_id`;
- 9 источников, без null в пяти колонках mapping;
- 500 вопросов, 742 ссылки на эталонные документы;
- покрытие эталонных ссылок mapping-файлом: 742/742;
- 30 вопросов без эталонных документов;
- unit-тесты и smoke test перечислены в итоговом журнале проверки.

## Выполненный baseline

Полный title-only BM25 построен по всем 511 961 строкам: 89 809 термов. На 470 вопросах с
непустыми qrels получено:

| Метрика | Значение |
| --- | ---: |
| nDCG@10 | 0,138745 |
| Recall@10 | 0,168584 |
| MAP@10 | 0,118296 |
| MRR@10 | 0,149909 |
| Средняя retrieval latency | 10,5 мс |

Latency измерена одним локальным прогоном после загрузки индекса и не переносится на другое
железо без повторного теста. Полные per-query результаты сохранены в `reports/generated/`.

Проверено 10 unit/integration тестов, включая in-memory Qdrant upload/search, а также отдельный
end-to-end demo hybrid search.

## Добавлено после переноса из NeedleInArXiv

- backend-модуль SPLADE с тем же ordering-контрактом, что в `notebooks/eval_splade.py`;
- скрипт `scripts/build_splade.py` для построения `index/splade`;
- режимы `splade`, `triple_hybrid_v1`, `triple_hybrid_v2`;
- optional reranker hook под `mixedbread-ai/mxbai-rerank-large-v1`;
- demo SPLADE на `hashing_sparse`, чтобы проверить triple-hybrid инфраструктуру без загрузки
  тяжёлой модели.

Метрики triple hybrid нужно пересчитать после построения SPLADE и dense-индексов на той же машине.

## Нельзя честно подтвердить без `embeddings_v1.npy`

- форму и dtype матрицы;
- модель, которой она создана;
- dense-метрики;
- фактическую задержку exact/HNSW;
- размер и память Qdrant-индекса.

Код для этих проверок готов. После копирования `.npy` сначала запустите:

```bash
python scripts/validate_inputs.py --require-embeddings
```

Не вставляйте числа в презентацию до генерации `reports/generated/metrics.json` и
`reports/generated/index_benchmark.json`.
