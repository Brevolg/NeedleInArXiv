# Экспериментальный протокол

1. Зафиксировать mapping, questions и embeddings по SHA-256.
2. Проверить форму/порядок векторов через `validate_inputs.py`.
3. Построить BM25 и SPLADE по одинаковому текстовому полю для всех сравнений.
4. Загрузить dense v1/v2 в разные коллекции Qdrant.
5. Выполнить один warm-up проход, затем оценить все вопросы в неизменном порядке.
6. Сохранить per-query выдачу и latency, затем агрегировать метрики.
7. Для HNSW сравнить `ef_search` по recall относительно exact и latency.
8. Повторить latency-прогон минимум три раза на одном компьютере; публиковать медиану и p95.

## Метрики

- nDCG@10: качество порядка релевантных документов;
- Recall@10: доля всех известных релевантных документов в top-10;
- MAP@10: точность в позициях релевантных документов;
- MRR@10: позиция первого релевантного документа;
- latency: end-to-end время retrieval без HTTP-сети;
- HNSW recall against exact: потери приближённого индекса.

Запросы без qrels исключаются из ranking-метрик как неопределённые и анализируются отдельно.
Для статистического сравнения сохраняются per-query значения; только средних недостаточно.

## Сравнения

| Сравнение | Что проверяет |
| --- | --- |
| BM25 vs dense v1 | семантическая модель против лексического baseline |
| BM25 vs SPLADE | обычный lexical sparse против learned sparse |
| dense v1 vs hybrid v1 | вклад RRF без влияния ANN |
| hybrid v1 vs triple_hybrid_v1 | вклад SPLADE в candidate pool |
| exact v2 vs HNSW v2 | компромисс latency/ANN recall |
| float v2 vs Qdrant int8 | влияние квантизации на память, latency и качество |
| triple_hybrid_v1 vs triple_hybrid_v2 | итоговое сравнение итераций |
| triple_hybrid_v1 vs triple_hybrid_v1 + rerank | вклад второго этапа reranker |
