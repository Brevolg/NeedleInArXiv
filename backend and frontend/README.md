# Семантический поиск по документам

Готовый end-to-end проект: проверка данных → BM25 → dense-индекс Qdrant → RRF-гибрид →
оценка → FastAPI и веб-интерфейс. Код сохраняет исходный порядок строк, проверяет размерность
векторов и не публикует выдуманные метрики.

## Что фактически передано

В PDF описан поисковик по научным статьям arXiv. Однако два приложенных Parquet-файла относятся
к EnterpriseRAG-Bench:

- `id_mapping_v1_parquet.parquet`: 511 961 строк, 511 957 уникальных `doc_id`, 9 источников;
- `questions_parquet.parquet`: 500 вопросов и 742 ссылки на эталонные документы;
- все 742 ссылки присутствуют в mapping;
- полных текстов нет — доступны только `title`, `source`, `char_len`, `n_chunks`;
- `embeddings_v1.npy` не был передан.

Поэтому проект по умолчанию работает с корпоративными документами. BM25 строится по заголовкам.
Если указать `CORPUS_PATH` с колонками `doc_id` и `text`/`abstract`, BM25 и сниппеты используют
полный текст. Альтернативный arXiv-конвейер сохранён в `scripts/prepare_arxiv.py`.

## Реализовано

- dense v1: `all-MiniLM-L6-v2`, точный полный просмотр Qdrant (`exact=True`);
- dense v2: SPECTER2, HNSW с настраиваемыми `m`, `ef_construct`, `ef_search`;
- SPLADE sparse retrieval с сохранением row-order mapping как в `notebooks/eval_splade.py`;
- scalar int8 quantization в Qdrant для v2 и отдельный анализ offline-квантизации;
- собственный сохраняемый BM25 на sparse CSR-матрице;
- hybrid v1/v2 через Reciprocal Rank Fusion;
- triple hybrid v1/v2: BM25 + dense + SPLADE через Reciprocal Rank Fusion;
- optional cross-encoder reranker hook для `mixedbread-ai/mxbai-rerank-large-v1`;
- фильтр по источникам и устранение повторных `doc_id` после retrieval;
- nDCG@10, Recall@10, MAP@10, MRR@10, latency и отдельный учёт no-answer запросов;
- PCA, benchmark exact vs HNSW, аудит входных файлов;
- FastAPI `/health`, `/search`, `/api/config` и адаптивная веб-страница;
- Docker Compose с Qdrant;
- unit-тесты и маленький автономный demo-набор.

## Быстрый запуск с вашими файлами

Требования: Python 3.11–3.13, Docker с Compose, около 16 GB RAM для комфортной полной
индексации. GPU не нужен для поиска; он заметно ускоряет повторное кодирование корпуса.

1. В полном архиве два Parquet уже лежат в `data/`. Если вы используете только код из Git,
   положите их вручную и добавьте отсутствующую матрицу:

   ```bash
   cp id_mapping_v1_parquet.parquet data/id_mapping_v1.parquet      # если файла ещё нет
   cp questions_parquet.parquet data/questions.parquet              # если файла ещё нет
   cp /путь/к/embeddings_v1.npy embeddings/embeddings_v1.npy
   cp .env.example .env
   ```

2. Установите зависимости и проверьте согласованность:

   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements-dev.txt
   python scripts/validate_inputs.py --require-embeddings
   pytest
   ```

3. Поднимите Qdrant, постройте BM25 и SPLADE, загрузите dense v1:

   ```bash
   docker compose up -d qdrant
   python scripts/build_bm25.py
   python scripts/build_splade.py
   python scripts/index_embeddings.py --iteration 1 --recreate
   ```

4. Запустите сервис:

   ```bash
   uvicorn service.main:app --host 0.0.0.0 --port 8000
   ```

   Интерфейс: `http://localhost:8000`, OpenAPI: `http://localhost:8000/docs`.
   По умолчанию production `.env.example` использует `triple_hybrid_v1`.

Можно запустить API в Docker после создания индексов: `docker compose up --build api`.

## Критическая проверка модели

`embeddings_v1.npy` должен иметь форму `(511961, D)` и соответствовать исходному порядку строк
mapping. Запросы необходимо кодировать той же моделью, которой были созданы документы. По PDF
предполагается `sentence-transformers/all-MiniLM-L6-v2` и `D=384`, но без самого `.npy` и
sidecar-файла это нельзя подтвердить. Если проверка покажет другую размерность или модель,
измените `MODEL_V1` в `.env`.

Наличие четырёх повторяющихся идентификаторов учтено специально: строки перед Qdrant не
удаляются, потому что это нарушило бы выравнивание с матрицей векторов. Повторы объединяются
только в готовой выдаче.

## Итерация 2

Для SPECTER2 установите дополнительный пакет:

```bash
pip install -e '.[specter2,plots]'
python scripts/encode_corpus.py \
  --model allenai/specter2 \
  --output embeddings/embeddings_v2.npy
python scripts/index_embeddings.py --iteration 2 --recreate
python scripts/benchmark_indexes.py --collection papers_v2 --model allenai/specter2
```

SPECTER2 оправдан для arXiv, потому что это модель научных документов. Для фактически переданного
корпоративного датасета это гипотеза, а не гарантированное улучшение; её необходимо принять или
отклонить по метрикам. Для корпоративных текстов можно задать другую модель через `MODEL_V2`,
не меняя остальной код.

## Оценка

```bash
python scripts/evaluate.py --modes dense_v1 bm25 splade hybrid_v1 triple_hybrid_v1
python scripts/evaluate.py --modes dense_v2 hybrid_v2 triple_hybrid_v2
python scripts/pca_analysis.py \
  --embeddings embeddings/embeddings_v1.npy \
  --output-dir reports/generated/pca_v1
python scripts/quantize_embeddings.py \
  --input embeddings/embeddings_v2.npy \
  --output embeddings/embeddings_v2_quantized
```

30 из 500 вопросов не имеют `expected_doc_ids`. Стандартные ranking-метрики для них не
определены, поэтому код не присваивает им искусственный ноль: они исключаются из nDCG/Recall/MAP
и оцениваются отдельно по доле пустых выдач.

## Автономный smoke test без большого `.npy`

```bash
python scripts/generate_demo_assets.py
set -a; source .env.demo; set +a
python scripts/build_bm25.py
python scripts/index_embeddings.py --iteration 1 --recreate
uvicorn service.main:app --port 8000
```

Demo использует детерминированный hashing-энкодер только для проверки инфраструктуры. Его
качество нельзя выдавать за результат модели.

## SPLADE и ordering

SPLADE-индекс сохраняется в `index/splade` как `matrix.npz`, `row_indices.npy`,
`sources.npy`, `metadata.json`. При сборке документов тексты сортируются по длине для более
ровных батчей, поэтому `row_indices[i]` хранит настоящий `row_index` для строки `i` sparse-матрицы.
Запросы кодируются с `sort_by_length=False`; строка `0` query-матрицы всегда относится к текущему
запросу. Это переносит фикс из `notebooks/eval_splade.py` и не даёт метрикам смотреть на чужие
документы.

## Уже посчитанные индексы

Если BM25S, dense FAISS и SPLADE уже посчитаны в ноутбуках, пересчитывать их не нужно.
Положите файлы так:

```text
external_indexes/
  bm25s_cache/
    chunk_ids.pkl              # или bm25s_chunk_ids.pkl / bm25_chunk_ids.pkl
    index/
      data.csc.index.npy
      indices.csc.index.npy
      indptr.csc.index.npy
      vocab.index.json
      params.index.json
  dense/
    faiss_index.bin
    dense_chunk_ids.pkl
    chunk_embeddings.npy
  splade/
    splade_index.npz
    splade_chunk_ids.npy
data/
  chunks_fixed_v1.parquet
```

`chunks_fixed_v1.parquet` нужен обязательно для chunk-level индексов: backend по нему делает
`chunk_id -> doc_id -> title/source`. В файле должны быть `chunk_id`, `doc_id` и желательно
`chunk_text`.

Затем включите режим:

```bash
cp .env.precomputed.example .env
python scripts/validate_precomputed_indexes.py
uvicorn service.main:app --host 127.0.0.1 --port 8000
```

Для dense важно указать ту же модель, которой строились `chunk_embeddings.npy` и
`faiss_index.bin`. Если индекс был построен через E5 и запросы кодировались с префиксом
`query: `, оставьте `DENSE_QUERY_PREFIX="query: "`.

## Large reranker hook

Реранкер выключен по умолчанию, потому что `mixedbread-ai/mxbai-rerank-large-v1` тяжелее обычного
retrieval и должен запускаться только после готового candidate pool. Для включения:

```bash
RERANKER_ENABLED=true
RERANKER_MODEL=mixedbread-ai/mxbai-rerank-large-v1
RERANKER_CANDIDATES=100
```

После этого в API можно передать `rerank=true`. Backend сначала строит candidate pool выбранным
режимом, например `triple_hybrid_v1`, затем cross-encoder пересортировывает кандидатов.

## Структура

```text
data/          входные parquet и подготовка arXiv
embeddings/    большие матрицы вне Git
index/         сохраняемый BM25; Qdrant живёт в Docker volume
search/        данные, энкодеры, BM25, Qdrant, RRF, метрики, engine
service/       FastAPI, HTML/CSS/JS
scripts/       валидация, кодирование, индексация, оценка, PCA, benchmark
tests/         unit-тесты
docs/          архитектура, аудит, экспериментальный протокол, защита
reports/       только генерируемые фактические результаты
```

Не коммитьте большие `.npy`, Parquet и Qdrant storage. Для команды храните их в общем файловом
хранилище вместе с SHA-256 и JSON-sidecar от `encode_corpus.py`.
