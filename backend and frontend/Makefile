.PHONY: install test validate validate-precomputed demo bm25 splade index-v1 index-v2 evaluate serve serve-precomputed up down

install:
	python -m pip install -r requirements.txt

test:
	pytest

validate:
	python scripts/validate_inputs.py

validate-precomputed:
	python scripts/validate_precomputed_indexes.py

demo:
	python scripts/generate_demo_assets.py

bm25:
	python scripts/build_bm25.py

splade:
	python scripts/build_splade.py

index-v1:
	python scripts/index_embeddings.py --iteration 1 --recreate

index-v2:
	python scripts/index_embeddings.py --iteration 2 --recreate

evaluate:
	python scripts/evaluate.py --modes dense_v1 bm25 splade hybrid_v1 triple_hybrid_v1

serve:
	uvicorn service.main:app --host 0.0.0.0 --port 8000 --reload

serve-precomputed:
	cp .env.precomputed.example .env
	uvicorn service.main:app --host 0.0.0.0 --port 8000 --reload

up:
	docker compose up --build

down:
	docker compose down
