# ===========================================================================
# Convenience targets. Run `make help` to see them.
# Everything assumes an activated virtualenv (see `make install`).
# ===========================================================================
.PHONY: help install train serve heal traffic test lint compose-up compose-down clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Create a venv and install the project (editable) + dev tools
	python -m venv .venv && . .venv/bin/activate && \
		pip install --upgrade pip && pip install -e ".[dev]"

train:  ## Train + register the initial production model
	python -m scripts.run_pipeline

serve:  ## Run the FastAPI serving app (http://localhost:8000/docs)
	uvicorn mlops.serving.app:app --reload

heal:  ## Run one self-healing cycle (advisory). Use ARGS='--simulate high --auto'
	python -m scripts.self_heal $(ARGS)

traffic:  ## Send synthetic traffic to a running server. Use ARGS='--n 500 --drift medium'
	python -m scripts.simulate_traffic $(ARGS)

test:  ## Run the test suite
	pytest

lint:  ## Lint + format-check with ruff
	ruff check src tests scripts

compose-up:  ## Bring up the full stack (mlflow + serving + prometheus + grafana)
	docker compose up --build

compose-down:  ## Tear the stack down and remove volumes
	docker compose down -v

clean:  ## Remove generated runtime state (models, logs, caches)
	rm -rf mlruns mlartifacts mlflow.db artifacts logs data/generated \
		.pytest_cache .ruff_cache && find . -name __pycache__ -type d -prune -exec rm -rf {} +
