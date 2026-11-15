.PHONY: install test lint fmt run-api docker clean

install:
	pip install -e ".[dev]"

test:
	pytest --cov=transcription_pipeline --cov-report=term-missing

lint:
	ruff check .
	ruff format --check .

fmt:
	ruff format .
	ruff check --fix .

run-api:
	uvicorn transcription_pipeline.api:app --reload --port 8000

docker:
	docker build -t transcription-pipeline .

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov dist build *.egg-info
	find . -name __pycache__ -type d -exec rm -rf {} +
