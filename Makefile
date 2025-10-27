PY?=python3
PIP?=$(PY) -m pip
UVICORN?=uvicorn
APP=app.main:app
PORT?=8000
HOST?=0.0.0.0

.PHONY: help
help:
	@echo "Targets: install, dev, lint, test, run, docker-build, docker-run, compose-up, compose-down"

.PHONY: install
install:
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

.PHONY: dev
dev:
	$(PIP) install -r requirements-dev.txt
	pre-commit install

.PHONY: lint
lint:
	pre-commit run --all-files

.PHONY: test
test:
	SKIP_WARMUP=1 pytest -q

.PHONY: run
run:
	$(UVICORN) $(APP) --host $(HOST) --port $(PORT) --reload

.PHONY: docker-build
docker-build:
	docker build -t vision-det:latest .

.PHONY: docker-run
docker-run:
	docker run --rm -it -p $(PORT):8000 --env-file .env vision-det:latest

.PHONY: compose-up
compose-up:
	docker compose up --build -d

.PHONY: compose-down
compose-down:
	docker compose down --remove-orphans
