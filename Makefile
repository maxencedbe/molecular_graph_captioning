PYTHON_VERSION := 3.11
VENV  := .venv

UNAME_S := $(shell uname -s)

ifeq ($(findstring NT,$(UNAME_S)),NT)
    PYTHON_EXEC := $(VENV)/Scripts/python.exe
    UV_PYTHON_EXEC := $(VENV)/Scripts/python.exe
else
    PYTHON_EXEC := $(VENV)/bin/python
    UV_PYTHON_EXEC := $(VENV)/bin/python
endif

.PHONY: env embeddings train infer

env:
	@command -v uv >/dev/null 2>&1 || { \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	}
	@echo "Setting up eval environment..."
	@uv venv $(VENV) --python $(PYTHON_VERSION) --no-project
	@uv pip install -r requirements.txt --python $(UV_PYTHON_EXEC)
	@echo "Virtual environment ready."

embeddings:
	@echo "Generating embeddings..."
	@PYTHONPATH=. $(PYTHON_EXEC) src/data/generate_description_embeddings.py
	@echo "Embeddings generation complete."

train:
	@echo "Training model..."
	@PYTHONPATH=. $(PYTHON_EXEC) src/train/train.py
	@echo "Model training complete."

infer:
	@echo "Running inference..."
	@PYTHONPATH=. $(PYTHON_EXEC) src/inference/inference.py
	@echo "Inference complete."