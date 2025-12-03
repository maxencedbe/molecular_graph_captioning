PYTHON_VERSION := 3.11
VENV  := .venv

.PHONY: env data_process

env:
	@command -v uv >/dev/null 2>&1 || { \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	}
	@echo "Setting up eval environment..."
	@uv venv $(VENV) --python $(PYTHON_VERSION) --no-project
	@uv pip install -r requirements.txt --python $(VENV)/bin/python
	@echo "Evaluation environment ready."

data_process:
	@echo "Processing data..."
	@$(VENV)/bin/python src/test_data.py
	@echo "Data processing complete."
