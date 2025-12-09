PYTHON_VERSION := 3.11
VENV  := .venv
PYTHON_EXEC = python3

.PHONY: env data_process test_model train infer

env:
	@command -v uv >/dev/null 2>&1 || { \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	}
	@echo "Setting up eval environment..."
	@uv venv $(VENV) --python $(PYTHON_VERSION) --no-project
	@uv pip install -r requirements.txt --python $(UV_PYTHON_EXEC)
	@echo "Evaluation environment ready."

data_process:
	@echo "Processing data..."
	@$(PYTHON_EXEC) src/test_data.py
	@echo "Data processing complete."

test_model:
	@echo "Testing model..."
	@$(PYTHON_EXEC) src/test_model.py
	@echo "Model testing complete."

train_enc:
	@echo "Training model..."
	@PYTHONPATH=. $(PYTHON_EXEC) src/train/train_Genc.py
	@echo "Model training complete."

train_t5:
	@echo "Training model..."
	@PYTHONPATH=. $(PYTHON_EXEC) src/train/train_T5.py
	@echo "Model training complete."

infer_enc:
	@echo "Running inference..."
	@PYTHONPATH=. $(PYTHON_EXEC) src/inference/infer_enc.py
	@echo "Inference complete."