VENV := .venv
PY   := $(VENV)/bin/python
CU130 := https://download.pytorch.org/whl/cu130

.PHONY: help venv install install-predict doctor test test-fast sample run deck clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-16s\033[0m %s\n",$$1,$$2}'

venv: ## create a 3.12 virtualenv (3.14 breaks whisperx)
	uv venv --python 3.12 $(VENV)

install: venv ## renderer only, no torch
	uv pip install --python $(PY) -e '.[dev]'

install-predict: venv ## cu130 torch + tribev2 (large)
	uv pip install --python $(PY) torch torchvision torchaudio --index-url $(CU130)
	uv pip install --python $(PY) -e '.[predict,dev]'
	# tribev2 pulls a PyPI torch; put the Spark wheel back.
	uv pip install --python $(PY) --force-reinstall torch torchvision torchaudio --index-url $(CU130)

doctor: ## preflight this Spark
	$(PY) -m videocortex_spark doctor

test: ## full suite
	$(PY) -m pytest

test-fast: ## skip the tests that rasterise surfaces
	$(PY) -m pytest -m "not slow"

sample: ## build and render the synthetic example (no model needed)
	$(PY) examples/make_sample.py
	$(PY) -m videocortex_spark draw examples/sample_run/predictions.npy --max-frames 6

run: ## venv + tests + synthetic sample
	./setup_and_run.sh

deck: ## loopback command deck (http://127.0.0.1:8730)
	$(PY) -m videocortex_spark serve

clean:
	rm -rf runs .videocortex-spark-cache .pytest_cache **/__pycache__ *.egg-info
