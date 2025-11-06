SHELL := /bin/bash
PY := 3.11

.PHONY: install
install:
	uv venv --python=$(PY); \
	uv sync;

