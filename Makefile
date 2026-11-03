.PHONY: help install train fast app test lint clean

help:
	@echo "CanopyIntel: Forest Loss Prediction & Deforestation Hotspot System"
	@echo "------------------------------------------------------------------"
	@echo "make install    : Install project dependencies"
	@echo "make train      : Execute full end-to-end training pipeline"
	@echo "make fast       : Execute accelerated training pipeline (--fast)"
	@echo "make app        : Launch Streamlit web dashboard"
	@echo "make test       : Run test suite (pytest -q)"
	@echo "make clean      : Remove cache files and temporary artifacts"

install:
	pip install -r requirements.txt

train:
	python -m src.train

fast:
	python -m src.train --fast

app:
	streamlit run app/streamlit_app.py

test:
	python -m pytest -q

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

