#!/bin/bash
.PHONY build_venv delete_venv install train :

build_venv:
	python3 -m venv .venv

delete_venv:
	sudo rm -r .venv

install: build_venv
	. .venv/bin/activate && \
	python3 -m pip list && \
	python3 -m pip install -U pip &&\
	python3 -m pip install -r ./rtdetr_pytorch/requirements.txt && \
	deactivate
train:
	. .venv/bin/activate && \
	bash rtdetr_pytorch/tools/train.sh

predict:
	. .venv/bin/activate && \
	bash rtdetr_pytorch/tools/predict.sh

predict_from_dir:
	. .venv/bin/activate && \
	bash rtdetr_pytorch/tools/predict_from_dir.sh