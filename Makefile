SHELL = /bin/bash
.ONESHELL:

CONFIG_PATH := "config.yaml"

.PHONY: train
train:
	python churner/ml/train.py ${CONFIG_PATH}