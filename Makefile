# Minimal makefile for Sphinx documentation

docs:
	sphinx-build -j auto -b html docs docs/_build/html

.PHONY: help Makefile docs