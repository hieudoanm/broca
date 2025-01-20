format:
	python3 -m black .

check:
	python3 -m ruff check . --fix

update:
	pip install --upgrade --force-reinstall -r requirements.txt

build:
	cp README.md source_docs/index.md
	python3 -m mkdocs build
