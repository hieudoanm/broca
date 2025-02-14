format:
	python3 -m black .

check:
	python3 -m ruff check . --fix

update:
	pip install --upgrade --force-reinstall -r requirements.txt

docs:
	rm -rf docs
	mkdir source_docs
	cp README.md source_docs/index.md
	python3 -m mkdocs build
	rm -rf source_docs
	touch docs/.nojekyll
