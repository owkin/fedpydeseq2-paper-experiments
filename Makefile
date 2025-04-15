install-poetry: ## Install poetry
	pip install poetry=="1.8.2"

install-all: ## Install all package and development dependencies
	poetry install --with linting

clean: clean-build clean-pyc ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -path ./.venv -prune -false -o -name '*.egg-info' -exec rm -fr {} +
	find . -path ./.venv -prune -false -o -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -path ./.venv -prune -false -o -name '*.pyc' -exec rm -f {} +
	find . -path ./.venv -prune -false -o -name '*.pyo' -exec rm -f {} +
	find . -path ./.venv -prune -false -o -name '*~' -exec rm -f {} +
	find . -path ./.venv -prune -false -o -name '__pycache__' -exec rm -fr {} +

lock: ## Lock dependencies
	poetry lock

format: ## format code ruff formatter and docformatter
	poetry run docformatter -r paper_experiments --in-place --wrap-summaries=88 --wrap-descriptions=87 --recursive
	poetry run docformatter -r fedpydeseq2_graphs --in-place --wrap-summaries=88 --wrap-descriptions=87 --recursive
	poetry run ruff format paper_experiments fedpydeseq2_graphs

lint: ## Check style with ruff linter
	poetry run pydocstyle paper_experiments fedpydeseq2_graphs
	poetry run ruff check --fix paper_experiments fedpydeseq2_graphs

typing: ## Check static typing with mypy
	poetry run mypy --config-file=pyproject.toml --install-types --non-interactive --show-traceback



pre-commit-checks: ## Run pre-commit checks on all files
	poetry run pre-commit run --hook-stage manual --all-files

lint-all: pre-commit-checks lint ## Run all linting checks.
