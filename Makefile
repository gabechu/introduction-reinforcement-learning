start-shell:
	PYTHONPATH=. poetry shell

unit-test:
	PYTHONPATH=. poetry run pytest -vv ./tests/unit

linting:
	poetry run mypy src tests
	poetry run flake8 src tests
