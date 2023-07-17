unit-test:
	PYTHONPATH=. poetry run pytest -vv ./tests/unit

lint_typing_checks:
	poetry run ruff src tests
	poetry run pyright src tests
