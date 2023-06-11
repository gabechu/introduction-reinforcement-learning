start-shell:
	PYTHONPATH=. poetry shell

unit-test:
	PYTHONPATH=. poetry run pytest -vv ./tests/unit
