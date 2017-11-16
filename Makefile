.PHONY: eval search

eval: model/*.py __main__.py
	python .


search: test/test_search_parameters.py
	$(PYTHON) -m unittest test.test_search_parameters
