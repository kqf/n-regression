.PHONY: eval search

# Yes I know it's silly, 
# but I didn't have time to setup virtualenv

PYTHON = python2

eval: model/*.py __main__.py
	$(PYTHON) .


search: test/test_search_parameters.py
	$(PYTHON) -m unittest test.test_search_parameters
