
test:
	cd tests; python -m unittest discover

coverage:
	cd tests; coverage run -m unittest discover
	cd tests; coverage html
	xdg-open tests/htmlcov/index.html

build:
	python setup.py sdist

install:
	python setup.py install
