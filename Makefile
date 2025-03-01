PYTHON = python3.13
SRC = src
FILE = main.py
FILELEARN = firstLearn.py

all: run

run: 
	$(PYTHON) $(SRC)/$(FILE)

learn:
	$(PYTHON) $(SRC)/$(FILELEARN)

open_data:
	open ./data/cifar-10-batches-py/readme.html