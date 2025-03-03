PYTHON=python3.13
SRC=src
FILELEARN=firstLearn.py
PREDICTION=prediction.py
CREATION=creationModel
UTILISATION=utilisationModel

all: run

run:
	$(PYTHON) $(SRC)/$(CREATION)/$(FILELEARN) $(ARGS)

prediction:
	$(PYTHON) $(SRC)/$(UTILISATION)/$(PREDICTION)

open_data:
	open ./data/cifar-10-batches-py/readme.html
