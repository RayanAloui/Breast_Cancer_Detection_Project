PYTHON = python
MAIN = main.py

DATA_TAB = data.csv
DATA_IMG = images.csv
DATA_RF  = data2.csv

.PHONY: help dso1 dso2 dso3 clean

help:
	@echo "Usage:"
	@echo "  make dso1    -> Lancer DSO1 (ML tabulaire)"
	@echo "  make dso2    -> Lancer DSO2 (CNN images)"
	@echo "  make dso3    -> Lancer DSO3 (Random Forest)"
	@echo "  make clean   -> Nettoyage"

dso1:
	$(PYTHON) $(MAIN) --runDSO1 

dso2:
	$(PYTHON) $(MAIN) --runDSO2 

dso3:
	$(PYTHON) $(MAIN) --runDSO3 

clean:
	rm -rf __pycache__ *.pkl *.joblib mlruns
