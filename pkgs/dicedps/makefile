include ../../makefile_common

.PHONY: forcings

forcings: dicedps/data/forcings.csv

dicedps/data/forcings.csv: dicedps/extrap_climate_forcings.py
	$(PYTHON) -m dicedps.extrap_climate_forcings

