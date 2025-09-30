# Makefile will include make test make clean make build make run 

# specify desired location for adpy python binary 
VENV:= /home/$(USER)/anaconda3/envs/deephallu
PYTHON:= ${VENV}/bin/python

# clean automatic generated files
clean:
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . | grep -E ".egg-info" | xargs rm -rf
	find . | grep -E ".coverage|coverage.xml" | xargs rm -rf
	find . | grep -E ".mypy_cache" | xargs rm -rf
	find . | grep -E ".ruff_cache" | xargs rm -rf
	find . | grep -E "htmlcov" | xargs rm -rf
	find . | grep -E ".DS_Store" | xargs rm -rf
	find . | grep -E "dist|build" | xargs rm -rf
	rm -rf .tox/
	rm -rf .env/

activate: 
	conda activate ${VENV}