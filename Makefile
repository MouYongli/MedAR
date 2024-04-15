# Makefile will include make test make clean make build make run 

# specify desired location for adpy python binary 
VENV:= /home/$(USER)/anaconda3/envs/medar
PYTHON:= ${VENV}/bin/python

# clean automatic generated files
clean:
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -rf *.egg-info

activate: 
	conda activate ${VENV}