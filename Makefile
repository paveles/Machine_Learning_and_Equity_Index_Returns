.PHONY: clean data lint requirements data train visualize

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = epml


ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	python -m pip install -U pip setuptools wheel
	python -m pip install -r requirements.txt

## Make Dataset
data: requirements
	python src/data.py

## Make Dataset
train: 
	python src/train.py

## Visualize
visualize: 
	python src/visualize.py

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src


## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,python))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=3
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	python -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already intalled.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source 'which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=python"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	python test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Make help."
	@echo "First Run:"
	@echo "- 'make create_environment' to create a new virtual environment. This new environment will be called "epml", an abbreviation for Equity Premium and Machine Learning."
	@echo "- Activate the new environment. In Anaconda, 'conda activate epml'."
	@echo "- Added new packages to 'requirements.txt' if needed."
	@echo "- 'make requirements' to install packages."
	@echo "Analysis:"
	@echo "- Activate the new environment before starting your analysis. In Anaconda, 'conda activate epml'."
	@echo "- 'make data' to prepare the data."
	@echo "- Change settings in 'settings.py' to choose models to be estimated and evaluated (for the first run, one simple model is already chosen)."
	@echo "- 'make train' to train the chosen models (please note that some models take long hours to run)."
	@echo "- 'make visualize' to get prediction accuracy and produce a figure summarizing strategy performance."