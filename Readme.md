Machine Learning and Equity Index Returns
==============================
## A time-series predictive framework that features:
- Application of advanced machine learning algorithms.
- Usage of scikit-learn pipelines that simplify automation of the analysis.
- New scikit-learn transformers.
- New scikit-learn time-series cross-validation methods (one step forward expanding window nested cross-validation).
- Domain-tailored statistical tests on the significance of improvement in prediction accuracy.
- Jupyter notebooks and a [report](/reports/Results.ipynb) that explain and visualize obtained findings.
- Clear project structure with a makefile based on a data science template.
- Dockerfile to run the project on any platform or in the cloud



## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for using this project.
    |
    ├── out
    │   ├── expanding      <- Output results for the models with one step ahead expanding window nested cross-validation.
    │   └── rolling        <- Output results for the models with one step ahead fixed rolling window nested cross-validation.
    |    
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module.
    │   │
    │   ├── data.py        <- Scripts to generate data.
    │   │   
    │   ├── train.py       <- Script to train models and then use trained models to make
    │   │                     predictions.   
    │   ├── visualize.py   <- Scripts to create exploratory and results-oriented visualization.
    │   │
    │   ├── model_configs.py          <- Configurations, GridSearch methods, cross-validation methods of the models.
    │   ├── settings.py               <- Global settings and variables + loads `model_configs`. 
    │   ├── transform_cv.py           <- Transformation and cross-validation methods used in the analysis.
    │   └── walkforward_functions.py  <- Main functions used to estimate and evaluate trained models. 
    │       
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org

## Workflow
- Setup:
  -  Clone or download this repository.
  -  Install Make. See the [website](https://www.gnu.org/software/make/).
  -  You can access help for the Makefile by typing `make` in the project folder.
  -  `make create_environment` to create a new virtual environment. This new environment will be called "epml", an abbreviation for Equity Premium and Machine Learning.
  -  Activate the new environment. In Anaconda, `conda activate epml`.
  -  Added new packages to `requirements.txt` if needed.
  -  `make requirements` to install packages.
-  Analysis:
   - Activate the new environment before starting your analysis. In Anaconda, `conda activate epml`.
   - `make data` to prepare the data.
   - Change settings in `settings.py` to choose models to be estimated and evaluated (for the first run, one simple model is already chosen).
   - `make train` to train the chosen models (please note that some models take long hours to run).
   - `make visualize` to get prediction accuracy and produce a figure summarizing strategy performance.

## Setup Details for Windows
There are some challenges to install Make on Windows. These steps might help:
- To install Make on Windows use a prebuilt [Installer](https://github.com/swcarpentry/windows-installer/releases/tag/v0.3) from Software Carpentry. Please add the Make directory to the the system environment variable PATH, e.g. `C:\Users\Admin\.swc\lib\make`.
- You can test Make by accessing help for the Makefile by typing `make` in the project folder.
- In case Make still does not work, please install [MSYS2](https://www.msys2.org/) and add its `bin` directory to the the system environment variable PATH, e.g. `C:\msys64\usr\bin`.
- Try restarting your computer.

## Alternatively, use Docker
To run this project via Docker:
- Install [Docker](https://www.docker.com/) on your computer.
- Build Docker image in the project folder containing the Dockerfile `docker image build --tag=epml .`.
- To run container with the predefined code in `main.py`, type `docker run -v Absolute/Path/At/Localhost/out:/code/out/ epml:latest`.  The `-v Absolute/Path/On/Localhost/out:/code/out/` part ensures that output produced in the Docker container is saved in the `/out/` folder of the project at your computer (please create this folder beforehand).
- Alternatively, run the container iteratively (with the ability to modify files and run make commands described in the workflow above inside the container) `docker container run -v Absolute/Path/At/Localhost/out:/code/out/ -it epml:latest bash`.
--------
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

