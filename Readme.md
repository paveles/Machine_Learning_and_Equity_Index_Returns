Machine Learning and Equity Index Returns
==============================
## A time-series predictive framework that features:
- Usage of **advanced machine learning algorithms** 
- Usage of **scikit-learn pipelines** that simplify automation of the analysis
- New **scikit-learn transformers**
- New **scikit-learn time-series cross-validation methods** (one step forward expanding window nested cross-validation)
- **Domain-tailored statistical tests** on significance in  prediction improvement
- **Jupyter** notebooks and [report](/reports/Results.ipynb) that explain and visualize obtained findings
- Clear project structure with a **makefile** based on data science template



Project Organization
------------

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
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data.py        <- Scripts to generate data
    │   │   
    │   ├── train.py       <- Script to train models and then use trained models to make
    │   │                     predictions   
    │   ├── visualize.py   <- Scripts to create exploratory and results-oriented visualizations
    │   │
    │   ├── model_configs.py          <- Configurations, GridSearch methods, cross-validation methods of the models
    │   ├── settings.py               <- Global settings and variables + loads `model_configs` 
    │   ├── transform_cv.py           <- Transformation and cross-validation methods used in the analysis
    │   └── walkforward_functions.py  <- Main functions used to estimate and evaluate trained models 
    │       
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

