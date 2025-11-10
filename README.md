# Rent Estimate Project - Gregory Yampolsky

Hi and welcome to my GitHub repository for the Rent Estimate case study! The goal of this project is to build a machine learning model that can predict the rent price of a property based on various features such as location, size, and amenities.

<img align="center" width="250" height="150" src=./Images/rent-estimate.png> 

**Email**: [gregyampolsky@gmail.com](gregyampolsky@gmail.com)

## Table of Contents
- [Folder Structure](#folder-structure)
- [Project Objective](#Project Objective)
- [Quick Start](#quick-start)  

## folder structure
```
├── README.md
├── requirements.txt
├── .circleci
│   └── config.yml
├── data
│   ├── NTAD_National_Transit_Map_Stops.csv
│   └── State_zhvi_bdrmcnt_1_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv
│   └── TestSet.csv
│   └── TrainSet.csv
│   └── Zip_zhvi_brmcnt_1_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv
├── docs
│   ├── DataScienceProject.txt
│   └── writeup.docx
├── notebooks
│   └── EDA.ipynb
│   └── model_evals.ipynb
├── src
│   └── tests
│       ├── test_data_processing.py
│   ├── preprocessing.py
│   └──modeling.py
│   └── parameter_store.py
├── models
│   ├──any models will be saved here.joblib
│   └── model_results.csv
├── requirements.txt
└── Images
    └── rent-estimate.png
``` 
All notebooks can be found in the **[notebooks folder](./notebooks)**.  It consists of one **[exploratory data analysis workbook](./notebooks/EDA.ipynb)** showing my work on exploring different feature engineerings and **[one model evaluation and fitting notebook](./notebooks/model_evals.ipynb)** where I compared several models, fine tuned hyperparameters and evaluated residuals.

The write up I did is in the docs folder.
## Project Object

The company wants a Python model that estimates market rent for single-family homes across the U.S. using basic property info and location. They gave two files: one to train on (with past rented prices over the last two years) and one to test on. 
Train a model on the provided historical data (TrainingSet.csv) using:
Latitude, Longitude
Bedrooms, Bathrooms
Square Feet
Year Built
Plus the target: Close Price (i.e. the rent it actually leased for)
Use that model to predict a “Market Rent” for new properties (like the ones in TestSet.csv or any similar dataframe).
Deliver a function that takes a pandas DataFrame with those columns and returns the same DF with an extra Market Rent column.
Explain your thinking — feature choices, modeling approach, and how you would improve it with more data (neighborhood features, seasonality, amenities, etc.).

## Quick Start
```
bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
'''