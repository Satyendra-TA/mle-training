# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data.

The following techniques have been used:

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## To excute the script
Run these before running the Python script:
- conda env create -f env.yml
- conda activate mle-dev
- pip install -e .
- python src/house_price_predictor/ingest_data.py /mnt/c/Users/satyendra.mishra/mle-training/artifacts
- python src/house_price_predictor/train.py data/processed/train.csv artifacts/
- python src/house_price_predictor/score.py artifacts/model.pkl data/processed/test.csv

- Please make sure the order of exection remains the same to ensure sucessfull execution
