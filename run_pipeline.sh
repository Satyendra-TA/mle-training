#!/bin/bash

# data ingestion
python src/house_price_predictor/ingest_data.py -o "data/processed"

# model training
python src/house_price_predictor/train.py -i "data/processed/train.csv" -m "artifacts/model.pkl"

# model evaluation
python src/house_price_predictor/score.py -m "artifacts/model.pkl" -d "data/processed/test.csv"