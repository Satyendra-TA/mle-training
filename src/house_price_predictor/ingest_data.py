import os
import tarfile
from argparse import ArgumentParser

import pandas as pd
from six.moves import urllib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
PROJECT_ROOT = "/mnt/c/Users/satyendra.mishra/Work/mle-training/"
HOUSING_PATH = os.path.join(PROJECT_ROOT, "data", "raw")


def fetch_housing_data(
    housing_url: str = HOUSING_URL, housing_path: str = HOUSING_PATH
) -> None:
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path: str = HOUSING_PATH) -> pd.DataFrame:
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def engineer_features(housing: pd.DataFrame) -> pd.DataFrame:
    housing["rooms_per_household"] = (
        housing["total_rooms"] / housing["households"]
    )
    housing["bedrooms_per_room"] = (
        housing["total_bedrooms"] / housing["total_rooms"]
    )
    housing["population_per_household"] = (
        housing["population"] / housing["households"]
    )
    return housing


if __name__ == "__main__":
    if not os.path.exists(os.path.join(HOUSING_PATH, "housing.csv")):
        fetch_housing_data()

    desc = """Fetches data, performs transformations,
              splits into train and validation sets
              and saves the splits in specified directory"""

    DEFAULT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

    parser = ArgumentParser(prog="ingest-data", description=desc)
    parser.add_argument("-o", "--output-path", type=str, default=DEFAULT_DIR)
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        raise "The specified path %s does not exist" % (args.output_path)

    housing_data = load_housing_data()

    train_set, test_set = train_test_split(
        housing_data, test_size=0.2, random_state=42
    )

    housing_df = train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set

    housing_labels = train_set["median_house_value"].copy()

    housing_num = housing_df.drop("ocean_proximity", axis=1)

    imputer = SimpleImputer(strategy="median")
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(
        X, columns=housing_num.columns, index=housing_df.index
    )

    housing_tr = engineer_features(housing_tr)

    housing_cat = housing_df[["ocean_proximity"]]
    housing_prepared = housing_tr.join(
        pd.get_dummies(housing_cat, drop_first=True)
    )

    X_test = test_set.drop("median_house_value", axis=1)
    y_test = test_set["median_house_value"].copy()

    X_test_num = X_test.drop("ocean_proximity", axis=1)
    X_test_prepared = imputer.transform(X_test_num)
    X_test_prepared = pd.DataFrame(
        X_test_prepared, columns=X_test_num.columns, index=X_test.index
    )

    X_test_prepared = engineer_features(X_test_prepared)

    X_test_cat = X_test[["ocean_proximity"]]
    X_test_prepared = X_test_prepared.join(
        pd.get_dummies(X_test_cat, drop_first=True)
    )

    housing_tr["median_house_value"] = housing_labels.to_numpy()
    X_test_prepared["median_house_value"] = y_test.to_numpy()

    housing_tr.to_csv(os.path.join(args.output_path, "train.csv"), index=False)
    X_test_prepared.to_csv(
        os.path.join(args.output_path, "test.csv"), index=False
    )
