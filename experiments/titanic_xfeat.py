import os

import pandas as pd
import xfeat
from xfeat import (ArithmeticCombinations, ConcatCombination, CountEncoder,
                   LabelEncoder, SelectNumerical)

from ayniy.preprocessing import xfeat_runner
from ayniy.utils import FeatureStore

categorical_cols = [
    "Sex",
    "SibSp",
    "Parch",
    "Cabin",
    "Embarked"
]

numerical_cols = [
    "Pclass",
    "Age",
    "Fare"
]

target_col = "Survived"
id_col = "PassengerId"
input_dir = "../input/titanic/"
output_dir = "../input/features/"


def load_titanic() -> pd.DataFrame:
    filepath = "../input/titanic/train_test.ftr"
    if not os.path.exists(filepath):
        # Convert dataset into feather format.
        train = pd.read_csv(input_dir + "train.csv")
        test = pd.read_csv(input_dir + "test.csv")

        xfeat.utils.compress_df(pd.concat([train, test], sort=False)).reset_index(
            drop=True
        ).to_feather(filepath)

    return pd.read_feather(filepath)


if __name__ == "__main__":

    train = load_titanic()

    # Numerical + Target
    xfeat_runner(
        pipelines=[SelectNumerical(use_cols=numerical_cols + [target_col])],
        input_df=train[numerical_cols + [target_col]],
        output_filename=output_dir + "SelectNumerical.ftr",
    )

    # ArithmeticCombinations
    xfeat_runner(
        pipelines=[ArithmeticCombinations(drop_origin=True, operator="+", r=2,)],
        input_df=train[numerical_cols],
        output_filename=output_dir + "ArithmeticCombinations.ftr",
    )

    # LabelEncoder + CountEncoder
    xfeat_runner(
        pipelines=[LabelEncoder(output_suffix=""), CountEncoder()],
        input_df=train[categorical_cols],
        output_filename=output_dir + "CountEncoder.ftr",
    )

    # ConcatCombination r=2 & CountEncoder
    xfeat_runner(
        pipelines=[
            LabelEncoder(output_suffix=""),
            ConcatCombination(drop_origin=True, r=2),
            CountEncoder(),
        ],
        input_df=train[categorical_cols],
        output_filename=output_dir + "ConcatCombinationCountEncoder.ftr",
    )

    features = FeatureStore(
        feature_names=[
            output_dir + "SelectNumerical.ftr",
            output_dir + "ArithmeticCombinations.ftr",
            output_dir + "CountEncoder.ftr",
            output_dir + "ConcatCombinationCountEncoder.ftr",
        ],
        target_col=target_col
    )

    X_train = features.X_train
    y_train = features.y_train
    X_test = features.X_test

    print(X_train.head())
    print(X_train.shape)
