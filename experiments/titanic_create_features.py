from ayniy.utils import Data, FeatureStore
from ayniy.preprocessing import (count_null,
                                 frequency_encoding,
                                 matrix_factorization,
                                 aggregation,
                                 detect_delete_cols)


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
input_dir = "../input/features/"
output_dir = "../input/"


if __name__ == "__main__":

    features = FeatureStore(
        feature_names=[
            input_dir + "SelectNumerical.ftr",
            input_dir + "ArithmeticCombinations.ftr",
            input_dir + "CountEncoder.ftr",
            input_dir + "ConcatCombinationCountEncoder.ftr",
        ],
        target_col=target_col
    )

    X_train = features.X_train
    y_train = features.y_train
    X_test = features.X_test

    X_train, X_test = count_null(X_train, X_test, encode_col=categorical_cols + numerical_cols)
    X_train, X_test = frequency_encoding(X_train, X_test, encode_col=categorical_cols)
    X_train, X_test = matrix_factorization(
        X_train,
        X_test,
        encode_col=categorical_cols,
        n_components_lda=5,
        n_components_svd=3
    )
    X_train, X_test = aggregation(
        X_train,
        X_test,
        groupby_dict=[
            {
                'key': ['Sex'],
                'var': numerical_cols,
                'agg': ['mean', 'sum', 'median', 'min', 'max', 'var', 'std']
            },
            {
                'key': ['SibSp'],
                'var': numerical_cols,
                'agg': ['mean', 'sum', 'median', 'min', 'max', 'var', 'std']
            },
            {
                'key': ['Parch'],
                'var': numerical_cols,
                'agg': ['mean', 'sum', 'median', 'min', 'max', 'var', 'std']
            },
            {
                'key': ['Cabin'],
                'var': numerical_cols,
                'agg': ['mean', 'sum', 'median', 'min', 'max', 'var', 'std']
            },
            {
                'key': ['Embarked'],
                'var': numerical_cols,
                'agg': ['mean', 'sum', 'median', 'min', 'max', 'var', 'std']
            },
        ],
        nunique_dict=[
            {
                'key': ['Sex'],
                'var': ['SibSp'],
                'agg': ['nunique']
            },
            {
                'key': ['Sex'],
                'var': ['Cabin'],
                'agg': ['nunique']
            },
        ]
    )

    print(X_train.shape, X_test.shape)
    unique_cols, duplicated_cols, high_corr_cols = detect_delete_cols(
        X_train,
        X_test,
        escape_col=categorical_cols,
        threshold=0.99
    )
    X_train.drop(unique_cols + duplicated_cols + high_corr_cols,
                 axis=1,
                 inplace=True)
    X_test.drop(unique_cols + duplicated_cols + high_corr_cols,
                axis=1,
                inplace=True)

    print(X_train.shape, X_test.shape)
    Data.dump(X_train, output_dir + 'X_train_fe000.pkl')
    Data.dump(X_test, output_dir + 'X_test_fe000.pkl')
    Data.dump(y_train, output_dir + 'y_train_fe000.pkl')
