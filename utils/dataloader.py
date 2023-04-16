import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def dataloader():
    data = pd.read_csv("data/adult_data.csv", header=None)
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week',
                    'native-country', 'income']
    data.columns = column_names
    data_test = pd.read_csv("data/adult_test.csv", header=None)
    data_test.columns = column_names

    data = data.dropna()
    data_test = data_test.dropna()

    # Bucketize age, assign a bin number to each age
    data['age'] = np.digitize(data['age'], bins=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    data_test['age'] = np.digitize(data_test['age'], bins=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    # Drop fnlwgt column
    data = data.drop(columns=['fnlwgt'])
    data_test = data_test.drop(columns=['fnlwgt'])

    # Replace income with booleans, Space before 50k is crucial
    data['income'] = (data['income'] == ' >50K').astype(int)
    data_test['income'] = (data_test['income'] == ' >50K.').astype(int)

    # numerical columns in dataset
    numvars = ['education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'age']

    # makes target and target_test, which are income for bias and gender for debias
    sensitive_features = ['gender']
    target_train = data[["income", "gender"]].copy()
    target_train.replace([' Male', ' Female'], [1, 0], inplace=True)

    target_test = data_test[["income", "gender"]].copy()
    target_test.replace([' Male', ' Female'], [1, 0], inplace=True)

    # makes dataset and dataset_test, which is the original data set without gender and income, split in a training and
    # test set
    data = data.drop(columns=sensitive_features)
    data_test = data_test.drop(columns=sensitive_features)

    dataset_train = data.drop("income", axis=1)
    dataset_test = data_test.drop("income", axis=1)

    # categorical column in dataset
    categorical = dataset_train.columns.difference(numvars)
    preprocessor = make_column_transformer(
        (StandardScaler(), numvars),
        (OneHotEncoder(handle_unknown='ignore'), categorical)
    )

    dataset_train = preprocessor.fit_transform(dataset_train)
    dataset_test = preprocessor.transform(dataset_test)

    return dataset_train, dataset_test, target_train, target_test, numvars, categorical

