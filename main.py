from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

from utils.metrics import *
from utils.dataloader import dataloader

from model.TF2.AdversarialDebiasing import AdversarialDebiasing

import time


# privileged is male, unprivileged is female
def train_pipeline(data, sens_feature, backend, debiased=True):
    # load data
    data = dataloader()  # else adult
    n_epoch = 100

    # Split data into train and test
    # x_train and x_test are datasets without income and gender
    # y_train and y_test are income and gender
    x_train, x_test, y_train, y_test, numvars, categorical = data

    # Classification = income
    classification = y_train.columns.to_list()
    classification.remove(sens_feature)
    classification = classification[0]

    clf = AdversarialDebiasing([sens_feature], adversary_loss_weight=0.5, num_epochs=n_epoch, batch_size=256,
                               random_state=279, debias=debiased)
    start = time.time()
    # x_train is dataset without gender and income, y_train is gender and income
    clf.fit(x_train, y_train, x_test, y_test)
    end = time.time()
    total = end - start
    if debiased:
        clf_type = 'Adversarial Debiasing'
    else:
        clf_type = 'Biased Classification'
    print(f"{clf_type} with {backend} Backend Training completed in {total} seconds!")

    print("\nTrain Results\n")
    y_pred = clf.predict(x_train)
    acc = accuracy_score(y_train[classification], y_pred)
    #deo = DifferenceEqualOpportunity(y_pred, y_train, sens_feature, classification, 1, 0, [0, 1])
    dao = DifferenceAverageOdds(y_pred, y_train, sens_feature, classification, 1, 0, [0, 1])
    print(f'\nTrain Acc: {acc}, \nDiff. in Average Odds: {dao}')

    start = time.time()
    print("\nTest Results\n")
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test[classification], y_pred)
    # deo = DifferenceEqualOpportunity(y_pred, y_test, sens_feature, classification, 1, 0, [0, 1])
    dao = DifferenceAverageOdds(y_pred, y_test, sens_feature, classification, 1, 0, [0, 1])
    print(f'\nTest Acc: {acc}, \nDiff. in Average Odds: {dao}')
    end = time.time()
    total = end - start
    print(f"{clf_type} with {backend} Backend Inference completed in {total} seconds!")


dataframe = 'adult'  # info[0]
sensitive_feature = 'gender'  # info[1]
Backend = 'TF2'  # info[2]

# classifier
#train_pipeline(
#    data=dataframe,
#    sens_feature=sensitive_feature,
#    backend=Backend,
#    debiased=False
#)

# Classifier with Adversary
train_pipeline(
    data=dataframe,
    sens_feature=sensitive_feature,
    backend=Backend,
    debiased=True
)
