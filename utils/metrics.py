from sklearn.metrics import confusion_matrix
import numpy as np
import scipy as scipy
import math


# outcome is the name of the variable we are interested in predicting, so I guess income
def DifferenceEqualOpportunity(y_pred,y_real,SensitiveCat, outcome, privileged, unprivileged, labels):
    '''
    ABS Difference in True positive Rate between the two groups
    :param y_pred: prediction
    :param y_real: real label
    :param SensitiveCat: Sensitive feature name
    :param outcome: Outcome feature name
    :param privileged: value of the privileged group
    :param unprivileged: value of the unprivileged group
    :param labels: both priv-unpriv value for CFmatrix
    :return:
    '''
    y_priv = y_pred[y_real[SensitiveCat]==privileged]
    y_real_priv = y_real[y_real[SensitiveCat]==privileged]
    y_unpriv = y_pred[y_real[SensitiveCat]==unprivileged]
    y_real_unpriv = y_real[y_real[SensitiveCat]==unprivileged]
    TN_priv, FP_priv, FN_priv, TP_priv = confusion_matrix(y_real_priv[outcome], y_priv, labels=labels).ravel()
    TN_unpriv, FP_unpriv, FN_unpriv, TP_unpriv = confusion_matrix(y_real_unpriv[outcome], y_unpriv, labels=labels).ravel()

    return abs(TP_unpriv/(TP_unpriv+FN_unpriv) - TP_priv/(TP_priv+FN_priv))


def DifferenceAverageOdds(y_pred, y_real, sensitivecat, outcome, privileged, unprivileged, labels):
    """
    Mean ABS difference in True positive rate and False positive rate of the two groups
    :param y_pred:
    :param y_real:
    :param sensitivecat:
    :param outcome:
    :param privileged:
    :param unprivileged:
    :param labels:
    :return:
    """

    # in our code, unprivileged = females, privileged = males
    y_priv = y_pred[y_real[sensitivecat] == privileged]
    y_real_priv = y_real[y_real[sensitivecat] == privileged]
    y_unpriv = y_pred[y_real[sensitivecat] == unprivileged]
    y_real_unpriv = y_real[y_real[sensitivecat] == unprivileged]
    TN_unpriv, FP_unpriv, FN_unpriv, TP_unpriv = confusion_matrix(y_real_unpriv[outcome], y_unpriv,
                                                                  labels=labels).ravel()
    TN_priv, FP_priv, FN_priv, TP_priv = confusion_matrix(y_real_priv[outcome], y_priv,  labels=labels).ravel()
    female_confusionmatrix = np.array([[TN_unpriv, FP_unpriv], [FN_unpriv, TP_unpriv]])
    male_confusionmatrix = np.array([[TN_priv, FP_priv], [FN_priv, TP_priv]])

    print(female_confusionmatrix)
    print(male_confusionmatrix)
    print('FPR female: ' + str(FP_unpriv/(FP_unpriv+TN_unpriv)) + '(0.0647) Debiased paper')
    print('FPR male: ' + str(FP_priv/(FP_priv+TN_priv)) + '(0.0701) Debiased paper')
    print('FNR female: ' + str(FN_unpriv/(FN_unpriv+TP_unpriv)) + '(0.04458) Debiased paper')
    print('FNR male: ' + str(FN_priv/(FN_priv+TP_priv)) + '(0.4349) Debiased paper')

    return 0.5*(abs(FP_unpriv/(FP_unpriv+TN_unpriv)-FP_priv/(FP_priv+TN_priv))+abs(TP_unpriv/(TP_unpriv+FN_unpriv)-
                                                                                   TP_priv/(TP_priv+FN_priv)))