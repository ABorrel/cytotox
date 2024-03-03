from sklearn import metrics
from numpy import loadtxt, arange
import pandas
from sklearn.preprocessing import StandardScaler
from tensorflow.python.module.module import valid_identifier
sc = StandardScaler()
import numpy as np

def performance(y_real, y_pred, typeModel, th_prob = 0.5, scale_y_pred = 0, p_filout = "", verbose=0):

    # not for use just for test
    #if verbose == 1:
    #    print("yreal => ", y_real)
    #    print("ypred => ", y_pred)

    # here I had a "scale" of the y_pred. If the y_pred is too high I scaled it to the highest value of y_real
    # and if too low I scalled it to the lowest value of y_real
    if scale_y_pred == 1:
        y_real = np.array(y_real)
        y_pred = np.array(y_pred)
        max_y_real = np.max(y_real)
        min_y_real = np.min(y_real)
        y_pred = np.where(y_pred < max_y_real, y_pred, max_y_real)
        y_pred = np.where(y_pred > min_y_real, y_pred, min_y_real)

    if typeModel == "classification":

        # change prob -> value
        # case of 2 values predicted
        try:y_pred = [1. if pred > th_prob else 0. for pred in y_pred]
        except:
            return {} 

        acc = metrics.accuracy_score(y_real, y_pred)
        bacc = metrics.balanced_accuracy_score(y_real, y_pred)
        mcc = metrics.matthews_corrcoef(y_real, y_pred)
        recall = metrics.recall_score(y_real, y_pred)
        roc_auc = metrics.roc_auc_score(y_real, y_pred)
        f1b = metrics.fbeta_score(y_real, y_pred, beta=0.5)

        # specificity & sensitivity 
        tn, fp, fn, tp = metrics.confusion_matrix(y_real, y_pred).ravel()
        specificity = float(tn) / (tn+fp)
        sensitivity = float(tp) / (tp+fn)

        if verbose == 1:
            print("======= PERF ======")
            print("Acc: ", acc)
            print("b-Acc: ", bacc)
            print("Sp: ", specificity)
            print("Se: ", sensitivity)
            print("MCC: ", mcc)
            print("Recall: ", recall)
            print("AUC: ", roc_auc)
            print("fb1: ", f1b)

        return {"Acc": [acc], "b-Acc": [bacc], "MCC": [mcc], "Recall": [recall], "AUC": [roc_auc], "Se": [sensitivity], "Sp": [specificity], "f1b": [f1b]}

    else:
        
        # need to manage the case where y pred has nan
        if np.isnan(y_pred).sum() > 0:
            return {"MAE": [0.0], "R2": [0.0], "EVS": [0.0], "MSE": [9999999], "MAXERR": [9999999], "MSE_log": [9999999] , "MDAE": [9999999], "MTD": [0.0], "MPD":[0.0] , "MGD":[0.0], "NB NAN":[np.isnan(y_pred).sum()]}
        
        MAE = metrics.mean_absolute_error(y_real, y_pred)
        R2 = metrics.r2_score(y_real, y_pred)
        EVS = metrics.explained_variance_score(y_real, y_pred)
        MSE = metrics.mean_squared_error(y_real, y_pred)
        MAXERR = metrics.max_error(y_real, y_pred)
        try:MSE_log = metrics.mean_squared_log_error(y_real, y_pred)
        except: MSE_log = 0.0
        MDAE = metrics.median_absolute_error(y_real, y_pred)
        MTD = metrics.mean_tweedie_deviance(y_real, y_pred)
        try:
            MPD = metrics.mean_poisson_deviance(y_real, y_pred)
            MGD = metrics.mean_gamma_deviance(y_real, y_pred)
        except:
            MPD = 0.0
            MGD = 0.0

        if verbose == 1:
            print("======= Perf ======")
            print("MAE: ", MAE)
            print("R2: ", R2)
            print("Explain Variance score: ", EVS)
            print("MSE: ", MSE)
            print("Max error: ", MAXERR)
            print("MSE log: ", MSE_log)
            print("Median absolute error: ", MDAE)
            print("Mean tweedie deviance: ", MTD)
            print("Mean poisson deviance: ", MPD)
            print("Mean gamma deviance: ", MGD)

        if p_filout != "":
            filout = open(p_filout, "w")
            filout.write(",MAE,R2,Explain Variance score,MSE,Max error,MSE log,Median absolute error,Mean tweedie deviance,Mean poisson deviance,Mean gamma deviance\n")
            filout.write("Pred,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"%(MAE, R2, EVS, MSE, MAXERR, MSE_log, MDAE, MTD, MPD, MGD))
            filout.close()


        return {"MAE": [MAE], "R2": [R2], "EVS": [EVS], "MSE": [MSE], "MAXERR": [MAXERR], "MSE_log": [MSE_log] , "MDAE": [MDAE], "MTD": [MTD], "MPD":[MPD] , "MGD":[MGD], "NB NAN":[0.0]}

def loadSet(p_set, l_col_to_order = [], variableToPredict = "Aff", sep = ","):
    """
    Data in input open first with panda and next convert to numpy format to exit
        - sep = ,
        - aff in last col with the name in input
        - also col selected previously
    """

    d_out = {}

    # open data
    d_out["dataset"] = pandas.read_csv(p_set, sep = sep)
    
    # take affinity col
    if variableToPredict != "":
        l_aff = d_out["dataset"][variableToPredict]
        l_aff = l_aff.to_numpy()
        d_out["aff"] = l_aff
        d_out["dataset"] = d_out["dataset"].drop(columns = [variableToPredict])
    
    # extra ID 
    if "ID" in list(d_out["dataset"].keys()):
        l_id =  list(d_out["dataset"]["ID"])
    elif "CASRN" in list(d_out["dataset"].keys()):
        l_id =  list(d_out["dataset"]["CASRN"])
    else:
        l_id = []
    d_out["id"] = l_id
    d_out["dataset"] = d_out["dataset"].iloc[:, 1:]
    
    # select specific col
    if l_col_to_order != []:
        d_out["dataset"] = d_out["dataset"][l_col_to_order]

    # list of features
    l_features = list(d_out["dataset"].columns)
    d_out["features"] = l_features
    d_out["nb_desc_input"] = len(l_features)

    # format dataset here
    d_out["dataset"] = d_out["dataset"].to_numpy()

    return d_out

def calibrate_prediction(df_pred, max_val_train, min_val_train, max_aff_inactive):
    """
    The goal of this function is to qualibrate the prediction of the model
    Everything that is more that the max value of the training set is set to the inactive value (100*max)
    Everything that is less that the min value of the training set is set to the minumum value of the training set
    Note that the predictied matrix need to be reduced before the function
    """
    # let's produce a copy of the dataframe
    # reduce risk of dataframe not changed
    df_pred_out = df_pred.copy()
    
    
    if max_aff_inactive < 0:
        mask_max_inact = (df_pred_out > (max_val_train+2))#log1 so +10 or +100
        mask_min_inact = (df_pred_out < (min_val_train-1))
        df_pred_out[mask_max_inact] = max_aff_inactive
        df_pred_out[mask_min_inact] = max_aff_inactive

    else:
        mask_max = (df_pred_out > max_val_train)
        mask_min = (df_pred_out < min_val_train)
        df_pred_out[mask_max] = max_aff_inactive
        df_pred_out[mask_min] = min_val_train


    return df_pred_out
