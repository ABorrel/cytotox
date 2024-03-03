import pathManager
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from skopt.space import Real, Integer, Categorical
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from skopt import gp_minimize
from skopt.utils import use_named_args
from sklearn.model_selection import cross_val_score
from skopt.plots import plot_convergence
import ML_toolbox

class runRegModeling:

    def __init__(self, p_dataset, p_dir_out):

        self.p_dataset = p_dataset
        self.p_dir_out = p_dir_out

        # create folder by dataset
        self.p_dir_by_dataset = pathManager.create_folder(p_dir_out + self.p_dataset.split("/")[-1][:-4] + "/")

        pass


    def remove_correlated_desc(self, df_in, threshold=0.8):

        # remove desc with variance = 0
        var = df_in.agg("var", numeric_only=True)
        df_in = df_in.drop(columns=var[var.between(0, 0.0001)].index)

        # reduce the number of descriptor with a correlation > 0.8
        df_corr = df_in.corr()
        # Create a mask for values above 90% 
        # But also below 100% since it variables correlated with the same one
        mask = (df_corr.corr() > threshold) & (df_corr.corr() < 1.0)
        high_corr = df_corr[mask]

        # Create a new column mask using any() and ~
        col_to_filter_out = ~high_corr[mask].any()

        # Apply new mask
        df_desc_coor = df_in[high_corr.columns[col_to_filter_out]]

        return df_desc_coor


    def format_dataset_for_modeling(self):
            
        df_dataset = pd.read_csv(self.p_dataset)
        df_dataset.index = df_dataset['DTXSID']
        df_dataset = df_dataset.drop(columns=['DTXSID'])

        # remove correlation
        l_AC50 = df_dataset['AC50']
        df_dataset = self.remove_correlated_desc(df_dataset, 0.95)

        # case we remove the AC50 column
        if not "AC50" in df_dataset.columns:
            df_dataset["AC50"] = l_AC50

        # log transform the AC50
        df_dataset["Log AC50"] = np.log10(df_dataset['AC50'])
        #df_dataset["Log AC50"] = df_dataset['AC50']
        max_ac50 = df_dataset["Log AC50"].max()
        self.max_ac50 = max_ac50
        self.min_ac50 = df_dataset["Log AC50"].min()
        self.inact_val =  10*self.max_ac50
        self.inact_val =  -10*self.min_ac50


        df_dataset["Class"] = 1
        df_dataset.loc[df_dataset["AC50"].isnull(), "Class"] = 0

        df_dataset['Log AC50'] = df_dataset['Log AC50'].replace(np.nan, -max_ac50)
        df_dataset = df_dataset.drop(columns=['AC50'])

        # write
        df_dataset.to_csv(self.p_dir_by_dataset + "formated.csv", index=False)


        # split train / test
        df_train, df_test = train_test_split(df_dataset, test_size=0.25)
        df_train.to_csv(self.p_dir_by_dataset + "train.csv", index=False)
        df_test.to_csv(self.p_dir_by_dataset + "test.csv", index=False)

        self.df_train = df_train
        self.df_test  = df_test


        # format for modeling
        #self.Y_train = self.df_train['Log AC50']
        #self.X_train = self.df_train.drop(columns=['Log AC50'])

        # remove correlated desc
        #self.X_train = self.remove_correlated_desc(self.X_train, 0.9)


        self.Y_test = self.df_test["Log AC50"]
        self.X_test = self.df_test.drop(columns=['Log AC50'])
        self.X_test = self.X_test.drop(columns=['Class'])



    def run_undersampling(self, run=10, type_model="Xboost", ratio_inact=0.5):

        nb_act = len(self.df_train[self.df_train["Class"] == 1])
        df_act = self.df_train[self.df_train["Class"] == 1]
        df_inact = self.df_train[self.df_train["Class"] == 0]

        nb_inact = nb_act * (ratio_inact + 0.5)

        i = 0
        l_model = []
        while i < run:
            df_inact_sampled = df_inact.sample(n=nb_inact)
            df_train = pd.concat([df_act, df_inact_sampled], axis=0)

            self.X_train = df_train.drop(columns=['Log AC50', 'Class'])
            self.Y_train = df_train['Log AC50']

            # run the model
            model = self.run_Xboost()
            l_model.append(model)
            i = i + 1


        df_pred_test = pd.DataFrame()
        for model in l_model:
            y_test_pred = model.predict(self.X_test)
            df_pred_test = pd.concat([df_pred_test, pd.DataFrame(y_test_pred)], axis=1)


        df_pred_test['mean'] = df_pred_test.mean(axis=1)
        df_pred_test.to_csv(self.p_dir_by_dataset + "test_undersampling_pred.csv", index=False)

        y_test_pred = df_pred_test['mean']
        y_test_pred_recalibrated = ML_toolbox.calibrate_prediction(y_test_pred, self.max_ac50, self.min_ac50, self.inact_val)
        df_pred_test = ML_toolbox.performance(self.Y_test, y_test_pred_recalibrated, "regression", p_filout=self.p_dir_by_dataset + "test_undersampling_performance.csv")



    def run_Xboost(self):

        model = XGBRegressor()
        space_XGBRegressor = [
            Real(1e-3, 10.0, name = "reg_lambda"),
            Real(1e-3, 10.0, name ="reg_alpha"),
            Real(0.1, 0.9, name ="colsample_bytree"),
            Real(0.1, 0.9, name ="colsample_bynode"),
            Real(0.1, 0.9, name ="colsample_bylevel"),
            Real(0.1, 0.9, name ="subsample"),
            Integer(2, 30, name ="max_depth"),
            Integer(10, 200, name ="n_estimators")
        ]

        #optimize the modeling
        @use_named_args(space_XGBRegressor)
        def objective(**params):
            model.set_params(**params)

            model.fit(self.X_train, self.Y_train)
            y_train_pred = model.predict(self.X_train)
            y_train_pred_recalibrated = ML_toolbox.calibrate_prediction(y_train_pred, self.max_ac50, self.min_ac50, self.inact_val)
            df_pred_train = ML_toolbox.performance(self.Y_train, y_train_pred_recalibrated, "regression")

            #print(df_pred_train)

            #retry a CV 5 fold
            #l_pref = cross_val_score(model, self.X_train, self.Y_train, cv=10, scoring='r2')
            #print(l_pref)


            #return -np.mean(l_pref)

            return -df_pred_train["R2"][0]

        res_gp = gp_minimize(objective, space_XGBRegressor, n_calls=100, random_state=0)

        print("Best score=%.4f" % res_gp.fun)
        print(res_gp.x)


        #plot_convergence(res_gp)
        final_model = XGBRegressor(reg_lambda=res_gp.x[0], reg_alpha=res_gp.x[1], colsample_bytree=res_gp.x[2], colsample_bynode=res_gp.x[3], colsample_bylevel=res_gp.x[4], subsample=res_gp.x[5], max_depth=res_gp.x[6], n_estimators=res_gp.x[7])
        

        # apply on the train
        final_model.fit(self.X_train, self.Y_train)
        y_train_pred = final_model.predict(self.X_train)

        return final_model


        #apply on test
        y_test_pred = final_model.predict(self.X_test[self.X_train.columns])


        y_train_pred_recalibrated = ML_toolbox.calibrate_prediction(y_train_pred, self.max_ac50, self.min_ac50, self.inact_val)
        y_test_pred_recalibrated = ML_toolbox.calibrate_prediction(y_test_pred, self.max_ac50, self.min_ac50, self.inact_val)


        df_pred_train = ML_toolbox.performance(self.Y_train, y_train_pred_recalibrated, "regression", p_filout=self.p_dir_by_dataset + "train_pred.csv")
        df_pred_test = ML_toolbox.performance(self.Y_test, y_test_pred_recalibrated, "regression", p_filout=self.p_dir_by_dataset + "test_pred.csv")

        print(df_pred_test)

        return final_model


        # need to add the CV


