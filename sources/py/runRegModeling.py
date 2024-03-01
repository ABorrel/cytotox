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


class runRegModeling:

    def __init__(self, p_dataset, p_dir_out):

        self.p_dataset = p_dataset
        self.p_dir_out = p_dir_out

        # create folder by dataset
        self.p_dir_by_dataset = pathManager.create_folder(p_dir_out + self.p_dataset.split("/")[-1][:-4] + "/")

        pass


    def format_dataset_for_modeling(self):
            
        df_dataset = pd.read_csv(self.p_dataset)
        df_dataset.index = df_dataset['DTXSID']
        df_dataset = df_dataset.drop(columns=['DTXSID'])

        # log transform the AC50
        df_dataset["Log AC50"] = np.log10(df_dataset['AC50'])
        max_ac50 = df_dataset["Log AC50"].max()
        self.max_ac50 = max_ac50

        df_dataset['Log AC50'] = df_dataset['Log AC50'].replace(np.nan, max_ac50*10)
        df_dataset = df_dataset.drop(columns=['AC50'])

        # write
        df_dataset.to_csv(self.p_dir_by_dataset + "formated.csv", index=False)


        # split train / test
        df_train, df_test = train_test_split(df_dataset, test_size=0.2)
        df_train.to_csv(self.p_dir_by_dataset + "train.csv", index=False)
        df_test.to_csv(self.p_dir_by_dataset + "test.csv", index=False)

        self.df_train = df_train
        self.df_test  = df_test



    def run_Xboost(self):

        model = XGBRegressor()

        space_XGBRegressor = [
            Real(1e-3, 10.0, name = "reg_lambda"),
            Real(1e-3, 10.0, name ="reg_alpha"),
            Real(0.1, 0.9, name ="colsample_bytree"),
            Real(0.1, 0.9, name ="colsample_bynode"),
            Real(0.1, 0.9, name ="colsample_bylevel"),
            Real(0.1, 0.9, name ="subsample"),
            Integer(2, 9, name ="max_depth"),
            Integer(10, 100, name ="n_estimators")
        ]


        print(self.df_train)
        X = self.df_train.drop(columns=['Log AC50'])
        Y = self.df_train['Log AC50']

        #optimize the modeling
        @use_named_args(space_XGBRegressor)
        def objective(**params):
            model.set_params(**params)

            return -np.mean(cross_val_score(model, X, Y, cv=5, n_jobs=-1,
                                            scoring="neg_mean_absolute_error"))

        res_gp = gp_minimize(objective, space_XGBRegressor, n_calls=10, random_state=0)

        "Best score=%.4f" % res_gp.fun


        sss
        # define model evaluation method
        cv = StratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        # evaluate model
        scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)



        pass