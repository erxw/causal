import pandas as pd
import numpy as np
import os

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

class PropensityScore():
    def __init__(self, treatment, controls, covariates = [], exact = [], repeat = True):
                
        if not covariates:
            covariates = treatment.columns.values
        
        def numeric_dropnan(df):
            df[covariates] = df[covariates].apply(pd.to_numeric, axis = 0, errors = 'coerce')
            return df.dropna(subset = covariates)
        
        self.treatment = numeric_dropnan(treatment)
        self.controls = numeric_dropnan(controls)
        self.dataframe = pd.concat([self.treatment, self.controls])
        self.x = self.dataframe[covariates].values
        self.y = np.append(np.ones(self.treatment.shape[0]), np.zeros(self.controls.shape[0]))
            
        self.repeat = repeat
        self.exact = exact
        self.covariates = covariates
        
    def norm(self, f = None):
        if f:
            self.x = f(self.x)
        else:
            self.x -= self.x.mean(axis = 0)
            self.x /= self.x.std(axis = 0)
        
    def fit(self):
        self.model = LogisticRegression()
        self.model.fit(self.x, self.y)
        self.scores = self.model.predict_proba(self.x)[:, 0]
        self.dataframe['score'] = self.scores
        
    def transform(self):
        assert 'score' in self.dataframe, 'fit first please'
        
        treat = self.dataframe.loc[self.y.astype('bool')]
        ctrl = self.dataframe.loc[~self.y.astype('bool')]
        self.control = pd.DataFrame()
        
        drop = []
        
        for i, features in treat.iterrows():
            if self.exact: 
                temp = ctrl[(ctrl[self.exact] == features[self.exact]).all(axis = 1).values]
            else:
                temp = ctrl
            if temp.empty: 
                drop.append(i)
                continue
                
            index = np.abs(features['score'] - temp['score']).idxmin()
            self.control = self.control.append(ctrl.loc[index])
            
            if not self.repeat:
                ctrl.drop(index, inplace = True)
                
        print(f'{len(drop)} entries dropped.')
        self.treatment.drop(labels = drop, inplace = True)
        
        return self.treatment, self.control
                
    def fit_transform(self):
        self.fit()
        return self.transform()