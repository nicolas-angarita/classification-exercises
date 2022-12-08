import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def prep_iris(iris_data):
    
    iris_data = iris_data.drop(columns = ['species_id', 'measurement_id', 'Unnamed: 0'])
    
    iris_data = iris_data.rename(columns={'species_name':'species'})

    dummies = pd.get_dummies(iris_data['species'], drop_first=True)
    
    iris_data = pd.concat([iris_data, dummies], axis = 1)
    
    return iris_data

def prep_titantic(titanic_df):
    
    titanic_df = titanic_df.drop(columns=['passenger_id','embarked','class', 'age','deck'])
    
    titanic_df['embark_town'].fillna('Southampton', inplace = True)
    
    dummies = pd.get_dummies(titanic_df[['sex', 'embark_town']], drop_first = True)
    
    titanic_df = pd.concat([titanic_df, dummies], axis = 1)
    
    return titanic_df

def prep_telco(telco_df):
    
    telco_df = telco_df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'Unnamed: 0'])
    
    telco_df['gender_encoded'] = telco_df.gender.map({'Female': 1, 'Male': 0})

    telco_df['partner_encoded'] = telco_df.partner.map({'Yes': 1, 'No': 0})

    telco_df['dependents_encoded'] = telco_df.dependents.map({'Yes': 1, 'No': 0})

    telco_df['phone_service_encoded'] = telco_df.phone_service.map({'Yes': 1, 'No': 0})

    telco_df['paperless_billing_encoded'] = telco_df.paperless_billing.map({'Yes': 1, 'No': 0})

    telco_df['churn_encoded'] = telco_df.churn.map({'Yes': 1, 'No': 0})
    
    
    dummy_df = pd.get_dummies(telco_df[['multiple_lines', 'online_security', 
                              'online_backup', 'device_protection', 
                              'tech_support', 'streaming_tv',   'streaming_movies',
                              'contract_type', 'internet_service_type', 'payment_type']],
                              drop_first=True)
    
    
    telco_df = pd.concat([telco_df, dummy_df], axis = 1)
    
    return telco_df    




def train_val_test(df,col):
    seed = 42
    
    train,val_test = train_test_split(df, train_size = 0.7, random_state = seed, stratify = df[col])
    
    validate, test = train_test_split(val_test, train_size = 0.5, random_state = seed, stratify = val_test[col])
    
    return train, validate, test