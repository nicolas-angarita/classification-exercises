import pandas as pd
import numpy as np
import seaborn as sns
import os
from env import get_connection


def get_titanic_data():

    if os.path.isfile('titanic.csv'):
        
        return pd.read_csv('titanic.csv')
    
    else:
       
        url = get_connection('titanic_db')
        
        query = '''
        SELECT *
        FROM passengers 
        '''

        df = pd.read_sql(query, url)
        
        df.to_csv('titanic.csv', index = False)

        return df  


def get_iris_data():

    if os.path.isfile('iris_df.csv'):
        
        return pd.read_csv('iris_df.csv')
    
    else:
       
        url = get_connection('iris_db')
        
        query = '''
        SELECT *
        FROM species
        JOIN measurements
        USING(species_id); 
        '''

        df = pd.read_sql(query, url)
        
        df.to_csv('iris_df.csv', index = False)

        return df   


def get_telco_data():

    if os.path.isfile('telco.csv'):
        
        return pd.read_csv('telco.csv')
    
    else:
       
        url = get_connection('telco_churn')
        
        query = '''
        SELECT *
        FROM customers
        JOIN contract_types USING(contract_type_id)
        JOIN internet_service_types USING(internet_service_type_id)
        JOIN payment_types USING(payment_type_id);
        '''

        df = pd.read_sql(query, url)
        
        df.to_csv('telco.csv', index = False)

        return df                