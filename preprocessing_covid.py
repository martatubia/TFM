import pandas as pd
import plotly.graph_objects as go
def discretize():
    df = pd.read_csv('COVID-19_real_continuous.csv')
    df.drop(['Season', 'Transportation_activity','Face_masks', 'Lockdown', 'Work_and_school_activity', 'Majority_COVID_19_variant', 'Leisure_activity'], axis=1, inplace=True)

    df.columns=['P tests', 'Excess mortality', 'Tests across all 4 Pillars',
        'Deaths with COVID', 'Reinfections', 'Patients in MVBs',
        'Hospital admissions', 'Second dose uptake', 'Patients in H',
        'New infections']

    for i in df.columns:
        df[i] = pd.qcut(df[i], q=5,labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
    ordinal_mapping = {'Very Low': 1, 'Low': 2, 'Medium':3, 'High': 4, 'Very High': 5}
    return df, ordinal_mapping

def column_values_dict_covid2(df):
    column_values_dict = {}

    for column in df.columns:
        # Obtiene la lista de valores únicos de la columna
        unique_values_aux = df[column].unique().tolist()

        # Añade 'missing' como el primer elemento de la lista de valores
        if 'missing' in unique_values_aux:
            unique_values_aux.pop(unique_values_aux.index('missing'))
            unique_values = ['missing'] + unique_values_aux
        else:
            unique_values = ['missing'] + unique_values_aux
        
        # Asigna la lista de valores al diccionario con la columna como clave
        column_values_dict[column] = unique_values
    
    return column_values_dict


def column_values_dict_covid(df):
    column_values_dict = {}

    for column in df.columns:
        
        column_values_dict[column] = ['missing', 'Very Low', 'Low', 'Medium', 'High', 'Very High']
    return column_values_dict

def column_values_covid(df):
    
    column_values_dict= ['missing', 'Very Low', 'Low', 'Medium', 'High', 'Very High']
    return column_values_dict
# def plot_matplotlib(df_filtered_reset):
