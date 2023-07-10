import pandas as pd
import plotly.graph_objects as go
def discretize():
    df = pd.read_csv('measures_v2.csv')
    df2=df.loc[df['profile_id']==17]
    df2.drop(['u_q', 'u_d','i_d','i_q','profile_id'], axis=1, inplace=True)
    
    df2.columns=['coolant', 'stator winding', 'stator tooth',
        'motor speed', 'pm', 'stator yoke', 'ambient', 'torque']

    for i in df2.columns:
        df2[i] = pd.qcut(df2[i], q=5,labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
    ordinal_mapping = {'Very Low': 1, 'Low': 2, 'Medium':3, 'High': 4, 'Very High': 5}
    return df2, ordinal_mapping

def column_values_dict_motor2(df):
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


def column_values_dict_motor(df):
    column_values_dict = {}

    for column in df.columns:
        
        column_values_dict[column] = ['missing', 'Very Low', 'Low', 'Medium', 'High', 'Very High']
    return column_values_dict

def column_values_motor(df):
    
    column_values_dict= ['missing', 'Very Low', 'Low', 'Medium', 'High', 'Very High']
    return column_values_dict
# def plot_matplotlib(df_filtered_reset):