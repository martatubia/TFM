import pandas as pd
import plotly.graph_objects as go

def discretize():
    df = pd.read_csv('ALARM_DATA.csv')
    df=df.sample(5000)
    df = df.replace({False: 'F'})
    df = df.replace({True:'T'})
    ordinal_mapping = {}
    return df, ordinal_mapping




def column_values_dict2(df):
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


def column_values_dict(df):
    column_values_dict = {'VENTLUNG': ['ZERO', 'LOW','NORMAL', 'HIGH'], 'MINVOLSET': [ 'LOW','NORMAL', 'HIGH'], 'DISCONNECT': [ 'T','F'],
 'VENTMACH': ['ZERO', 'LOW', 'NORMAL', 'HIGH'], 'PRESS': ['ZERO', 'LOW', 'NORMAL', 'HIGH'], 'FIO2': ['LOW', 'NORMAL'],
 'PVSAT': ['LOW','NORMAL', 'HIGH'], 'EXPCO2': ['ZERO', 'LOW','NORMAL', 'HIGH'], 'HREKG': [ 'LOW','NORMAL', 'HIGH'],
 'ERRCAUTER': [ 'T','F'], 'HRBP': [ 'LOW','NORMAL', 'HIGH'], 'ERRLOWOUTPUT': [ 'T','F'], 'MINVOL': ['ZERO', 'LOW','NORMAL', 'HIGH'],
 'KINKEDTUBE': [ 'T','F'], 'STROKEVOLUME': [ 'LOW','NORMAL', 'HIGH'], 'LVFAILURE': [ 'T','F'], 'LVEDVOLUME': [ 'LOW','NORMAL', 'HIGH'],
 'HYPOVOLEMIA': [ 'T','F'], 'HRSAT': ['HIGH', 'LOW', 'NORMAL'], 'HISTORY': [ 'T','F'], 'ANAPHYLAXIS': [ 'T','F'],
 'PAP': [ 'LOW','NORMAL', 'HIGH'], 'INTUBATION': ['NORMAL', 'ONESIDED', 'ESOPHAGEAL'], 'VENTTUBE': ['LOW', 'ZERO', 'HIGH', 'NORMAL'],
 'VENTALV': ['ZERO', 'LOW','NORMAL', 'HIGH'], 'CATECHOL': ['HIGH', 'NORMAL'], 'ARTCO2': [ 'LOW','NORMAL', 'HIGH'],
 'HR': [ 'LOW','NORMAL', 'HIGH'], 'CO': ['HIGH', 'LOW', 'NORMAL'], 'BP': ['HIGH', 'LOW', 'NORMAL'], 'TPR': [ 'LOW','NORMAL', 'HIGH'],
 'PCWP':[ 'LOW','NORMAL', 'HIGH'], 'INSUFFANESTH': [ 'T','F'], 'SHUNT': ['NORMAL', 'HIGH'], 'PULMEMBOLUS': [ 'T','F'], 
 'CVP': [ 'LOW','NORMAL', 'HIGH'], 'SAO2': [ 'LOW','NORMAL', 'HIGH']}

    for column in df.columns:
        column_values_dict[column]=['missing']+column_values_dict[column]
    
    
    return column_values_dict

