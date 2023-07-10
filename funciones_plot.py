import plotly.graph_objects as go
import pandas as pd
def final_preprocessing_plotly(df_filtered_reset):
    df_aux=pd.DataFrame.from_records(df_filtered_reset['Explanations'])
    df_filtered_reset=pd.concat([df_aux, df_filtered_reset],axis=1)
    df_filtered_reset.drop(['Explanations'],axis=1, inplace=True)
    df_filtered_reset['GBF']=round(df_filtered_reset['GBF'].head(5),2)
    df_filtered_reset_missing=df_filtered_reset.fillna('missing')
    return df_filtered_reset_missing,df_filtered_reset


def plot_plotly(column_values_dict, df_filtered_reset):
    import plotly.express as px
    dimensions = []

    for i in df_filtered_reset.columns[:-1]:
        
        dimensions.append(dict(
            range=[1, len(column_values_dict[i])+1],
            label=i,
            ticktext=column_values_dict[i],
            tickvals=list(range(1, int(len(column_values_dict[i]))+1)),
            values=list(column_values_dict[i].index(k)+1 for k in list(df_filtered_reset[i]))
        ))

    fig = go.Figure(data=go.Parcoords(line=dict(color = df_filtered_reset['GBF'],colorscale='Jet',showscale = True), dimensions=dimensions))
    #paper_bgcolor="LightSteelBlue",
    fig.update_layout(
    title="Alarm dataset")
    fig.write_image("fig3.png")
    fig.show()

import matplotlib.pyplot as plt
from pgmpy.inference import VariableElimination

def final_preprocessing_matplotlib(ordinal_mapping,df_filtered_reset):
    df_aux=pd.DataFrame.from_records(df_filtered_reset['Explanations'])
    df_filtered_reset=pd.concat([df_aux, df_filtered_reset],axis=1)
    df_filtered_reset.drop(['Explanations'],axis=1, inplace=True)

    #matplotlib needs numerical attributes to plot paralell coordinates
    
    for i in df_aux.columns:
        df_filtered_reset[i] = df_filtered_reset[i].map(ordinal_mapping)

    df_filtered_reset['GBF']=round(df_filtered_reset['GBF'].head(5),2)
    df_filtered_reset_missing= df_filtered_reset.fillna(0)
    
    return df_filtered_reset_missing, df_filtered_reset

def plot_matplotlib(ordinal_mapping, column_values, df_filtered_reset, evidence, model):
    cadena = ""
    for clave, valor in evidence.items():
        cadena += f" {clave}: {valor};"
    
    #ordinal_mapping = {'Very Low': 1, 'Low': 2, 'Medium':3, 'High': 4, 'Very High': 5}
    
    plt.figure(figsize=(10, 6))
    pc=pd.plotting.parallel_coordinates(frame=df_filtered_reset,class_column= 'GBF', colormap='viridis')
    #plt.yticks([0, 1, 2, 3, 4,5], ['Missing','Very low', 'Low','Medium', 'High', 'Very High'])
    plt.yticks([0, 1, 2, 3, 4,5], column_values)
    plt.legend().remove()
    columns_names=list(df_filtered_reset.columns[:-1])
    belief_propagation = VariableElimination(model)
    a=belief_propagation.map_query(variables=columns_names,evidence=evidence)
    transformed_dict = dict(map(lambda item: (item[0], ordinal_mapping[item[1]]), a.items()))
    plt.plot(list(transformed_dict.keys()), list(transformed_dict.values()), color='black', linestyle='--', linewidth=4, label='MAP')


    plt.title(cadena)
    plt.legend()
    plt.show()