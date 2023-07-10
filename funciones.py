from pgmpy.models import BayesianNetwork, JunctionTree
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import MarkovNetwork
from pgmpy.inference import VariableElimination
import numpy as np
import pandas as pd
import random
from itertools import product
from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import K2Score
import networkx as nx

def initialize(evidence, model):
# Esta función inicializa el proceso de búsqueda local y selecciona una variable aleatoria y un estado aleatorio para comenzar la búsqueda.

    variables=list(set(model.nodes())-set(list(evidence.keys())))
    rIndex = random.randrange(len(variables))
    rVar=variables[rIndex]
    #Elijo el estado de esa variable aleatoriamente
    rIndex = random.randrange(len(model.states[rVar]))
    rVarState=model.states[rVar][rIndex]
    state=dict()
    state[rVar]=rVarState
    

    return state, rVar

def neighbors_add(state, variables, model):
#Genera vecinos para el estado actual al agregar diferentes valores a las variables no observadas.

    neigh=[]
    for i in range(len(variables)):
        name=variables[i]
        
        for j in range(len(model.states[name])):
            rVarState=model.states[name][j]
            neigh.append({**state, **{name:rVarState}})
   
    return neigh
            
def neighbors_change(state, model):
#Genera vecinos para el estado actual al cambiar los valores de las variables no observadas.

    neigh_add=[]
    for i in state.keys():

        for j in model.states[i]:
            
            #if int(j)!=int(state[i]):
            if j!=state[i]:    
                state_aux=state.copy()
                state_aux[i]=j
                neigh_add.append(state_aux.copy())
 
    return neigh_add

def GBF(dict_query, evidence, model):
# Calculo del Generalized Bayes factor

    array1=[model.states[i] for i in dict_query.keys()]
   
    combinations=[]
    for elem in product(*array1):
        combinations.append(elem)
    #En las tablas de las queries aparecen todas las posibles combinaciones
    index=tuple(list(dict_query.values()))
    posiciones = combinations.index(index)
    
    inference = VariableElimination(model)
   
    phi_query1=inference.query(dict_query, evidence=evidence)
    phi_query2=inference.query(dict_query)
   
    flat_list1=phi_query1.values.flatten()
    flat_list2=phi_query2.values.flatten()
    probab1=flat_list1[posiciones]
    probab2=flat_list2[posiciones]
    
    gbf=(probab1*(1-probab2))/(probab2*(1-probab1))
    return gbf


def get_MRE(neigh1, neigh2,evidence, model):
# Compara los GBF de los vecinos y devuelve el vecino con el GBF más alto.

    best=neigh2[0]
    best_gbf=GBF(neigh2[0], evidence, model)
   
    for i in neigh1:
        a=GBF(i, evidence, model)
        
        if a>best_gbf:
            best=i
            best_gbf=a
    
    for i in neigh2:
        a=GBF(i, evidence,model)
        
        if a>best_gbf:
            best=i
            best_gbf=a
    return best, best_gbf


def strong_dom(dict1, df,gbf):
# Esta función verifica si el diccionario dict1 está fuertemente dominado por alguna instancia en el dataframe df en términos de explicaciones 
# y GBF.
    

    for s1 in range(len(df)):
           
        if not all(((j in dict1) and df.iloc[s1]['Explanations'][j]==dict1[j] and gbf<=df.iloc[s1]['GBF']) for j in df.iloc[s1]['Explanations']):
            #print('el candidato no esta dominado por esta instancia del dataframe existente')
            return False
        
            #  print('El siguiente candidato será rechazado por estar dominado fuertemente')
            #  print('dataframe', df.iloc[s1]['Explanations'], 'gbf', df.iloc[s1]['GBF'])
            #  print('dict', dict1, 'gbf', gbf)
    return True
         
def weak_dom(dict1, df,gbf):
# Esta función verifica si el diccionario dict1 está débilmente dominado por alguna instancia en el dataframe df en términos de explicaciones
# y GBF.     

    for s1 in range(len(df)):
           
        if not all(((j in df.iloc[s1]['Explanations']) and df.iloc[s1]['Explanations'][j]==dict1[j] and gbf<=df.iloc[s1]['GBF']) for j in dict1):
            #print('el candidato no esta debilment dominado por esta instancia del dataframe existente')
            return False
        
            # print('dataframe', df.iloc[s1]['Explanations'], 'gbf', df.iloc[s1]['GBF'])
            # print('dict', dict1, 'gbf', gbf)
    return True
                
def strong_dom2(dict1, dict2, gbf1, gbf2):

    if all(key in dict1 and dict1[key] == dict2[key] for key in dict2) and gbf2>=gbf1:
        
        return True
    else:
        return False

def weak_dom2(dict1, dict2, gbf1, gbf2):

    if all(key in dict2 and dict2[key] == dict1[key] for key in dict1) and gbf2>=gbf1:
        
        return True
    else:
        return False 
    
 
def get_kMRE(selected_df, model, value):
    
# Esta función get_kMRE realiza el algoritmo de búsqueda de MRE para generar una lista de explicaciones más relevantes. 
# Utiliza las funciones initialize, neighbors_add, neighbors_change, get_MRE, GBF, strong_dom, weak_dom, strong_dom2 y weak_dom2 para realizar el proceso de búsqueda. 
# Luego, filtra las explicaciones encontradas eliminando aquellas que están dominadas fuertemente o débilmente por otras explicaciones.
# Finalmente, devuelve las tres mejores explicaciones filtradas.  

    evidence = selected_df.to_dict('records')[0]
    dead_iter = 5
   
    kMRE = []
    kGBF = []
    kdf = pd.DataFrame()
    kdf['Explanations'] = None
    kdf['GBF'] = None

    for l in range(25):
        state, rVar = initialize(evidence, model)
        not_better = 0
        for k in range(10):

            if k == 0:
                xlocbest = state
                gbflocal = GBF(state, evidence, model)

            variables = list(model.nodes())
            variables = list((set(variables) - set(list(evidence.keys()))) - set(list(xlocbest.keys())))

            neigh1 = neighbors_add(xlocbest, variables=variables, model=model)
            neigh2 = neighbors_change(xlocbest, model)

            state, best_gbf = get_MRE(neigh1, neigh2, evidence, model)
            if best_gbf > gbflocal:
                xlocbest = state
                gbflocal = best_gbf
            else:
                not_better += 1
                if not_better == dead_iter:
                    kMRE.append(xlocbest)
                    kGBF.append(gbflocal)
                    break
        kMRE.append(xlocbest)
        kGBF.append(gbflocal)
    poolMRE = []

    for i in range(len(kMRE)):

        if len(kdf) == 0:
            df2 = pd.DataFrame({'Explanations': [kMRE[i]], 'GBF': kGBF[i]})
            kdf = pd.concat([kdf, df2], ignore_index=True)

        else:

            if kGBF[i] > kdf['GBF'].min() and not (strong_dom(kMRE[i], kdf, kGBF[i])) and not (
                    weak_dom(kMRE[i], kdf, kGBF[i])):
                df2 = pd.DataFrame({'Explanations': [kMRE[i]], 'GBF': kGBF[i]})
                kdf = pd.concat([kdf, df2], ignore_index=True)

    rows_to_remove = []
    for i in range(len(kdf)):
        current_row = kdf.iloc[i]
        for j in range(i + 1, len(kdf)):
            if strong_dom2(current_row['Explanations'], kdf.iloc[j]['Explanations'], current_row['GBF'], kdf.iloc[j]['GBF']) or weak_dom2(current_row['Explanations'], kdf.iloc[j]['Explanations'], current_row['GBF'], kdf.iloc[j]['GBF']):
                rows_to_remove.append(i)

    df_filtered = kdf.drop(rows_to_remove)
    df_filtered_reset = df_filtered.sort_values('GBF', ascending=False).reset_index(drop=True).head(value)
    return df_filtered_reset
    
    
def learn_model(df):


    scoring_method = K2Score(data=df)
    est = HillClimbSearch(data=df)
    estimated_model = est.estimate(scoring_method=scoring_method, max_indegree=4, max_iter=int(1e4))

    model = BayesianNetwork(estimated_model.edges)
    model.fit(df, estimator=BayesianEstimator, prior_type="BDeu") # default equivalent_sample_size=5
    return model

def compare_instances(instance1, instance2):
    """
    Compara dos instancias de un DataFrame y cuenta las diferencias.

    Args:
        instance1 (pandas.Series): Primera instancia a comparar.
        instance2 (pandas.Series): Segunda instancia a comparar.

    Returns:
        int: Número de diferencias entre las dos instancias.
    """
    differences = 0

    for column, value1 in instance1.iteritems():
        value2 = instance2[column]

        if pd.isnull(value1) and not pd.isnull(value2):
            differences += 1
        if not pd.isnull(value1) and pd.isnull(value2):
            differences += 1
        elif not pd.isnull(value1) and not pd.isnull(value2) and value1 != value2:
            differences += 1

    return differences