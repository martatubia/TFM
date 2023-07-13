import tkinter as tk
from tkinter import messagebox
import pandas as pd

column_values = {}  # Definir column_values como una variable global
increase=1
margins=[]

class DataFrameColumnSelectorApp:
    def __init__(self, root, dataframe):
        self.root = root
        self.dataframe = dataframe
        self.column_names = list(dataframe.columns)
        self.selected_columns = []
        self.increase = None  # Variable para almacenar el valor de Increase

        self.root.title("Evidence variables")
        self.create_widgets()

    def create_widgets(self):
        # Frame para seleccionar columnas
        column_frame = tk.Frame(self.root)
        column_frame.pack(padx=10, pady=10)

        tk.Label(column_frame, text="Select the columns:").pack(anchor='w')

        for column_name in self.column_names:
            var = tk.BooleanVar()
            checkbox = tk.Checkbutton(column_frame, text=column_name, variable=var, onvalue=True, offvalue=False,
                                      command=lambda col=column_name, var=var: self.toggle_column(col, var))
            checkbox.pack(anchor='w')

        # Etiqueta y campo de entrada para el valor 'Increase'
        tk.Label(column_frame, text="Density function increase:").pack(anchor='w')
        self.increase_entry = tk.Entry(column_frame)
        self.increase_entry.pack(anchor='w')

        submit_button = tk.Button(self.root, text="Insert the values", command=self.open_values_window)
        submit_button.pack(pady=10)

    def open_values_window(self):
        if not self.selected_columns:
            messagebox.showwarning("Advertencia", "Por favor, selecciona al menos una columna.")
            return

        self.increase = self.get_numeric_input(self.increase_entry.get(), "Increase")
        if self.increase is None:
            return

        values_window = tk.Toplevel(self.root)
        values_window.title("Insert the values")
        values_frame = tk.Frame(values_window)
        values_frame.pack(padx=10, pady=10)

        def submit_values():
            global column_values  # Utilizar la variable global column_values

            for col in self.selected_columns:
                try:
                    value = float(self.value_entries[col].get())
                    margin = float(self.margin_entries[col].get())
                    column_values[col] = value # Actualizar la variable global
                    margins.append(margin)
                except ValueError:
                    messagebox.showerror("Error", "Por favor, ingresa valores numéricos para todas las columnas seleccionadas.")
                    return

            increase = self.increase  # Actualizar la variable global con el valor de Increase

            messagebox.showinfo("Información", "Valores guardados exitosamente.")
            values_window.destroy()

        tk.Label(values_frame, text="Insert margins and evidence values:").pack(anchor='w')

        self.value_entries = {}
        self.margin_entries = {}
        for column_name in self.selected_columns:
            tk.Label(values_frame, text=f"{column_name} - valor:").pack(anchor='w')
            value_entry = tk.Entry(values_frame)
            value_entry.pack(anchor='w')
            self.value_entries[column_name] = value_entry

            tk.Label(values_frame, text=f"{column_name} - margin:").pack(anchor='w')
            margin_entry = tk.Entry(values_frame)
            margin_entry.pack(anchor='w')
            self.margin_entries[column_name] = margin_entry

        submit_button = tk.Button(values_frame, text="Save", command=submit_values)
        submit_button.pack(pady=10)

    def toggle_column(self, column_name, var):
        if var.get():
            self.selected_columns.append(column_name)
        else:
            self.selected_columns.remove(column_name)

    def get_numeric_input(self, input_value, label):
        try:
            value = float(input_value)
            return value
        except ValueError:
            messagebox.showerror("Error", f"Por favor, ingresa un valor numérico válido para '{label}'.")
            return None


df = pd.read_csv('COVID-19_real_continuous.csv')


df.drop(['Season', 'Transportation_activity','Face_masks', 'Lockdown', 'Work_and_school_activity', 'Majority_COVID_19_variant', 'Leisure_activity'], axis=1, inplace=True)


# Crear la ventana principal
root = tk.Tk()
app = DataFrameColumnSelectorApp(root, df)
root.mainloop()
#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from EDAspy.optimization import EDA, EGNA
from EDAspy.optimization import EdaResult
from EDAspy.optimization.custom.probabilistic_models import GBN
from EDAspy.optimization.custom.initialization_models import UniformGenInit, MultiGaussGenInit
from time import process_time
from EDAspy.optimization import tools
import networkx as nx
from scipy.stats import multivariate_normal
import math
import matplotlib.pyplot as plt



class myEGNA_evidence(EDA):

    def __init__(self,
                 size_gen: int,
                 max_iter: int,
                 dead_iter: int,
                 n_variables: int,
                 landscape_bounds: tuple,
                 evidences_soft: dict=None,
                 variable_names: list=None,
                 ev_change: list=None,
                 alpha: float = 0.5,
                 elite_factor: float = 0.4,
                 disp: bool = True,
                 black_list: list = None,
                 white_list: list = None,
                 parallelize: bool = False,
                 init_data: np.array = None,
                 int_var: list=None,
                 increase: int=1):
        r"""
            :param size_gen: Population size. Number of individuals in each generation.
            :param max_iter: Maximum number of iterations during runtime.
            :param dead_iter: Stopping criteria. Number of iterations with no improvement after which, the algorithm finish.
            :param n_variables: Number of variables to be optimized.
            :param landscape_bounds: Landscape bounds only for initialization. Limits in the search space.
            :param evidences_soft: observed evidence.
            :param variable_names: names of the variables.
            :param ev_chande: allowed change in the observed variables.
            :param alpha: Percentage of population selected to update the probabilistic model.
            :param elite_factor: Percentage of previous population selected to add to new generation (elite approach).
            :param disp: Set to True to print convergence messages.
            :param black_list: list of tuples with the forbidden arcs in the GBN during runtime.
            :param white_list: list of tuples with the mandatory arcs in the GBN during runtime.
            :param parallelize: True if the evaluation of the solutions is desired to be parallelized in multiple cores.
            :param init_data: Numpy array containing the data the EDA is desired to be initialized from. By default, an
            initializer is used.
            :param int_var: variables to be further interpreted if needed. The beta coefficients variation associated to those variables
            will be shown.
        """

        super().__init__(size_gen=size_gen, max_iter=max_iter, dead_iter=dead_iter,
                         n_variables=n_variables, alpha=alpha, elite_factor=elite_factor, disp=disp,
                         parallelize=parallelize, init_data=init_data)

        self.vars = [str(i) for i in range(n_variables)]
        self.landscape_bounds = landscape_bounds
        #Nota: GBN toma estos argumentos de entrada (self, variables: list, white_list: list = None, black_list: list = None, evidences: dict = None)
        self.pm = GBN(self.vars, black_list=black_list, white_list=white_list)        
        self.init = MultiGaussGenInit(self.n_variables, lower_bound=self.landscape_bounds[0],upper_bound=self.landscape_bounds[1])
       
        self.evidences_soft=evidences_soft
        self.ev_change=ev_change
        self.variable_names=variable_names
        self.int_var=int_var
        self.increase=increase
      
    def minimize(self,  threshold: float, output_runtime: bool = True, *args, **kwargs) -> EdaResult:
            """
            Minimize function to execute the EDA optimization. By default, the optimizer is designed to minimize a cost
            function; if maximization is desired, just add a minus sign to your cost function.

            :param cost_function: cost function to be optimized and accepts an array as argument.
            :param output_runtime: true if information during runtime is desired.
            :return: EdaResult object with results and information.
            :rtype: EdaResult
            """

            history = []
            not_better = 0

            t1 = process_time()

            self.generation = self._initialize_generation()
            self._update_pm()
            
            self.original_arcs=self.pm.print_structure()
            self.original_mu=self.pm.get_mu(self.pm.variables)
            self.original_logl=self.pm.logl(pd.DataFrame([self.original_mu] ))[0]
            self.original_prob=math.exp(self.original_logl)
            self.learned_arcs=self.original_arcs
            self.global_logl=self.original_logl
            
            print('SIGMA ORIGINAL \n', self.pm.get_sigma(self.pm.variables))
            a=pd.DataFrame(self.pm.get_sigma(self.pm.variables))
            a.round(2).to_csv('sigma_original.csv', header=False, index=False)
            print('\n MU ORIGINAL \n', self.pm.get_mu(self.pm.variables))
            pd.DataFrame(self.pm.get_mu(self.pm.variables)).to_csv('mu_original.csv', header=False, index=False)

            
            dictionary = dict(zip(self.pm.variables, self.variable_names))
            self.dict=dictionary
            
            # Additional information about the variation of some variables' beta coefficients. 
            # Firstly we show original parameters
            if self.int_var is not None:
                for v in self.int_var:
                    a=list(self.dict.keys())[list(self.dict.values()).index(v)]
                    parents = self.pm.pm.cpd(a).evidence()
                    parents_trad=list()
                    for w in parents:
                        parents_trad.append(self.dict[w])
                    coefs = self.pm.pm.cpd(a).beta
                    print('\n Parents y betas originales ', parents_trad, coefs)
            
            for _ in range(self.max_iter):
                self._check_generation(self.costfunction3)
                self._truncation()
                self._update_pm()
                       
                best_mae_local = min(self.evaluations)
              
                history.append(best_mae_local)
                best_ind_local = np.where(self.evaluations == best_mae_local)[0][0]
                best_ind_local = self.generation[best_ind_local]
                
                current_logl=self.pm.logl(pd.DataFrame([self.pm.get_mu(self.pm.variables)]))[0]
                
                # update the best values ever
                if best_mae_local < self.best_mae_global:
                    self.best_mae_global = best_mae_local
                    self.best_ind_global = best_ind_local
                    not_better = 0
                    

                else:
                    not_better += 1
                    if not_better == self.dead_iter:
                        break
                
                # Stop the iteration if the value of the dense function at the mode has increased enough
                if current_logl>self.global_logl:
                    self.global_logl=current_logl
                    if math.exp(self.global_logl-self.original_logl)>self.increase:
                        self.learned_arcs=self.pm.print_structure()
                        print('La probabilidad ha aumentado el factor deseado')
                        break
                    
                self._new_generation()

                if output_runtime:
                    print('IT: ', _, '\tBest cost: ', self.best_mae_global)

            if self.disp:
                print("\tNFEVALS = " + str(len(history) * self.size_gen) + " F = " + str(self.best_mae_global))
                print("\tX = " + str(self.best_ind_global))

            t2 = process_time()
            eda_result = EdaResult(self.best_ind_global, self.best_mae_global, len(history) * self.size_gen,
                                history, self.export_settings(), t2-t1)
            
            SHD, diff_edges=self.SHD(self.original_arcs, self.learned_arcs)
            aumento_prob=math.exp(self.global_logl-self.original_logl)

            rounded = [np.round(x,3) for x in self.best_ind_global]
            interp=[rounded, np.round(self.best_mae_global,6), round(aumento_prob,3), round(SHD,3)]
            
            # Save the results if the resulting network is not too much different and the dense function has increase by a sufficient amount.
            if SHD<5 and self.best_mae_global<1 and aumento_prob>self.increase:
                a=pd.DataFrame(self.pm.get_sigma(self.pm.variables))
                a.round(2).to_csv('sigma_opt.csv', header=False, index=False)
       
                arcos_opt=self.pm.print_structure()
                arcos_orig=self.original_arcs
                
                #translation between variables (as integers) and variable names
                for i,j in enumerate(arcos_orig):   
                    arcos_orig[i]=(dictionary[j[0]],dictionary[j[1]])
                for i,j in enumerate(arcos_opt):   
                    arcos_opt[i]=(dictionary[j[0]],dictionary[j[1]])
                    
                tools.plot_bn(arcs=arcos_opt,var_names=self.variable_names, title= 'Opt BN ', output_file= 'final.png')
                tools.plot_bn(arcs=arcos_orig,var_names=self.variable_names,title= 'Original BN', output_file= 'original.png')
                
                #Further interpretation for beta coefficients after the optimization.
                if self.int_var is not None:
                    for v in self.int_var:
                        a=list(self.dict.keys())[list(self.dict.values()).index(v)]
                        parents = self.pm.pm.cpd(a).evidence()
                        parents_trad=list()
                        for w in parents:
                            parents_trad.append(self.dict[w])
                        coefs = self.pm.pm.cpd(a).beta
                        print('\n Parents y betas optimizados ', parents_trad, coefs)
                
                with open("Aristas.txt", "a") as f:
                    print('Aristas distintas', diff_edges, file=f)
                    print('SHD', SHD, file=f)
                
            return eda_result, interp
      

    
    def costfunction3(self, sol):
       
        id=list()
        for v in self.evidences_soft.keys():
            id.append(int(list(self.dict.keys())[list(self.dict.values()).index(v)]))
        mu=[self.pm.get_mu(self.pm.variables)]
        resta=abs(sol[id]-list(self.evidences_soft.values()))
        my_boolean_list=[resta[i]<self.ev_change[i] for i in range(len(self.ev_change))]
        resultado=all(my_boolean_list)
        if resultado:
            a=self.pm.logl(pd.DataFrame([sol]))[0]
            b=self.pm.logl(pd.DataFrame(mu))[0]
            c=b-a 
        else:
            c=9999
        return c
    
       
    #Función para calcular la distancia de Hamming entre DAGs
    def SHD(self,edges1,edges2):
        dist=0
        edges2_rev = [t[::-1] for t in edges2]
        edges1_rev = [t[::-1] for t in edges1]
        diff_edges=[]
        for i in edges1:
            if (i not in edges2) and (i not in edges2_rev):
                dist+=1
                diff_edges.append((self.dict[i[0]], self.dict[i[1]]))
                
        
            if (i not in edges2) and (i in edges2_rev):
                dist+=1
                diff_edges.append((self.dict[i[0]], self.dict[i[1]]))
        for i in edges2:
            if (i not in edges1) and (i not in edges1_rev):
                dist+=1
                diff_edges.append((self.dict[i[0]], self.dict[i[1]]))
            
        return dist, diff_edges
                
        
        
# df = pd.read_csv('measures_v2.csv')
# df2=df.loc[df['profile_id']==17]
# df2.drop(['profile_id', 'u_q','u_d', 'i_d', 'i_q'], axis=1, inplace=True)

# df2=df2.sample(n=3000)

# variable_names=df2.columns
# nmp=df2.to_numpy()

# results=[]
# for i in range(10):
#     if i==0:
#         eda3=myEGNA_evidence(size_gen=3000,max_iter=10, dead_iter=5, n_variables=8,landscape_bounds=(-10,10), evidences_soft={'torque':25, 'coolant':18.9}, ev_change=[0.5,0.1], init_data=nmp, alpha=0.5, variable_names=['coolant','stator_winding','stator_tooth','motor_speed','pm','stator_yoke','ambient','torque'], int_var=['pm'])
#     else:
#         eda3=myEGNA_evidence(size_gen=3000,max_iter=10, dead_iter=5, n_variables=8,landscape_bounds=(-10,10), evidences_soft={'torque':25, 'coolant':18.9}, ev_change=[0.5,0.1], white_list=arcs,  init_data=nmp, alpha=0.5,  variable_names=['coolant','stator_winding','stator_tooth','motor_speed','pm','stator_yoke','ambient','torque'], int_var=['pm'])
#     eda_i, result_i= eda3.minimize(threshold=0.9, cost_function=None)
#     if result_i[1]<=1 and result_i[2]>1.5:
#         arcs=eda3.pm.print_structure()
        
#     else:
#         arcs=None
#     results.append(result_i)



        



variable_names=df.columns
print(variable_names)
nmp=df.to_numpy()
results=[]


# We run the EDA several times, taking advantage of the white list automation from one iteration to the next.
for i in range(2):
    if i==0:
        eda3=myEGNA_evidence(size_gen=1500,max_iter=20, dead_iter=5, n_variables=10,landscape_bounds=(-10,10), evidences_soft=column_values, ev_change=margins, init_data=nmp, alpha=0.5, variable_names=variable_names, int_var=['Positive_tests'], increase=increase )
    else:
        eda3=myEGNA_evidence(size_gen=1500,max_iter=20, dead_iter=5, n_variables=10,landscape_bounds=(-10,10), evidences_soft=column_values, ev_change=margins, white_list=arcs,  init_data=nmp, alpha=0.5,  variable_names=variable_names, int_var=['Positive_tests'], increase=increase)
    eda_i, result_i= eda3.minimize(threshold=0.9, cost_function=None)
    if result_i[1]<=1 and result_i[2]>1.5:
        #White list actualization from one iteration to the next, if the cost function is less than the unit and the dense function has increased by a sufficient amount.
        arcs=eda3.pm.print_structure()
    else:
        arcs=None
    results.append(result_i)
    


#Print the results
print ("{:<90} {:<8} {:<15}  {:<15}".format('Best individual', 'Cost','Dense increase', 'SHD'))

for v in results:
    best, best_cost, increase_prob,shd= v
    print (" {:<90} {:<8} {:<15} {:<15} ".format( str(best), str(best_cost), increase_prob,shd))
    
