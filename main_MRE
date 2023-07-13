import tkinter as tk
from tkinter import ttk
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.estimators import BayesianEstimator
import funciones
import preprocessing_covid
import preprocessing_motor, preprocessing_alarm
import funciones_plot

selected_df = None

def show_dropdown(df, selection_vars):
    selected_columns = [column for column, var in selection_vars.items() if var.get() == 1]
    values = {}
    if selected_columns:
        values = {column: list(df[column].unique()) for column in selected_columns}
        create_dropdown_window(values)

def create_dropdown_window(values):
    dropdown_window = tk.Toplevel(root)
    dropdown_window.title("Desplegables")

    dropdowns = {}

    for column, column_values in values.items():
        label = tk.Label(dropdown_window, text=f"Seleccione un valor para {column}:")
        label.pack()

        selected_var = tk.StringVar(dropdown_window)
        selected_var.set(column_values[0])

        dropdown = ttk.Combobox(dropdown_window, values=column_values, textvariable=selected_var)
        dropdown.pack()

        dropdowns[column] = selected_var

    def save_selection():
        global selected_df
        selected_values = {column: var.get() for column, var in dropdowns.items()}
        selected_df = pd.DataFrame(selected_values, index=[0])
        dropdown_window.destroy()

    save_button = tk.Button(dropdown_window, text="Guardar selección", command=save_selection)
    save_button.pack()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Selección de columnas")
    root.geometry('500x500')
    df, ordinal_mapping = preprocessing_alarm.discretize()
    column_names = df.columns.tolist()

    num_rows=int(len(df.columns)/2)
      
    selection_vars = {}
    for column in column_names:
        var = tk.IntVar()
        checkbox = tk.Checkbutton(root, text=column, variable=var)
        checkbox.grid(row=(column_names.index(column) % num_rows), column=(column_names.index(column) // num_rows), sticky='w')

        #checkbox.pack()
        selection_vars[column] = var

    ok_button = tk.Button(root, text="OK", command=lambda: show_dropdown(df, selection_vars))
  
    ok_button.grid(row=num_rows, column=0, columnspan=1)
    #ok_button.pack()

    root.mainloop()

    if selected_df is not None:
        model=funciones.learn_model(df)

        df_filtered_reset=funciones.get_kMRE(selected_df,model)
        evidence = selected_df.to_dict('records')[0]
        df_filtered_reset1, df_filtered_reset_na = funciones_plot.final_preprocessing_plotly(df_filtered_reset)
        
        column_values1 = preprocessing_alarm.column_values_dict(df)
        funciones_plot.plot_plotly(column_values1, df_filtered_reset1)

        # df_filtered_reset2, df_filtered_reset_na = funciones_plot.final_preprocessing_matplotlib(ordinal_mapping, df_filtered_reset)
        # column_values2 = preprocessing_covid.column_values_covid(df)
        # funciones_plot.plot_matplotlib(ordinal_mapping, column_values2, df_filtered_reset2, evidence, model)
        
        # df_filtered_reset1, df_filtered_reset_na = funciones_plot.final_preprocessing_plotly(df_filtered_reset)
        # column_values1 = preprocessing_alarm.column_values_dict(df)
        # funciones_plot.plot_plotly(column_values1, df_filtered_reset1)
        
        print(df_filtered_reset_na)
        fila0=pd.Series(df_filtered_reset_na.iloc[0,:-1])
        lista=[0]
        for index, row in df_filtered_reset_na.iloc[1:,:-1].iterrows():
            instance = pd.Series(row)
            comparison=funciones.compare_instances(fila0, instance)
            lista.append(comparison)
            
            print('\n Indice k: ', index+1, '; Distancia: ', comparison)
        df_filtered_reset_na['Comparison']=lista
        print(df_filtered_reset_na)
        df_filtered_reset_na.to_csv('kMRE_diversity3_alarm.csv', index=False)
        
        
        

        
        

            





