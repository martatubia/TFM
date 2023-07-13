# Soy Marta Alonso Tubía
En este repositorio se encuentran los trabajos realizados en el Trabajo de Fin de Máster del MSc in Data Science.

# Probabilidad condicionada a una inferencia:
- approximate_inference_int
- variable_elimination_int
- junction_tree_int

Funciones auxiliares para generar los informes pdf a modo de resumen del 'reasoning process' en el caso de Variable elimination y junction tree clustering algorithm: my_draw, markdown, pdf_Variable_elimination, txt2pdf. 

Si no se desea generar ningun informe, simplemente comentar 'generarPDF' al final del archivo.

Ejemplos de informes: informe_VE, informe_JT

# Inferencia abductiva
- main_EDA: es una extensión de EDAspy. Para ejecutar con otros dataset, solo es necesario cambiar el nombre del csv de lectura.

# Explicación más relevante
- main_MRE : es el programa principal. Utiliza como files auxiliares:
  - functiones : calcula las k-MREs.
  - funciones_plot : representa en coordenadas paralelas las k-MRE. En principio incluye la posibilidad de hacerlo con matplotlib y con plotly, aunque en la tesis aparezca solamente el resultado que ofrece plotly.
  - preprocessing_alarm: preprocesado del dataset ALARM para realizar la representación de las k-MRE. Hay que tener en cuenta que los valores de las variables pueden aparecer desordenados en los ejes. En nuestro caso eran variables ordinales en su mayoría y quisimos que apareciesen en orden los valores, por mantener la semántica. Para ello se customizó una función column_values_dict, que devuelve para cada variable sus posibles valores ordenados. Como norma general se deberá customizar para cada dataset.
  



