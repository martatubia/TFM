from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork, JunctionTree, DynamicBayesianNetwork
from pgmpy.factors.discrete import DiscreteFactor
from opt_einsum import contract
from tqdm.auto import tqdm
from pgmpy.global_vars import SHOW_PROGRESS
import sys
from pgmpy.factors import factor_product
import itertools
import networkx as nx
from opt_einsum import contract
from tqdm.auto import tqdm
from pgmpy.inference.EliminationOrder import MinFill, MinNeighbors, MinWeight, WeightedMinFill
from pgmpy.factors.discrete import TabularCPD

class VariableElimination_INT(VariableElimination):
    
    def best_heuristic(self):
        order_weightedminfill=WeightedMinFill(self.model).get_elimination_order(self.model.nodes)
        cost_WMF=self.induced_width(order_weightedminfill)

        order_minneighbors=MinNeighbors(self.model).get_elimination_order(self.model.nodes)
        cost_MN=self.induced_width(order_minneighbors)

        order_minweight=MinWeight(self.model).get_elimination_order(self.model.nodes)
        cost_MW=self.induced_width(order_minweight)

        order_minfill=MinFill(self.model).get_elimination_order(self.model.nodes)
        cost_MF=self.induced_width(order_minfill)

        d={"WeightedMinFill": cost_MF, "MinNeighbors": cost_MN, "MinWeight": cost_MW, "MinFill": cost_WMF}
        elimination_heur=min(d, key=d.get)
        print("La heurística de eliminación óptima es", elimination_heur)
        x=['WMF', 'MN', 'MW', 'MF']
        y=[cost_WMF, cost_MN, cost_MW, cost_MF]

        plt.bar(x,y,align='center')
        plt.title("Comparison of heuristics for selecting variable elimination ordering")
      

        
        plt.savefig("Cost_variable_eliminationordering.png", dpi=600)
        return elimination_heur
    
    def variable_elimination2(
        self,
        variables,
        operation,
        evidence=None,
        elimination_order="MinFill",
        joint=True,
        show_progress=True,
    ):
   
        archivo2 = open("variable_elimination2.txt", "w")
        # Redirigir la salida estándar al archivo
        sys.stdout = archivo2
        
        # Step 1: Deal with the input arguments.
        if isinstance(variables, str):
            raise TypeError("variables must be a list of strings")
        if isinstance(evidence, str):
            raise TypeError("evidence must be a list of strings")

        # Dealing with the case when variables are not provided.
        if not variables:
            all_factors = []
            for factor_li in self.factors.values():
                all_factors.extend(factor_li)
            if joint:
                return factor_product(*set(all_factors))
            else:
                return set(all_factors)
        
        # Step 2: Prepare data structures to run the algorithm.
        eliminated_variables = set()
        # Get working factors and elimination order
        
        working_factors = self._get_working_factors(evidence)
    
       
       
        elimination_order = self._get_elimination_order(
            variables, evidence, elimination_order, show_progress=show_progress
        )
        print('\n')
        print('Elimination order:', ', '.join(str(elemento) for elemento in elimination_order))
        print('\n')
        
        # Step 3: Run variable elimination
        if show_progress and SHOW_PROGRESS:
            pbar = tqdm(elimination_order)
        else:
            pbar = elimination_order
        phis=[]
        
        for var in pbar:
            print('Variable to eliminate', var)
            if show_progress and SHOW_PROGRESS:
                #pbar.set_description(f"Eliminating: {var}")
            # Removing all the factors containing the variables which are
            # eliminated (as all the factors should be considered only once)
                factors = [
                factor
                for factor, _ in working_factors[var]
                if not set(factor.variables).intersection(eliminated_variables)
            ]

            print('Factors involved \n')
            for i in factors:
                print(i)
            phi = factor_product(*factors)
            print('\n')
            phi = getattr(phi, operation)([var], inplace=False)
            print('Phi \n', phi)
            print('\n')
            phis.append((phi, factors))
            del working_factors[var]
            for variable in phi.variables:
                working_factors[variable].add((phi, var))
            eliminated_variables.add(var)

        # Step 4: Prepare variables to be returned.
        final_distribution = set()
        for node in working_factors:
            for factor, origin in working_factors[node]:
                if not set(factor.variables).intersection(eliminated_variables):
                    final_distribution.add((factor, origin))
        final_distribution = [factor for factor, _ in final_distribution]

        if joint:
            if isinstance(self.model, BayesianNetwork):
                print('Final distribution \n')
                print(factor_product(*final_distribution).normalize(inplace=False))
                return factor_product(*final_distribution).normalize(inplace=False)
            else:
                return factor_product(*final_distribution)
        else:
            query_var_factor = {}
            if isinstance(self.model, BayesianNetwork):
                for query_var in variables:
                    phi = factor_product(*final_distribution)
                    query_var_factor[query_var] = phi.marginalize(
                        list(set(variables) - set([query_var])), inplace=False
                    ).normalize(inplace=False)
            else:
                for query_var in variables:
                    phi = factor_product(*final_distribution)
                    query_var_factor[query_var] = phi.marginalize(
                        list(set(variables) - set([query_var])), inplace=False
                    )
            sys.stdout = sys.__stdout__

            # Cerrar el archivo
            archivo2.close()
            return query_var_factor

    def query2(
            self,
            variables,
            evidence=None,
            virtual_evidence=None,
            elimination_order="greedy",
            joint=True,
            show_progress=True,
        ):

            evidence = evidence if evidence is not None else dict()

            # Step 1: Parameter Checks
            common_vars = set(evidence if evidence is not None else []).intersection(
                set(variables)
            )
            if common_vars:
                raise ValueError(
                    f"Can't have the same variables in both `variables` and `evidence`. Found in both: {common_vars}"
                )

            # Step 2: If virtual_evidence is provided, modify the network.
            if isinstance(self.model, BayesianNetwork) and (virtual_evidence is not None):
                self._virtual_evidence(virtual_evidence)
                virt_evidence = {"__" + cpd.variables[0]: 0 for cpd in virtual_evidence}
                return self.query(
                    variables=variables,
                    evidence={**evidence, **virt_evidence},
                    virtual_evidence=None,
                    elimination_order=elimination_order,
                    joint=joint,
                    show_progress=show_progress,
                )
            #print('Network before being pruned: ', ', '.join(str(elemento) for elemento in self.model.nodes()))
            
            print('\n')
            # Step 3: Prune the network based on variables and evidence.
            if isinstance(self.model, BayesianNetwork):
                model_reduced, evidence = self._prune_bayesian_model(variables, evidence)
                factors = model_reduced.cpds
                # print('Network pruning based on variables and evidence: ', ', '.join(str(elemento) for elemento in model_reduced.nodes()))
                # print('\n')
                print('Factors of the reduced model \n')
                for i in factors:
                    print(i)
            else:
                model_reduced = self.model
                factors = self.model.factors
                
                
            
            poss = nx.circular_layout(self.model)
            fig, ax = plt.subplots()
           
            
            nx_reduced_graph=nx.DiGraph()
            nx_reduced_graph.add_edges_from(self.model.edges())
            highlight_nodes=model_reduced.nodes()
        
                            
        
            nx.draw(nx_reduced_graph,pos=poss, with_labels = False, node_color='lightblue', node_size=300, arrowsize=20)
            nx.draw_networkx_nodes(nx_reduced_graph, poss, nodelist=highlight_nodes, node_color='blue', node_size=300)
            nx.draw_networkx_labels(nx_reduced_graph,pos=poss,font_size=7, font_weight="bold", alpha=1)

            # plt.title('Subtree for variable elimination')
            #ax.set_title("Reduced model for variable elimination (optimization)")

            plt.savefig('reduced.png',  dpi=600)    
                

            # Step 4: If elimination_order is greedy, do a tensor contraction approach
            #         else do the classic Variable Elimination.
            if elimination_order == "greedy":
                # Step 5.1: Compute the values array for factors after reducing them to provided
                #           evidence.
                evidence_vars = set(evidence)
                reduce_indexes = []
                reshape_indexes = []
                for phi in factors:
                    indexes_to_reduce = [
                        phi.variables.index(var)
                        for var in set(phi.variables).intersection(evidence_vars)
                    ]
                    indexer = [slice(None)] * len(phi.variables)
                    for index in indexes_to_reduce:
                        indexer[index] = phi.get_state_no(
                            phi.variables[index], evidence[phi.variables[index]]
                        )
                    reduce_indexes.append(tuple(indexer))
                    reshape_indexes.append(
                        [
                            1 if indexer != slice(None) else phi.cardinality[i]
                            for i, indexer in enumerate(reduce_indexes[-1])
                        ]
                    )

                # Step 5.2: Prepare values and index arrays to do use in einsum
                if isinstance(self.model, JunctionTree):
                    var_int_map = {
                        var: i
                        for i, var in enumerate(
                            set(itertools.chain(*model_reduced.nodes()))
                        )
                    }
                else:
                    var_int_map = {var: i for i, var in enumerate(model_reduced.nodes())}
                einsum_expr = []
                for index, phi in enumerate(factors):
                    einsum_expr.append(
                        (phi.values[reduce_indexes[index]]).reshape(reshape_indexes[index])
                    )
                    einsum_expr.append([var_int_map[var] for var in phi.variables])
                result_values = contract(
                    *einsum_expr, [var_int_map[var] for var in variables], optimize="greedy"
                )

                # Step 5.3: Prepare return values.
                result = DiscreteFactor(
                    variables,
                    result_values.shape,
                    result_values,
                    state_names={var: model_reduced.states[var] for var in variables},
                )
               
                if joint:
                    if isinstance(
                        self.model, (BayesianNetwork, JunctionTree, DynamicBayesianNetwork)
                    ):
                        print('result normalizado \n', result.normalize(inplace=False))
                        return result.normalize(inplace=False)
                    else:
                        return result
                else:
                    result_dict = {}
                    all_vars = set(variables)
                    if isinstance(
                        self.model, (BayesianNetwork, JunctionTree, DynamicBayesianNetwork)
                    ):
                        for var in variables:
                            result_dict[var] = result.marginalize(
                                all_vars - {var}, inplace=False
                            ).normalize(inplace=False)
                    else:
                        for var in variables:
                            result_dict[var] = result.marginalize(
                                all_vars - {var}, inplace=False
                            )

                    return result_dict

            else:
                
        
              
                # Step 5.1: Initialize data structures for the reduced bn.
                reduced_ve = VariableElimination_INT(model_reduced)
                reduced_ve._initialize_structures()

                # Step 5.2: Do the actual variable elimination
                result = reduced_ve.variable_elimination2(
                    variables=variables,
                    operation="marginalize",
                    evidence=evidence,
                    elimination_order=elimination_order,
                    joint=joint,
                    show_progress=show_progress,
                )
            
            
            return result
    



from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import txt2pdf
import PyPDF2

class args():
    def __init__(self):
        self.filename='archivo_salida.txt'
        self.font='Courier'

        self.font_size=9.0
        self.extra_vertical_space=0.0
        self.kerning=0.0
        self.media='A4'
        self.landscape=False
        self.margin_left=2.0
        self.margin_right=2.0
        self.margin_top=2.0
        self.margin_bottom=2.0
        self.output='output_VE.pdf'
        self.author=''
        self.title=''
        self.quiet=False
        self.break_on_blanks=False
        self.encoding='utf8'
        self.page_numbers=False
        self.line_numbers=False
    
def generarPDF():
    ruta_archivo = 'informe_VE.pdf'
    w, h = A4
    x = 120
    y = h - 45
    # Crear un lienzo para el PDF
    c = canvas.Canvas(ruta_archivo, pagesize=letter)

    # Título del informe
    titulo = "Variable Elimination - reasoning"

    c.setFont("Helvetica-Bold", 15)
    c.drawString(200, h-80, titulo)

    ruta_imagen1 = 'reduced.png'
    c.drawImage(ruta_imagen1,80, h-460, width=340, height=340)
    
    ruta_imagen2 = 'Cost_variable_eliminationordering.png'
    c.drawImage(ruta_imagen2,80, h-820, width=340, height=340)
    c.save()
    txt2pdf.PDFCreator(args(),txt2pdf.Margins(right=2.0, left=2.0, top=2.0, bottom=2.0)).generate()
    

    # Abrir los archivos PDF en modo de lectura binaria
    with open('informe_VE.pdf', "rb") as pdf1_file, open('output_VE.pdf', "rb") as pdf2_file:
        # Crear objetos PDFReader para los archivos PDF
        pdf1_reader = PyPDF2.PdfReader(pdf1_file)
        pdf2_reader = PyPDF2.PdfReader(pdf2_file)

        # Crear un nuevo objeto PDFWriter
        pdf_writer = PyPDF2.PdfWriter()

        # Agregar todas las páginas del archivo 1 al PDFWriter
        for page_num in range(len(pdf1_reader.pages)):
            page = pdf1_reader.pages[page_num]
            pdf_writer.add_page(page)

        # Agregar todas las páginas del archivo 2 al PDFWriter
        for page_num in range(len(pdf2_reader.pages)):
            page = pdf2_reader.pages[page_num]
            pdf_writer.add_page(page)

        # Guardar el PDF concatenado en un nuevo archivo
        output_file = "archivo_concatenado_VE.pdf"
        with open(output_file, "wb") as output:
            pdf_writer.write(output)

        print("La concatenación se ha completado. El archivo resultante se encuentra en:", output_file)


from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import MaximumLikelihoodEstimator
alarm_model = get_example_model("asia")
samples = BayesianModelSampling(alarm_model).forward_sample(size=int(1e5))
model_struct = BayesianNetwork(ebunch=alarm_model.edges())
model_struct.fit(data=samples, estimator=MaximumLikelihoodEstimator)




import sys
def query_with_logging(model, variables, evidence=None):
    archivo = open("pasos_query.txt", "w")
    sys.stdout = archivo

    inference = VariableElimination_INT(alarm_model)
    best_heu=inference.best_heuristic()
    resultado = inference.query2(variables, evidence, elimination_order=best_heu)

    sys.stdout = sys.__stdout__
    archivo.close()


query_with_logging(model_struct,variables=["bronc"], evidence={"lung": "yes"})

archivo1 = open("pasos_query.txt", 'r')
archivo2 = open('variable_elimination2.txt', 'r')
contenido1 = archivo1.read()
contenido2 = archivo2.read()
archivo1.close()
archivo2.close()

archivo_salida = open('archivo_salida.txt', 'w')
archivo_salida.write(contenido1)
archivo_salida.write(contenido2)
archivo_salida.close()


generarPDF()







