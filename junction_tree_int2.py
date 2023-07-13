import networkx as nx
from pgmpy.models import BayesianModel
from tqdm.auto import tqdm
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import my_draw
import matplotlib.patches as mpatches
from pgmpy.factors.discrete import TabularCPD
#from markdown import args, generarPDF

from tqdm.auto import tqdm
from pgmpy.inference import BeliefPropagation 
from pgmpy.models import BayesianNetwork, JunctionTree
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import MarkovNetwork
import math
import matplotlib.pyplot as plt


class BayesianNetwork_int(BayesianNetwork):
    def __init__(self,ebunch=None, latents=set(), interp=True):
        super().__init__(ebunch=ebunch,latents=latents)
        self.interp=interp

    def to_markov_model_int(self):
        moral_graph = self.moralize()
        mm = MarkovNetwork(moral_graph.edges())
        mm.add_nodes_from(moral_graph.nodes())
        mm.add_factors(*[cpd.to_factor() for cpd in self.cpds])
        if self.interp==True:
           return moral_graph, mm
        else:
            return mm

    def cost_triangulate(self, heuristic):
        
        cliques=[]
        triangulated_graph = nx.find_cliques(self.to_markov_model().triangulate(heuristic=heuristic))

        for i in triangulated_graph:
            cliques.append(i)
        w=0
        
        for i in range(0, len(cliques)):
            a=1
            for j in cliques[i]:
                a=a*self.get_cardinality(j)
            w=w+a
        w=math.log(w,2)
        return w

    def best_heuristic(self):
        d=['H1', 'H2', 'H3', 'H4', 'H5', 'H6']
        we=[]
        dicc={'H1': 0, 'H2': 0, 'H3': 0, 'H4': 0, 'H5': 0, 'H6': 0}


        for h in d:
            dicc[h]=self.cost_triangulate(h)
            we.append(dicc[h])
        elimination_order=min(dicc, key=dicc.get)
        fig_heur=plt.figure()
        plt.bar(d,we,align='center')
        plt.title("Comparison of w(G) after triangulating with  different heuristics")
        plt.savefig("Cost_of_triangulation_heuristics.jpg",  dpi=600)
        return elimination_order

    def to_junction_tree_int(self):
        
        moral_graph, mm= self.to_markov_model_int()
        triang_graph = mm.triangulate(heuristic=self.best_heuristic())
        if self.interp==True:
            return moral_graph, triang_graph, super().to_junction_tree()
        else:
            
            return super().to_junction_tree()
        


class BeliefPropagation_int(BeliefPropagation):
    def __init__(self, model):
        super().__init__(model)
        #print(self.model)
        self.model=model
        

    def dibujar(self, model, posic='circular'):
        poss = self.switch(model, posic)
        moral_graph, triang_graph, self.junction_tree = model.to_junction_tree_int()
        labels = {}  
        for enum in enumerate(self.model.nodes()): 
            labels[enum[1]]='N_'+str(enum[0])
        
        # fig=plt.figure()
        nx_moral_graph=nx.Graph(moral_graph.edges())
        nx_triang_graph=nx.Graph(triang_graph.edges())
        nx_junct_graph=nx.DiGraph()
        nx_junct_graph.add_edges_from(self.junction_tree.edges())
        fig = plt.figure("Moralized graph and triangulated graph", figsize=(12,12))
        axgrid=fig.add_gridspec(16,4)
        ax0=fig.add_subplot(axgrid[0:4,:])
        ax1=fig.add_subplot(axgrid[4:8,:])
        ax2 = fig.add_subplot(axgrid[8:12,:])
        ax3 = fig.add_subplot(axgrid[12:16,:])
                
        nx.draw(model,ax=ax0, pos=poss, with_labels = False, arrowsize=20)
        nx.draw_networkx_labels(model,pos=poss, ax=ax0,labels=labels,font_size=7, font_weight="bold", alpha=1)

        nx.draw(nx_moral_graph,ax=ax1, pos=poss, with_labels = False)
        nx.draw_networkx_labels(nx_moral_graph,pos=poss, ax=ax1,labels=labels,font_size=7, font_weight="bold", alpha=1)
        patchList = []
        for key in labels.keys():
            data_key = mpatches.Patch(color='none', label=str(key)+'='+str(labels[key]))
            patchList.append(data_key)

        ax0.legend(loc="upper right", handles=patchList, fontsize='x-small')

        nx.draw(nx_triang_graph,ax=ax2,pos=poss,  with_labels=False)
        nx.draw_networkx_labels(nx_triang_graph,pos=poss, ax=ax2,labels=labels,font_size=7, font_weight="bold", alpha=1)

        edges=model.to_junction_tree().edges()
        sepset={}

        for i in edges:
            sepset[i]=set(i[0]).intersection(set(i[1]))
                
        labels2 = {}  
        for enum in enumerate(self.junction_tree.nodes()):    
            labels2[enum[1]]='C_'+str(enum[0])

        poss = self.switch(self.junction_tree, posic)
        nx.draw_networkx_nodes(nx_junct_graph, pos=poss, ax=ax3, nodelist=set(nx_junct_graph.nodes), node_size=300)
        nx.draw_networkx_labels(nx_junct_graph,pos=poss,ax=ax3,labels=labels2,font_size=7, font_weight="bold")
        nx.draw_networkx_edges(nx_junct_graph, pos=poss, ax=ax3,edgelist=self.junction_tree.edges())
        patchList = []
        for key in labels2.keys():
            data_key = mpatches.Patch(color='none', label=str(key)+'='+str(labels2[key]))
            patchList.append(data_key)

        ax3.legend( handles=patchList, loc='upper right', fontsize='x-small' )
        
      
        ax0.set_title("Bayesian network")
        ax1.set_title("Moralized")
        ax2.set_title("Triangulated")
        ax3.set_title("Junction")
        fig.tight_layout()
        ax3.axis('off')
        plt.savefig("Construccion_junction_tree.jpg",  dpi=600)
        
    def switch(self,model, posic):
        #A partir del string 'circular', 'kamada', 'random', o 'spring' se generan los tipos de grafos correspondientes.
            if posic == "circular":
                return nx.circular_layout(model)
            elif posic == "kamada":
                return nx.kamada_kawai_layout(model)
            elif posic == "random":
                return nx.random_layout(model)
            elif posic == "spring":
                return nx.spring_layout(model)
            else:
                print("Elija entre circular, kamada, random o spring. Default layout=spring")
                return nx.spring_layout(model)
            
    def query_int(self,variables,evidence=None,virtual_evidence=None,joint=True,show_progress=True):
        self.dibujar(self.model)
        print(super().query(variables,evidence=None,virtual_evidence=None,joint=True,show_progress=True))
        
            
    def query_ev_sensit(
            self,
            variables,
            evidence=None,
            virtual_evidence=None,
            joint=True,
            show_progress=True,
        ):
            if evidence is None:
                return 'Introduce evidencia a comparar'
            evidence = evidence if evidence is not None else dict()
            orig_model = self.model.copy()
            self.dibujar(self.model)
            # Step 1: Parameter Checks
            common_vars = set(evidence.keys() if evidence is not None else []).intersection(set(variables))
            if common_vars:
                raise ValueError(
                    f"Can't have the same variables in both `variables` and `evidence`. Found in both: {common_vars}"
                )

            # Step 2: If virtual_evidence is provided, modify model and evidence.
            if isinstance(self.model, BayesianNetwork) and (virtual_evidence is not None):
                self._virtual_evidence(virtual_evidence)
                virt_evidence = {"__" + cpd.variables[0]: 0 for cpd in virtual_evidence}
                return self.query(
                    variables=variables,
                    evidence={**evidence, **virt_evidence},
                    virtual_evidence=None,
                    joint=joint,
                    show_progress=show_progress,
                )
            results=[]
            for i in range(len(evidence)):
                
                # Step 3: Do network pruning.
                if isinstance(self.model, BayesianNetwork_int):
                    self.model, evidence = self._prune_bayesian_model(variables, evidence)
                self._initialize_structures()

                # Step 4: Run inference.
                result = self._query_ev_sensit(
                    variables=variables,
                    operation="marginalize",
                    evidence=evidence,
                    joint=joint,
                    show_progress=show_progress,
                )
                print('results \n', result)
                self.__init__(orig_model)
                # if i==0:
                self.dibujar_subtree(subtree=subtree, filename='subtree'+str(i)+'.png')

                if joint:
                    results.append(result.normalize(inplace=False))
                else:
                    results.append(result)

            
            with open("Comparison_evidence.txt", "w") as o:
                
                o.write(str(results))
                

                
    
    def _query_ev_sensit(self, variables, operation, evidence=None, joint=True, show_progress=True):
        #Idéntica a la función _query de la librería pgmpy pero haciendo la variable subtree global
        

        is_calibrated = self._is_converged(operation=operation)
        # Calibrate the junction tree if not calibrated
        if not is_calibrated:
            self.calibrate()

        if not isinstance(variables, (list, tuple, set)):
            query_variables = [variables]
        else:
            query_variables = list(variables)
        query_variables.extend(evidence.keys() if evidence else [])

        # Find a tree T' such that query_variables are a subset of scope(T')
        nodes_with_query_variables = set()
        for var in query_variables:
            nodes_with_query_variables.update(
                filter(lambda x: var in x, self.junction_tree.nodes())
            )
        subtree_nodes = nodes_with_query_variables

        # Conversion of set to tuple just for indexing
        nodes_with_query_variables = tuple(nodes_with_query_variables)
        # As junction tree is a tree, that means that there would be only path between any two nodes in the tree
        # thus we can just take the path between any two nodes; no matter there order is
        for i in range(len(nodes_with_query_variables) - 1):
            subtree_nodes.update(
                nx.shortest_path(
                    self.junction_tree,
                    nodes_with_query_variables[i],
                    nodes_with_query_variables[i + 1],
                )
            )
        subtree_undirected_graph = self.junction_tree.subgraph(subtree_nodes)
        global subtree
        # Converting subtree into a junction tree
        if len(subtree_nodes) == 1:
            subtree = JunctionTree()
            subtree.add_node(subtree_nodes.pop())
        else:
            subtree = JunctionTree(subtree_undirected_graph.edges())
            print(subtree.nodes())

        # Selecting a node is root node. Root node would be having only one neighbor
        if len(subtree.nodes()) == 1:
            root_node = list(subtree.nodes())[0]
        else:
            root_node = tuple(
                filter(lambda x: len(list(subtree.neighbors(x))) == 1, subtree.nodes())
            )[0]
        clique_potential_list = [self.clique_beliefs[root_node]]
        
        # For other nodes in the subtree compute the clique potentials as follows
        # As all the nodes are nothing but tuples so simple set(root_node) won't work at it would update the set with
        # all the elements of the tuple; instead use set([root_node]) as it would include only the tuple not the
        # internal elements within it.
        parent_nodes = set([root_node])
        nodes_traversed = set()
        while parent_nodes:
            parent_node = parent_nodes.pop()
            for child_node in set(subtree.neighbors(parent_node)) - nodes_traversed:
                clique_potential_list.append(
                    self.clique_beliefs[child_node]
                    / self.sepset_beliefs[frozenset([parent_node, child_node])]
                )
                parent_nodes.update([child_node])
            nodes_traversed.update([parent_node])

        # Add factors to the corresponding junction tree
        subtree.add_factors(*clique_potential_list)

        # Sum product variable elimination on the subtree
        variable_elimination = VariableElimination(subtree)
        
        if operation == "marginalize":
            
            return variable_elimination.query(
            variables=variables,
            evidence=evidence,
            joint=joint,
            show_progress=show_progress,
        )
        elif operation == "maximize":
            return variable_elimination.map_query(
                variables=variables, evidence=evidence, show_progress=show_progress
            )
            

    def dibujar_subtree(self, subtree, posic='circular', filename='output.png'):
        #Plot del subarbol del junction tree que se usa para hacer variable elimination.
        
        poss = self.switch(self.junction_tree, posic)
        fig, ax = plt.subplots()
        # fig1=plt.figure('Subtree')
        
        nx_junct_graph=nx.DiGraph()
        nx_junct_graph.add_edges_from(self.junction_tree.edges())
        highlight_nodes=subtree.nodes()
     
                        
        labels2 = {}  
        for enum in enumerate(self.junction_tree.nodes()):    
            labels2[enum[1]]='C_'+str(enum[0])
            
        patchList = []
        for key in labels2.keys():
            if key in list(subtree.nodes()):
                data_key = mpatches.Patch(color='blue', label=str(key)+'='+str(labels2[key]))
            else:
                data_key = mpatches.Patch(color='lightblue', label=str(key)+'='+str(labels2[key]))
            patchList.append(data_key)
            
            
        nx.draw(nx_junct_graph,pos=poss, with_labels = False, node_color='lightblue', node_size=300, arrowsize=20)
        nx.draw_networkx_nodes(nx_junct_graph, poss, nodelist=highlight_nodes, node_color='blue', node_size=300)
        nx.draw_networkx_labels(nx_junct_graph,pos=poss, labels=labels2,font_size=7, font_weight="bold", alpha=1)

  
        ax.set_title("Subtree for variable elimination")
        ax.legend( handles=patchList, loc='upper right', fontsize='x-small', bbox_to_anchor=(0.9, 0.1))
        # plt.show()
        plt.savefig(filename,  dpi=600)
            



    def _calibrate_junction_tree(self, operation):
        #Además de calibrar hace una representación del paso de mensajes. Modifica la función _calibrate_junction_tree de pgmpy.
    
        # Initialize clique beliefs as well as sepset beliefs
        self.clique_beliefs = {
            clique: self.junction_tree.get_factors(clique)
            for clique in self.junction_tree.nodes()
        }
        self.sepset_beliefs = {
            frozenset(edge): None for edge in self.junction_tree.edges()
        }
        #nivel avanzado de interpretabilidad
        with open("Calibration_result.txt", "w") as o:
            o.write('Factores iniciales de los cliques \n')
        with open("Calibration_result.txt", "a") as o:
            for valor in self.clique_beliefs.values(): 
                o.write(str(valor))
                o.write('\n')
        
        edges1=[]
        edges2=[]
        dict_nodes=dict()
        labels = {}  
        for enum in enumerate(self.junction_tree.nodes()): 
            labels[enum[1]]= 'C_'+str(enum[0])
            dict_nodes['C_'+str(enum[0])]=enum[1]
                
        for clique in self.junction_tree.nodes():
           
            if not self._is_converged(operation=operation):
               
                plt.figure()
                graph=nx.DiGraph()

                neighbors = self.junction_tree.neighbors(clique)
                # update root's belief using neighbor clique's beliefs
                # upward pass
                
                for neighbor_clique in neighbors:
                    self._update_beliefs(neighbor_clique, clique, operation=operation)
                    edges1=edges1+[(neighbor_clique,clique)]
                #labels for the edges: in order to add the equivalent name of the edge, y use a dictionary with name:name_equivalence and search for the index whose value is the edge (stack overflow)
                mensajes_up = [r'$\beta$'+'('+str(list(dict_nodes.keys())[list(dict_nodes.values()).index(enum[1][1])])+')'+'='+r'$\beta$'+'('+str(list(dict_nodes.keys())[list(dict_nodes.values()).index(enum[1][1])])+')'+r'$\sigma$'+'('+str(list(dict_nodes.keys())[list(dict_nodes.values()).index(enum[1][0])])+','+str(list(dict_nodes.keys())[list(dict_nodes.values()).index(enum[1][1])])+')'+'/'+r'$\mu$'+'('+str(list(dict_nodes.keys())[list(dict_nodes.values()).index(enum[1][0])])+','+str(list(dict_nodes.keys())[list(dict_nodes.values()).index(enum[1][1])])+')' for enum in enumerate(edges1)]  
                Dict_up = dict(zip(edges1, mensajes_up))
                graph.add_edges_from(edges1)
                
                
                bfs_edges = nx.algorithms.breadth_first_search.bfs_edges(
                    self.junction_tree, clique
                )
                # update the beliefs of all the nodes starting from the root to leaves using root's belief
                # downward pass
                for edge in bfs_edges:
                    self._update_beliefs(edge[0], edge[1], operation=operation)
                    edges2=edges2+[(edge[0], edge[1])]
                
                graph.add_edges_from(edges2)
                
                mensajes_down = [r'$\beta$'+'('+str(list(dict_nodes.keys())[list(dict_nodes.values()).index(enum[1][1])])+')'+'='+r'$\beta$'+'('+str(list(dict_nodes.keys())[list(dict_nodes.values()).index(enum[1][1])])+')'+r'$\sigma$'+'('+str(list(dict_nodes.keys())[list(dict_nodes.values()).index(enum[1][0])])+','+str(list(dict_nodes.keys())[list(dict_nodes.values()).index(enum[1][1])])+')'+'/'+r'$\mu$'+'('+str(list(dict_nodes.keys())[list(dict_nodes.values()).index(enum[1][0])])+','+str(list(dict_nodes.keys())[list(dict_nodes.values()).index(enum[1][1])])+')' for enum in enumerate(edges2)]  
                
                Dict = dict(zip(edges2, mensajes_down))
                
                  
               
                
                pos1 = nx.spring_layout(graph)
                nx.draw_networkx_nodes(graph, pos=pos1, nodelist=set(graph.nodes), node_size=300)

                #Add colors to edges acoording to the direction of the messages: edges1 and edges 2
                listedges=['r' for edge in edges1]+['g' for edge in edges2]
                
                nx.draw_networkx_edges(graph, pos=pos1,edgelist=(edges1)+(edges2), connectionstyle='arc3, rad = 0.3', edge_color=listedges, label='upward')
                nx.draw_networkx_labels(graph,pos1,labels,font_size=7, font_weight="bold", alpha=1)

              
                red_patch=mpatches.Patch(color='red', label='upward')
                blue_patch = mpatches.Patch(color='green', label='downward')
                list_patch=[red_patch, blue_patch]
                for x in dict_nodes:
                    list_patch.append(mpatches.Patch('blue', label=x+'='+str(dict_nodes[x])))
                
                plt.legend(handles=list_patch, fontsize=8)
                #import matplotlib.patches as mpatches
            
                my_draw.my_draw_networkx_edge_labels(graph, pos=pos1, edge_labels=Dict,rotate=False,rad = 0.3, font_color='green', font_size=6)
                my_draw.my_draw_networkx_edge_labels(graph, pos=pos1, edge_labels=Dict_up,rotate=False,rad = 0.3, font_color='red', font_size=6)

                
            else:
                break
        with open("Calibration_result.txt", "a") as o:
            o.write('factores calibrados de los cliques\n')
            for valor in self.clique_beliefs.values(): 
                o.write(str(valor))
                o.write('\n')
        
        plt.savefig("Calibracion.jpg",  dpi=600)
    





from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import K2Score
df = pd.read_csv("ASIA_DATA.csv")
scoring_method = K2Score(data=df)
est = HillClimbSearch(data=df)
estimated_model = est.estimate(scoring_method=scoring_method, max_indegree=4, max_iter=int(1e4))
model = BayesianNetwork_int(estimated_model.edges)
model.fit(df, estimator=BayesianEstimator, prior_type="BDeu") 


inference2=BeliefPropagation_int(model)
inference2.query_ev_sensit(variables=["bronc"], evidence={"lung": "yes"})


from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import txt2pdf

# Ruta del archivo PDF
class args():
    def __init__(self):
        self.filename='Comparison_evidence.txt'
        self.font='Courier'
        self.font_size=10.0
        self.extra_vertical_space=0.0
        self.kerning=0.0
        self.media='A4'
        self.landscape=False
        self.margin_left=2.0
        self.margin_right=2.0
        self.margin_top=2.0
        self.margin_bottom=2.0
        self.output='output.pdf'
        self.author=''
        self.title=''
        self.quiet=False
        self.break_on_blanks=False
        self.encoding='utf8'
        self.page_numbers=False
        self.line_numbers=False
    
def generarPDF():
    ruta_archivo = 'informe_JT.pdf'
    w, h = A4
    x = 120
    y = h - 45
    # Crear un lienzo para el PDF
    c = canvas.Canvas(ruta_archivo, pagesize=letter)

    # Título del informe
    titulo = "Junction tree - reasoning"
    c.setFont("Helvetica-Bold", 15)
    c.drawString(200, h-80, titulo)

    c.setFont('Helvetica-Bold', 14)
    resultado = "Sequence of trees"
    c.drawString(60,h-100, resultado)
    c.setFont('Helvetica', 12)
   
    ruta_imagen1 = 'Construccion_junction_tree.jpg'
    c.drawImage(ruta_imagen1,80, h-580, width=460, height=460)

    c.showPage()
    c.setFont('Helvetica-Bold', 14)
    resultado = "Calibration"
    c.drawString(60, h-80, resultado)
    c.setFont('Helvetica-Bold', 12)
    linea1= "It is based on message passing that consists of an upward pass and a downward pass. "
    linea2='In a calibrated clique-tree, the marginal probability over particular variables does not depend'
    linea3='on the clique we selected.'
    # linea3=
    c.drawString(60, h-100, linea1)
    c.drawString(60, h-120, linea2)
    c.drawString(60, h-140, linea3)
    # c.drawString(60, h-520, linea3)

    ruta_imagen2 = 'Calibracion.jpg'
    c.drawImage(ruta_imagen2, 80,h-440 , width=280, height=280)
    
    c.showPage()
    c.setFont('Helvetica-Bold', 14)
    resultado = "Subtree"
    c.drawString(60, h-80, resultado)
    ruta_imagen3 = 'subtree0.png'
    c.drawImage(ruta_imagen3, 80, h-450, width=350, height=350)
    c.showPage()
    c.save()
    txt2pdf.PDFCreator(args(),txt2pdf.Margins(right=2.0, left=2.0, top=2.0, bottom=2.0)).generate()
    




generarPDF()

