import os
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
utils = importr('utils')
base = importr('base')
utils.chooseCRANmirror(ind=1) # select the first mirror in the list
import pgmpy
from rpy2.robjects import pandas2ri
pandas2ri.activate()
import rpy2.robjects as ro
import pandas as pd
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import GibbsSampling
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import ApproxInference
class GibbsSampling_int(GibbsSampling):
    def __init__(self, model):
        
        super().__init__(model)

    #Función de sampleado con diagnóstico de convergencia

    def sample_convergencia(self,size=1000,diag='gelman',multivariate=True, evidence=None):
        import subprocess
        samples = self.sample(size=size) 
        samples2=self.sample(size=size)
        samples3=self.sample(size=size)
        samples.to_csv('numeros.csv', header=True, index=False)
        samples2.to_csv('numeros2.csv', header=True, index=False)
        samples3.to_csv('numeros3.csv', header=True, index=False)
        import rpy2.robjects as robjects
        # Load packages
        coda= importr('coda')
        path=os.getcwd()

        if diag=='gelman':
            
            if multivariate==False:

                

                gelman_save=robjects.r('''
                        f <- function(path) {
                        setwd(path)
                        objeto<-gelman.diag(mcmc.list(mcmc(read.csv('numeros.csv',header = TRUE, sep = ",")),mcmc(read.csv('numeros2.csv',header = TRUE, sep = ",")),mcmc(read.csv('numeros3.csv',header = TRUE, sep = ","))),multivariate=FALSE)
                        capture.output(objeto, file="Shrink_factors.txt")
                        pdf('Gelman_rubin.pdf')
                        gelman.plot(mcmc.list(mcmc(read.csv('numeros.csv',header = TRUE, sep = ",")),mcmc(read.csv('numeros2.csv',header = TRUE, sep = ","))))
                        dev.off()
                        
                        }''')
                
            else:
             

                gelman_save=robjects.r('''
                        f <- function(path) {
                        setwd(path)
                        objeto<-gelman.diag(mcmc.list(mcmc(read.csv('numeros.csv',header = TRUE, sep = ",")),mcmc(read.csv('numeros2.csv',header = TRUE, sep = ",")),mcmc(read.csv('numeros3.csv',header = TRUE, sep = ","))),multivariate=TRUE)
                        capture.output(objeto, file="Shrink_factors.txt")
                        pdf('Gelman_rubin.pdf')
                        gelman.plot(mcmc.list(mcmc(read.csv('numeros.csv',header = TRUE, sep = ",")),mcmc(read.csv('numeros2.csv',header = TRUE, sep = ","))))
                        dev.off()
                        
                        }''')
                
            gelman_save(path)
            path_2 = 'Gelman_rubin.pdf'
            archivo=open("Shrink_factors.txt", "r")
            print(archivo.read())
            subprocess.Popen([path_2], shell=True)
            
        elif diag=='geweke':
            geweke_save=robjects.r('''
                        f <- function(path) {
                        setwd(path)
                        objeto<-geweke.diag(mcmc.list(mcmc(read.csv('numeros.csv',header = TRUE, sep = ",")),mcmc(read.csv('numeros2.csv',header = TRUE, sep = ",")),mcmc(read.csv('numeros3.csv',header = TRUE, sep = ","))))
                        capture.output(objeto, file="geweke.txt")
                        pdf('Geweke.pdf')
                        geweke.plot(mcmc.list(mcmc(read.csv('numeros.csv',header = TRUE, sep = ",")),mcmc(read.csv('numeros2.csv',header = TRUE, sep = ","))))
                        dev.off()
                        
                        }''')
            
            geweke_save(path)
            path_2 = 'Geweke.pdf'
            archivo=open("geweke.txt", "r")
            print(archivo.read())
            subprocess.Popen([path_2], shell=True)
            
        elif diag=='heidel':
            geweke_save=robjects.r('''
                        f <- function(path) {
                        setwd(path)
                        objeto<-heidel.diag(mcmc.list(mcmc(read.csv('numeros.csv',header = TRUE, sep = ",")),mcmc(read.csv('numeros2.csv',header = TRUE, sep = ",")),mcmc(read.csv('numeros3.csv',header = TRUE, sep = ","))))
                        capture.output(objeto, file="heidel.txt")
                        }''')
            
            geweke_save(path)
            archivo=open("heidel.txt", "r")
            print(archivo.read())
          


        mcmcplot=robjects.r('''
                        f <- function(path) {
                        setwd(path)
                        library(coda)
                        pdf('mcmcplot.pdf')
                        densplot(mcmc(read.csv('numeros.csv',header = TRUE, sep = ",")), trace=FALSE)
                        dev.off()
                        
                        }''')
       
        mcmcplot(path)
        
        path_3 = 'mcmcplot.pdf'
        subprocess.Popen([path_3], shell=True)
        return samples

class ApproxInference_GibbsChoice(ApproxInference):
    def __init__(self, model):
        super().__init__(model)
     
        print('En el método query puede elegir el qué tipo de sampling hacer en función de si hay evidencia (type_ev= likelihood or rejection) o de si no hay evidencia (type_non_ev = Gibbs or Forward)\n')

       
    def query_GibbsChoice(self,variables,n_samples=15000,evidence=None, type_ev='likelihood', type_non_ev='Gibbs', diag='gelman', virtual_evidence=None,joint=True,show_progress=True):
        type_ev_list=['rejection', 'likelihood']
        type_non_ev_list=['Gibbs', 'Forward']
        diag_list=['gelman', 'geweke', 'heidel']

        
        if evidence==None:
            if type_non_ev not in  type_non_ev_list:
                return 'No es posible ese tipo de muestreo sin evidencia. Solo son posibles Gibbs y Forward. \n'
            if diag not in diag_list:
                return 'Diagnostico de convergencia de MC no válido. \n'
        else:
            if type_ev not in  type_ev_list:
                return 'No es posible ese tipo de muestreo con evidencia. Solo son posibles likelihood and rejection. \n'
        if evidence==None:

            if type_non_ev=='Forward':
                print(self.query(variables,n_samples=n_samples,evidence=None, virtual_evidence=None,joint=True,show_progress=True))
            else:

                # Step 1: Generate samples for the query
                aa=GibbsSampling_int(self.model)
                print(n_samples)
                samples=aa.sample_convergencia(size=n_samples, diag=diag, multivariate=True)
                print(samples)
                # Step 2: Compute the distributions and return it.
                b=self.get_distribution(samples=samples, variables=variables, joint=joint)
                
                return b
        else:
            if type_ev=='likelihood':
                from pgmpy.sampling import BayesianModelSampling
                aa=BayesianModelSampling(self.model)
                for var, state in evidence.items():
                    if state not in self.model.states[var]:
                        raise ValueError(f"Evidence state: {state} for {var} doesn't exist")
                        
                samples = aa.likelihood_weighted_sample(size=n_samples,evidence=[(k, v) for k, v in evidence.items()])
                dnomin=samples['_weight'].sum()
                print('Resultado con likelihood weighted sampling')
                return samples.groupby(variables)['_weight'].sum()/dnomin
            else:
                print('Resultado con rejection sampling')
                return self.query(variables,n_samples=n_samples,evidence=evidence, virtual_evidence=None,joint=True,show_progress=True)


            

from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import K2Score
import networkx as nx

df = pd.read_csv("ASIA_DATA.csv")
scoring_method = K2Score(data=df)
est = HillClimbSearch(data=df)
estimated_model = est.estimate(scoring_method=scoring_method, max_indegree=4, max_iter=int(1e4))

model = BayesianNetwork(estimated_model.edges)
model.fit(df, estimator=BayesianEstimator, prior_type="BDeu") 
inference4=ApproxInference_GibbsChoice(model)
print(inference4.query_GibbsChoice(variables=["bronc"], diag='gelman'))










