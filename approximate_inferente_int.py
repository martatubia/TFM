

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


alarm_model = BayesianNetwork([("Burglary", "Alarm"),("Earthquake", "Alarm"),("Alarm", "JohnCalls"),("Alarm", "MaryCalls"),])


cpd_burglary = TabularCPD(
    variable="Burglary", variable_card=2, values=[[0.999], [0.001]])


cpd_earthquake = TabularCPD(
    variable="Earthquake", variable_card=2, values=[[0.998], [0.002]]
)


cpd_alarm = TabularCPD(
    variable="Alarm",
    variable_card=2,
    values=[[0.999, 0.71, 0.06, 0.05], [0.001, 0.29, 0.94, 0.95]],
    evidence=["Burglary", "Earthquake"],
    evidence_card=[2, 2],)


cpd_johncalls = TabularCPD(
    variable="JohnCalls",
    variable_card=2,
    values=[[0.95, 0.1], [0.05, 0.9]],
    evidence=["Alarm"],
    evidence_card=[2],)


cpd_marycalls = TabularCPD(
    variable="MaryCalls",
    variable_card=2,
    values=[[0.1, 0.7], [0.9, 0.3]],
    evidence=["Alarm"],
    evidence_card=[2],)

print(cpd_marycalls)

# Associating the parameters with the model structure
alarm_model.add_cpds(
    cpd_burglary, cpd_earthquake, cpd_alarm, cpd_johncalls, cpd_marycalls
)
# Associating the parameters with the model structure
alarm_model.add_cpds(cpd_burglary, cpd_earthquake, cpd_alarm, cpd_johncalls, cpd_marycalls)
gibbs_alarm=GibbsSampling(alarm_model)
cpds={"MaryCalls": cpd_marycalls, "JohnCalls": cpd_johncalls, "Alarm": cpd_alarm, "Earthquake": cpd_earthquake, "Burglary": cpd_burglary}

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

       
    def query_GibbsChoice(self,variables,n_samples=10000,evidence=None, type_ev='likelihood', type_non_ev='Gibbs', diag='gelman', virtual_evidence=None,joint=True,show_progress=True):
        # cpds2=self.model.get_cpds()
        # dict_cpds=dict()
        # for j in cpds2:
        #     dict_cpds[j.variable]=j

        # import itertools
        # roots=self.model.get_roots()

        # for i in roots:
            
        #     for j in list(dict_cpds[i].values):
        #         if j<0.002:
        #             print('Variable ' + str(i)+ ' has extreme low probability value: ', str(j))
        #             children=self.model.get_children(node=i)
                    
        #             for c in children:
        #                 values_children = list(itertools.chain(*list(itertools.chain(*dict_cpds[c].values))))
        #                 buscando_discrepancias = [value>0.8 for value in values_children]
        #                 if any(buscando_discrepancias):
        #                     print('Además, presenta discrepancia entre prior y posterior con su variable hija '+ c + '(posterior > 0.8)\n')
        #                     print(dict_cpds[i])
        #                     print(dict_cpds[c])



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
                samples=aa.sample_convergencia(size=n_samples, diag=diag)
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


            


inference4=ApproxInference_GibbsChoice(alarm_model)
variables=['MaryCalls', 'Alarm']
evidence={'Earthquake':1}
print(inference4.query_GibbsChoice(variables=variables))










