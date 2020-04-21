import streamlit as st
import numpy as np, pandas as pd
from scipy.integrate import odeint
from numpy import linalg as LA
import plotly.express as px
from streamlit import caching
import matplotlib.pyplot as plt
import itertools
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
        
IncubPeriod = 0

#Cálculo dos parâmetros do modelo SEIR            
def params(IncubPeriod, FracMild, FracCritical, FracSevere, TimeICUDeath, CFR, DurMildInf, DurHosp, i, FracAsym, DurAsym, N):
        
        a0 = 1/(IncubPeriod) #Frequência de incubação até possibilidade de transmissão

        f = FracAsym #Fração de assintomáticos
        
        g0 = 1/DurAsym #Taxa de recuperação dos assintomáticos
        
        if FracCritical==0:
            u=0 #Taxa de mortos
        else:
            u=(1/TimeICUDeath)*(CFR/FracCritical) #Taxa de mortos
            
        g1 = (1/DurMildInf)*FracMild #Taxa de recuperação I1
        p1 =(1/DurMildInf) - g1 #Taxa de progreção I1

        g3 =(1/TimeICUDeath)-u #Taxa de recuperação I3
        
        p2 =(1/DurHosp)*(FracCritical/(FracCritical+FracSevere)) #Taxa de progressão I2
        g2 = (1/DurHosp) - p2 #Taxa de recuperação de I2

        ic=np.zeros(8) #Inicia vetor da população (cada índice para cada tipo de infectado, exposto, etc)
        ic[0]= N-i #População sucetível
        ic[1] = i #População exposta
        
        return a0, u, g0, g1, g2, g3, p1, p2, f, ic
#Menu dos parâmetros gerais

#Modelo SEIR
def seir(y,t,a0,g0,g1,g2,g3,p1,p2,u,b0,b1,b2,b3,f): 
        
    dy=[0, #Sucetiveis y[0]
        0, #Expostos y[1]
        0, #I0 - Assintomáticos y[3]
        0, #I1 - Leves y[4]
        0, #I2 - Graves y[5]
        0, #I3 - Críticos y[6]
        0, #Recuperados y[7]
        0] #Mortos y[8]
    
    S = y[0] #Sucetiveis y[0]
    E0 = y[1] #Expostos y[1]
    I0 = y[2] #I0 - Assintomáticos y[3]
    I1 = y[3] #I1 - Leves y[4]
    I2 = y[4] #I2 - Graves y[5]
    I3 = y[5] #I3 - Críticos y[6]
    R = y[6] #Recuperados y[7]
    D = y[7] #Mortos y[8]
    
    seas=1
    
    dy[0] = -(b0*I0 + b1*I1 +b2*I2 + b3*I3)*S*seas #Variação de sucetíveis
    
    dy[1] = (b0*I0 + b1*I1 + b2*I2 + b3*I3)*S*seas - a0*E0 #Variação de expostos não transmissores
        
    dy[2] = f*a0*E0 - g0*I0 #Variação de assintomáticos
    
    dy[3] = (1-f)*a0*E0 - g1*I1 - p1*I1 #Variação de casos leves

    dy[4] = p1*I1-g2*I2-p2*I2 #Variação de casos graves
    
    dy[5] = p2*I2-g3*I3-u*I3 #Variação de casos críticos
    
    dy[6] = g0*I0+g1*I1+g2*I2+g3*I3 #Variação de recuperados
    
    dy[7] = u*I3 #Variação de mortos
    
    return dy

def main(IncubPeriod):

    #Definindo valores padrões dos parâmetros
    IncubPeriod = 5
    DurMildInf = 6
    FracSevere = 0.15
    FracCritical = 0.05
    FracMild = 1 - FracSevere - FracCritical
    ProbDeath = 0.5
    TimeICUDeath = 8
    DurHosp = 6
    tmax = 32
    i = 1
    TimeStart = 0
    TimeEnd = tmax
    FracAsym = 0.2
    DurAsym = 6
    CFR = FracCritical*ProbDeath
    b0 = 0.5
    b1 = 2.0
    b2 = 0.1
    b3 = 0.1
    reduc1 = 0.5
    reduc2 = 0.5
    reduc3 = 0.5
    reducasym = 0.5
    
    st.title("Report da simulação do COVID-19")
    dados = pd.read_csv('dados_cidades.csv',encoding = "ISO-8859-1")
    #cidade = st.selectbox("Selecione a cidade", list(dados['Cidade']))
    cidade = 'Passo Fundo'
    dados = dados.set_index('Cidade')
    dados['População'] = dados['População']#.apply(lambda x: ''.join(x.split('.')))
    N = int(dados.loc[cidade,'População'])
    st.table(pd.DataFrame(dados.loc[cidade,:].to_dict(), index = ['']))
    page = 'Report'
    if page == 'Report':
        dados_casos = pd.read_csv(cidade + '.csv', encoding = "ISO-8859-1")
        dados_casos = dados_casos.reset_index(drop = True).reset_index()  
        dados_casos['index_aux'] = dados_casos['index']
        dados_casos['Sim'] = 'REAL'
        lista_expostos = [1,5]
        lista_b0 = [0.3,0.5,0.7,1]
        lista_b1 = [0.3,0.5,0.7,1]
        lista_f = [0.2]
        lista_delay = [10,15]
        lista_f_grave = [0.2,0.25,0.3,0.4]
        lista_f_critico = [0.20,0.25,0.3]
        lista_p_morte = [0.5]
        
        b2 = b2/N
        b3 = b2/N
        
        erro = pd.DataFrame(itertools.product(lista_expostos,lista_b0,lista_b1, lista_f, lista_f_grave, lista_f_critico, lista_delay, lista_p_morte), columns = ['E','b0','b1','f','f_grave','f_critico','delay','p_morte'])
        maxs = 21
        window = 10
        ################### Encontra parâmetros que minimiza o erro do modelo em relação aos dados reais #############
        def rmse_calc(x, maxs, windows):
            E = x['E'] #Expostos iniciais
            b0 = x['b0']/N #Taxa de transmissão assintomática 
            b1 = x['b1']/N #Taxa de transmissão inf.leve
            f = x['f'] #Fração de assintomáticos
            FracSevere = x['f_grave'] #Fração grave
            FracCritical = x['f_critico'] #Fração crítico 
            FracMild = 1 - FracSevere - FracCritical #Fração leve
            ProbDeath = x['p_morte'] #Probabilidade de morte
            CFR = FracCritical*ProbDeath #Taxa de morte
            
            pop = np.zeros(8)
            pop[0] = N - E
            pop[1] = E
            
            a0, u, g0, g1, g2, g3, p1, p2, f, ic = params(IncubPeriod, FracMild, FracCritical, FracSevere, TimeICUDeath, CFR, DurMildInf, DurHosp, i, FracAsym, DurAsym, N)
            
            delay = x['delay']
            dados_casos['index'] = dados_casos['index_aux'] + delay
            
            T = dados_casos['index'].max()
            tvec=np.arange(0,T+1,1)
            soln = odeint(seir,pop,tvec,args=(a0,g0,g1,g2,g3,p1,p2,u,b0,b1,b2,b3,f))
            
            names = ["Sucetíveis","Expostos","Assintomáticos","Inf. Leve","Inf. Grave","Inf. Crítico","Recuperados","Mortos"]
            df_ = pd.DataFrame(soln, columns = names)
            df_['index'] = tvec
            df_aux = df_[(df_.index >= delay) & (df_.index < T+1)].reset_index(drop = True)
            #st.write(max,max-window)
            
            MSE_dom =0# mean_squared_error(y_true = (dados_casos['Domiciliar']), y_pred = (df_aux['Inf. Crítico']))
            MSE_crit = 0#mean_squared_error(y_true = (dados_casos['UTI'][maxs-window:maxs]), y_pred = (df_aux['Inf. Crítico'][maxs-window:maxs]))
            MSE_grave = mean_squared_error(y_true = (dados_casos['Enfermaria'][maxs-window:maxs]), y_pred = (df_aux['Inf. Grave'][maxs-window:maxs]))
            MSE_mortos = mean_squared_error(y_true = (dados_casos['Obitos'][maxs-window:maxs]), y_pred = (df_aux['Mortos'][maxs-window:maxs]))

            return np.array([round(MSE_dom**(0.5),1),round(MSE_grave**(0.5),1), round(MSE_crit**(0.5),4), round(MSE_mortos**(0.5),1)])
        
        
        erro['rmse_list'] = erro.apply(lambda x: rmse_calc(x,maxs,window),axis = 1)
        erro['rmse'] = erro['rmse_list'].apply(np.sum)
        st.write(erro.sort_values('rmse'))
        st.write(erro.shape)
        parametros = erro.reset_index(drop = True).sort_values('rmse').reset_index(drop = True).head(10)

        for k in range(0,10):
            st.title('---------------------------------------------------')
            st.subheader('Familia de parâmetros '+str(k))
            E = parametros.iloc[k,0]
            b0 = parametros.iloc[k,1]/N
            b1 = parametros.iloc[k,2]/N
            f = parametros.iloc[k,3]
            FracSevere = parametros.iloc[k,4]
            FracCritical = parametros.iloc[k,5]
            FracMild = 1 - FracSevere - FracCritical
            ProbDeath = parametros.iloc[k,7]
            CFR = FracCritical*ProbDeath
            
            a0, u, g0, g1, g2, g3, p1, p2, f, ic = params(IncubPeriod, FracMild, FracCritical, FracSevere, TimeICUDeath, CFR, DurMildInf, DurHosp, i, FracAsym, DurAsym, N)
            
            pop = np.zeros(8)
            pop[0] = N - E
            pop[1] = E
            
            delay = parametros.iloc[k,6]
            
            tvec=np.arange(0,365,1)
            soln=odeint(seir,pop,tvec,args=(a0,g0,g1,g2,g3,p1,p2,u,b0,b1,b2,b3,f))
            names = ["Sucetíveis","Expostos","Assintomáticos","Inf. Leve","Inf. Grave","Inf. Crítico","Recuperados","Mortos"]
            df_ = pd.DataFrame(soln, columns = names)
            df_['index'] = tvec - delay
            df_ = pd.melt(df_,id_vars = ['index'], var_name='Tipo', value_name='População')
            fig = px.line(df_[~df_['Tipo'].isin(['Sucetíveis','Recuperados','Mortos'])], x="index", y='População', color = 'Tipo')
            
            st.title('Progressão natural do COVID-19')
            st.plotly_chart(fig)
            data_aux = dados_casos['index_aux'].max()
            
            dados_casos['Tempo (dias)'] = dados_casos['index_aux'] + delay
            dados_casos['Sim'] = 'Real'
            dados_casos['Inf. Leve'] = dados_casos['Domiciliar']
            dados_casos['Inf. Grave'] = dados_casos['Enfermaria']
            dados_casos['Inf. Crítico'] = dados_casos['UTI']
            dados_casos['Mortos'] = dados_casos['Obitos']
            
            T = dados_casos['index'].max()
            tvec=np.arange(0,T+10,1)
            soln = odeint(seir,pop,tvec,args=(a0,g0,g1,g2,g3,p1,p2,u,b0,b1,b2,b3,f))
                
            names = ["Sucetíveis","Expostos","Assintomáticos","Inf. Leve","Inf. Grave","Inf. Crítico","Recuperados","Mortos"]
            df_ = pd.DataFrame(soln, columns = names)[['Inf. Leve', 'Inf. Grave','Inf. Crítico','Mortos']]
            df_['Tempo (dias)'] = tvec
            df_['Sim'] = 'Regressão'
            dados_casos['Tempo (dias)'] = dados_casos['index_aux']
            df_['Tempo (dias)'] = df_['Tempo (dias)'] - delay
            df_ = df_[(df_['Tempo (dias)'] >= 0) & (df_['Tempo (dias)'] < dados_casos['Tempo (dias)'].max() + 10)]
            
            st.subheader("Regressão de casos leves:")
            df_ = df_.append(dados_casos[['Tempo (dias)','Inf. Leve','Inf. Grave','Inf. Crítico','Mortos','Sim']])
            fig = px.line(df_, x="Tempo (dias)", y='Inf. Leve', color = 'Sim')
            st.plotly_chart(fig)

            st.subheader("Regressão de casos graves:")
            
            fig = px.line(df_, x="Tempo (dias)", y='Inf. Grave', color = 'Sim')
            st.plotly_chart(fig)
            
            st.subheader("Regressão de casos críticos:")
            
            fig = px.line(df_, x="Tempo (dias)", y='Inf. Crítico', color = 'Sim')
            st.plotly_chart(fig)
            
            st.subheader("Regresão de mortes")
            
            fig = px.line(df_, x="Tempo (dias)", y='Mortos', color = 'Sim')
            st.plotly_chart(fig)
                
        
        
if __name__ == "__main__":
    main(IncubPeriod)
    

