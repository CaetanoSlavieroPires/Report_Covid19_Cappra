import streamlit as st
import numpy as np, pandas as pd
from scipy.integrate import odeint
from numpy import linalg as LA
import plotly.express as px
from streamlit import caching
from pywaffle import Waffle
import matplotlib.pyplot as plt
import itertools
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
        
IncubPeriod = 0

#Taxa reprodutiva padrão
def taxa_reprodutiva(N, be, b0, b1, b2, b3, p1, p2, g0, g1, g2, g3, a1, u, f):
    
    return N*((be/a1)+f*(b0/g0)+(1-f)*((b1/(p1+g1))+(p1/(p1+g1))*(b2/(p2+g2)+ (p2/(p2+g2))*(b3/(u+g3)))))


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

    
#Simulação com intevenção
def simulacao(TimeStart, TimeEnd, tmax, pop, N, a0, b0, b1, b2 , b3, b0Int, b1Int, b2Int, b3Int, g0, g1, g2, g3, p1, p2, u, names, f,delay):
    TimeStart = TimeStart + delay
    TimeEnd = TimeEnd + delay
    tmax = tmax + delay
    if TimeEnd>TimeStart: #Se há intervenção
            tvec = np.arange(0,TimeStart,1) #A simulação sem intervenção termina em t = TimeStart
            sim_sem_int_1 = odeint(seir,pop,tvec,args=(a0,g0,g1,g2,g3,p1,p2,u,b0,b1,b2,b3,f))
            pop = sim_sem_int_1[-1] #Salva a população atual
           #Criando DataFrame
            df_sim_com_int = pd.DataFrame(sim_sem_int_1, columns = names)
            df_sim_com_int['Tempo (dias)'] = tvec
            df_sim_com_int['Simulação'] = 'Com intervenção'            
                #Simulação após o início da intervenção
            tvec=np.arange(TimeStart,TimeEnd,1)
            sim_com_int = odeint(seir,pop,tvec,args=(a0,g0,g1,g2,g3,p1,p2,u, b0Int, b1Int, b2Int, b3Int,f))
            pop = sim_com_int[-1] #Salva população atual
            #Criando DataFrame
            df_aux = pd.DataFrame(sim_com_int, columns = names)
            df_aux['Tempo (dias)'] = tvec
            df_aux['Simulação'] = 'Com intervenção'
            #Append dataframe
            df_sim_com_int = df_sim_com_int.append(df_aux)
                
            if TimeEnd < tmax: #Se a intervenção termina antes do tempo final
                tvec = np.arange(TimeEnd,tmax,1) #A simulação sem intervenção termina em t = TimeStart
                    #Simulação sem intervenção (após o fim da intervenção)
                sim_sem_int_2 = odeint(seir,pop,tvec,args=(a0,g0,g1,g2,g3,p1,p2,u,b0,b1,b2,b3,f))
                    #Criando dataframe
                df_aux = pd.DataFrame(sim_sem_int_2, columns = names)
                df_aux['Tempo (dias)'] = tvec
                df_aux['Simulação'] = 'Com intervenção'
                    #Append dataframe
                df_sim_com_int = df_sim_com_int.append(df_aux)
                        
            return df_sim_com_int
    

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

def new_growth_rate(g0,g1,g2,g3,p1,p2,b0,b1,b2,b3,u,a0,N,f): #Growth rate após o update
    
    JacobianMat=np.array([
                 [-a0, 0, N*b0, N*b1, N*b2, N*b3, 0, 0],
                 [a0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, -g0, 0, 0, 0, 0, 0],
                 [0, 0, 0, -p1-g1, 0, 0, 0, 0],
                 [0, 0, 0, p1, -p2-g2, 0, 0, 0],
                 [0, 0, 0, 0, p2, -u-g3, 0, 0],
                 [0, 0, g0, g1, g2, g3 , 0, 0],
                 [0, 0, 0, 0, 0, u, 0, 0]
                ])
    
    eig = LA.eig(JacobianMat)
    eigvalue = eig[0].real
    eigvector = eig[1]
    
    r = max(eigvalue)
    
    MaxEigenVector=eigvector.T[np.argmax(eigvalue)]
    MaxEigenVector=MaxEigenVector/MaxEigenVector[len(MaxEigenVector)-1]
    MaxEigenVector=MaxEigenVector.real
    DoublingTime=np.log(2)/r
    
    return r, DoublingTime


def main(IncubPeriod):
    pic = "https://images.squarespace-cdn.com/content/5c4ca9b7cef372b39c3d9aab/1575161958793-CFM6738ESA4DNTKF0SQI/CAPPRA_PRIORITARIO_BRANCO.png?content-type=image%2Fpng"
    st.sidebar.image(pic, use_column_width=False, width=100, caption=None)
    
    #Definindo valores padrões dos parâmetros
    IncubPeriod = 5
    DurMildInf = 6
    FracSevere = 0.15
    FracCritical = 0.05
    FracMild = 1 - FracSevere - FracCritical
    ProbDeath = 0.5
    TimeICUDeath = 8
    DurHosp = 6
    tmax = 365
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
    
    tmax = st.sidebar.slider("Tempo máximo da simulação em dias", min_value=0, max_value=1000, value=tmax, step=1)

    TimeStart = st.sidebar.slider(label="Tempo de início da intervenção (dias)",min_value = 0, max_value = tmax, value = TimeStart, step = 1) 
        #Fim da intervenção
    TimeEnd = st.sidebar.slider(label="Tempo de fim da intervenção (dias)", min_value = 0, max_value = tmax, value = TimeEnd, step = 1) 
    #Redução da transmissão de assintomáticos
    reducasym = st.sidebar.slider("Redução na transmissão causada por infecções assintomáticas (%)", min_value=0, max_value=100, value=int(reducasym*100), step = 1)/100
        #Taxa de transmissão (infecções leves)
    reduc1 = st.sidebar.slider("Redução na transmissão causada por infecções leves (%)", min_value=0, max_value=100, value=int(reduc1*100), step=1)/100   
        #Taxa de transmissão (infecções graves, relativa a infecção leve)
    reduc2 = st.sidebar.slider("Redução na transmissão causada por infecções graves (%)", min_value=0, max_value=100, value=int(reduc2*100), step=1)/100 
        #Taxa de transmissão (infecções críticas, relativa a infecção leve)
    reduc3 = st.sidebar.slider("Redução na transmissão causada por infecções críticas (%)", min_value=0, max_value=100, value=int(reduc3*100), step=1)/100

    
    st.title("Report da simulação do COVID-19")
    dados = pd.read_csv('dados_cidades.csv',encoding = "ISO-8859-1")
    cidade = st.selectbox("Selecione a cidade", list(dados['Cidade']))
    parametros = pd.read_csv('parametros_cidades.csv')
    st.table(parametros)
    parametros = parametros[parametros['Cidade'] == cidade].sort_values('rmse')
    st.table(parametros)
    dados = dados.set_index('Cidade')
    dados['População'] = dados['População']#.apply(lambda x: ''.join(x.split('.')))
    N = int(dados.loc[cidade,'População'])
    st.table(pd.DataFrame(dados.loc[cidade,:].to_dict(), index = ['']))
    page = 'Report'
    if page == 'Report':
        dados_casos = pd.read_csv('dados_cidades_2.csv')
        dados_casos = dados_casos[dados_casos['Cidade']==cidade].reset_index(drop = True).reset_index()  
        st.subheader('Casos confirmados da cidade')
        
        dados_casos['index_aux'] = dados_casos['index']
        dados_casos['Sim'] = 'REAL'
        dados_casos['Infectados'] = dados_casos['UTI'] + dados_casos['Ativos']
        fig = px.line(dados_casos, x="Data", y='Infectados')
        st.plotly_chart(fig)

        dados_casos['Infectados'] = dados_casos['UTI']
        st.subheader('Casos confirmados e internados na UTI')
        fig = px.line(dados_casos, x="Data", y='Infectados')
        st.plotly_chart(fig)
        
        b2 = b2/N
        b3 = b2/N
        
        E = parametros.iloc[0,0]
        b0 = parametros.iloc[0,1]/N
        b1 = parametros.iloc[0,2]/N
        f = parametros.iloc[0,3]
        FracSevere = parametros.iloc[0,4]
        FracCritical = parametros.iloc[0,5]
        FracMild = 1 - FracSevere - FracCritical
        ProbDeath = parametros.iloc[0,7]
        CFR = FracCritical*ProbDeath
        
        a0, u, g0, g1, g2, g3, p1, p2, f, ic = params(IncubPeriod, FracMild, FracCritical, FracSevere, TimeICUDeath, CFR, DurMildInf, DurHosp, i, FracAsym, DurAsym, N)
        
        pop = np.zeros(8)
        pop[0] = N - E
        pop[1] = E
        
        delay = parametros.iloc[0,6]
        
        tvec = np.arange(0,tmax+delay,1)
        soln = odeint(seir,pop,tvec,args=(a0,g0,g1,g2,g3,p1,p2,u,b0,b1,b2,b3,f))
        names = ["Sucetíveis","Expostos","Assintomáticos","Inf. Leve","Inf. Grave","Inf. Crítico","Recuperados","Mortos"]
        df_ = pd.DataFrame(soln, columns = names)
        df_['index'] = tvec - delay
        df_ = pd.melt(df_,id_vars = ['index'], var_name='Tipo', value_name='População')
        fig = px.line(df_[~df_['Tipo'].isin(['Sucetíveis','Recuperados','Mortos'])], x="index", y='População', color = 'Tipo')
        
        st.title('Progressão natural do COVID-19')
        st.plotly_chart(fig)
        data_aux = dados_casos['index_aux'].max()
        st.subheader('Estimativa para daqui 7 dias:')
        data = (df_[(df_['Tipo'].isin(["Sucetíveis","Expostos",'Recuperados','Mortos','Inf. Crítico','Inf. Grave','Assintomáticos', 'Inf. Leve'])) & (df_['index'] == data_aux + 7)][['Tipo','População']].set_index('Tipo')/N*100)
        data = data.round(3).to_dict()['População']
        st.table(pd.DataFrame(data, index = ['%']))

        fig = plt.figure(
            FigureClass=Waffle, 
            rows=5,
            columns = 20,
            values=data, 
            colors=("#58c736","#a4de26","#0b2fe3","#f5c842","#fc9403","#f73c02","#07e8f0","#ed07bf"),
            title={'label': '', 'loc': 'left'},
            labels=["{0} ({1}%)".format(k, v) for k, v in data.items()],
            legend={'loc': 'lower left', 'bbox_to_anchor': (0, -0.2),'ncol':3, 'framealpha': 10},
            icons='user', icon_size=45, 
            icon_legend=True,
            figsize=(15, 6)
        )
        fig.gca().set_facecolor('#EEEEEE')
        fig.set_facecolor('#EEEEEE')
        plt.show()
        st.pyplot()
        
        st.subheader('Estimativa para daqui 30 dias:')
        data = (df_[(df_['Tipo'].isin(["Sucetíveis","Expostos",'Recuperados','Mortos','Inf. Crítico','Inf. Grave','Assintomáticos', 'Inf. Leve'])) & (df_['index'] == data_aux + 30)][['Tipo','População']].set_index('Tipo')/N*100)
        data = data.round(3).to_dict()['População']
        st.table(pd.DataFrame(data, index = ['%']))
        fig = plt.figure(
            FigureClass=Waffle, 
            rows=5,
            columns = 20,
            values=data, 
            colors=("#58c736","#a4de26","#0b2fe3","#f5c842","#fc9403","#f73c02","#07e8f0","#ed07bf"),
            title={'label': '', 'loc': 'left'},
            labels=["{0} ({1}%)".format(k, v) for k, v in data.items()],
            legend={'loc': 'lower left', 'bbox_to_anchor': (0, -0.2),'ncol':3, 'framealpha':10},
            icons='user', icon_size=45, 
            icon_legend=True,
            figsize=(15, 6)
        )
        fig.gca().set_facecolor('#EEEEEE')
        fig.set_facecolor('#EEEEEE')
        plt.show()
        st.pyplot()
        
        st.subheader('Estimativa para o fim da simulação:')
        data = (df_[(df_['Tipo'].isin(["Sucetíveis",'Recuperados','Mortos'])) & (df_['index'] == df_['index'].max())][['Tipo','População']].set_index('Tipo')/N*100)
        data = data.round(3).to_dict()['População']
        st.table(pd.DataFrame(data, index = ['%']))
        fig = plt.figure(
            FigureClass=Waffle, 
            rows=5,
            columns = 20,
            values=data, 
            colors=("#58c736", "#34c7c5", "#c734ae"),
            title={'label': '', 'loc': 'left'},
            labels=["{0} ({1}%)".format(k, v) for k, v in data.items()],
            legend={'loc': 'lower left', 'bbox_to_anchor': (0, -0.2), 'ncol': 3, 'framealpha': 10},
            icons='user', icon_size=30, 
            icon_legend=True,
            figsize=(10, 3)
        )
        fig.gca().set_facecolor('#EEEEEE')
        fig.set_facecolor('#EEEEEE')
        plt.show()
        st.pyplot()
        
        dados_casos['Tempo (dias)'] = dados_casos['index_aux'] + delay
        dados_casos['Sim'] = 'Real'
        dados_casos['Inf. Grave'] = dados_casos['Ativos']
        dados_casos['Inf. Crítico'] = dados_casos['UTI']
        dados_casos['Mortos'] = dados_casos['Obitos']
        
        T = dados_casos['index'].max()
        tvec=np.arange(0,T+10,1)
        soln = odeint(seir,pop,tvec,args=(a0,g0,g1,g2,g3,p1,p2,u,b0,b1,b2,b3,f))
        st.write(soln)    
        
	names = ["Sucetíveis","Expostos","Assintomáticos","Inf. Leve","Inf. Grave","Inf. Crítico","Recuperados","Mortos"]
        df_ = pd.DataFrame(soln, columns = names)[['Inf. Grave','Inf. Crítico','Mortos']]
        df_['Tempo (dias)'] = tvec
        df_['Sim'] = 'Regressão'
        dados_casos['Tempo (dias)'] = dados_casos['index_aux']
        df_['Tempo (dias)'] = df_['Tempo (dias)'] - delay
        df_ = df_[(df_['Tempo (dias)'] >= 0) & (df_['Tempo (dias)'] < dados_casos['Tempo (dias)'].max() + 10)]
        st.subheader("Regressão de casos graves:")
        
        df_ = df_.append(dados_casos[['Tempo (dias)','Inf. Grave','Inf. Crítico','Mortos','Sim']])
        fig = px.line(df_, x="Tempo (dias)", y='Inf. Grave', color = 'Sim')
        st.plotly_chart(fig)
        
        st.subheader("Regressão de casos críticos:")
        
        fig = px.line(df_, x="Tempo (dias)", y='Inf. Crítico', color = 'Sim')
        st.plotly_chart(fig)
        
        st.subheader("Regresão de mortes")
        
        fig = px.line(df_, x="Tempo (dias)", y='Mortos', color = 'Sim')
        st.plotly_chart(fig)
                
        
        st.title("Capacidade do sistema de saúde e medidas de intervenção")

        AvailHospBeds= int(dados.loc[cidade,'Leitos Hospitalares Adulto'])
        AvailICUBeds=int(dados.loc[cidade,'Leitos UTI Adulto'])
        ConvVentCap=int(dados.loc[cidade,'Número de Respiradores'])

        #Calculo das taxas de transmissão durante a intervenção
        b1Int = b1*(1 - reduc1)
        b2Int = b2*(1 - reduc2)
        b3Int = b3*(1 - reduc3)
        b0Int = b0*(1 - reducasym)

        names = ["Sucetíveis","Expostos","Assintomáticos","Inf. Leve","Inf. Grave","Inf. Crítico","Recuperados","Mortos"]

#########  Simulação sem intervenção #########################################################
        tvec=np.arange(0,tmax+delay,1)
        sim_sem_int = odeint(seir,pop,tvec,args=(a0,g0,g1,g2,g3,p1,p2,u,b0,b1,b2,b3,f))
        #Criando dataframe
        df_sim_sem_int = pd.DataFrame(sim_sem_int, columns = names)
        df_sim_sem_int['Tempo (dias)'] = tvec
        df_sim_sem_int['Tempo (dias)'] = df_sim_sem_int['Tempo (dias)'] - delay
        df_sim_sem_int['Simulação'] = 'Sem intervenção'
#############################################################################################
        
        #Simulação com intervenção
        df_sim_com_int = simulacao(TimeStart, TimeEnd, tmax, pop, N, a0, b0, b1, b2 , b3, b0Int, b1Int, b2Int, b3Int, g0, g1, g2, g3, p1, p2, u, names, f, delay)
        df_sim_com_int['Tempo (dias)'] = df_sim_com_int['Tempo (dias)'] - delay
        y_index = 'Número de pessoas'  

            
        st.subheader('Casos graves e críticos vs Leitos de hospital + UTI')
        df_sim_com_int[y_index] = df_sim_com_int["Inf. Grave"] + df_sim_com_int["Inf. Crítico"]
        df_sim_sem_int[y_index] = df_sim_sem_int["Inf. Grave"] + df_sim_sem_int["Inf. Crítico"]
        df = df_sim_sem_int[['Tempo (dias)',y_index, 'Simulação']].append(df_sim_com_int[['Tempo (dias)',y_index, 'Simulação']])
####################################################################################################

        #Gráficos da capacidade hospitalar################################################################    
    
        data1 = []
        for x in range(0, tmax):
            data1.append([x,'Leitos hospitalares + UTI',AvailHospBeds + AvailICUBeds])
                
        df = df.append(pd.DataFrame(data1, columns = ['Tempo (dias)','Simulação',y_index]))
        fig = px.line(df, x="Tempo (dias)", y=y_index, color = 'Simulação')
        st.plotly_chart(fig)
            

        st.subheader('Infecções críticas vs Leitos na UTI')
        df_sim_com_int[y_index] = df_sim_com_int["Inf. Crítico"]
        df_sim_sem_int[y_index] = df_sim_sem_int["Inf. Crítico"]
        df = df_sim_sem_int[['Tempo (dias)',y_index, 'Simulação']].append(df_sim_com_int[['Tempo (dias)',y_index, 'Simulação']])
            
        data1 = []
        for x in range(0, tmax):
            data1.append([x,'Leitos da UTI',AvailICUBeds])
                
        df = df.append(pd.DataFrame(data1, columns = ['Tempo (dias)','Simulação',y_index]))
        fig = px.line(df, x="Tempo (dias)", y=y_index, color = 'Simulação')
        st.plotly_chart(fig)
        
        st.subheader('Infecções críticas vs Número de respiradores')
        df_sim_com_int[y_index] = df_sim_com_int["Inf. Crítico"]
        df_sim_sem_int[y_index] = df_sim_sem_int["Inf. Crítico"]
        df = df_sim_sem_int[['Tempo (dias)',y_index, 'Simulação']].append(df_sim_com_int[['Tempo (dias)',y_index, 'Simulação']])
            
        data1 = []
        data2 = []
        data3 = []
        for x in range(0, tmax):
            data1.append([x,'Respiradores',ConvVentCap])
                #data2.append([x,'Ventilação em protocolo de convenção',ContVentCap])
                #data3.append([x,'Ventilação em protocolo de crise',CrisisVentCap])
                
        df = df.append(pd.DataFrame(data1, columns = ['Tempo (dias)','Simulação',y_index]))
            #df = df.append(pd.DataFrame(data2, columns = ['Tempo (dias)','Simulação',y_index]))
            #df = df.append(pd.DataFrame(data3, columns = ['Tempo (dias)','Simulação',y_index]))
            
        fig = px.line(df, x="Tempo (dias)", y=y_index, color = 'Simulação')
        st.plotly_chart(fig)
        
        st.subheader("Data do colapso do sistema de saúde após o primeiro caso")
        df_sim_sem_int['Soma'] = df_sim_sem_int['Inf. Grave'] + df_sim_sem_int['Inf. Crítico']
        dict_sem = {
            'Leitos hospitalares':int(df_sim_sem_int[df_sim_sem_int['Inf. Grave'] > AvailHospBeds]['Tempo (dias)'].head(1).to_list()[0]),
            'Leitos de UTI':int(df_sim_sem_int[df_sim_sem_int['Inf. Crítico'] > AvailICUBeds]['Tempo (dias)'].head(1).to_list()[0]),
            'Respiradores':int(df_sim_sem_int[df_sim_sem_int['Inf. Crítico'] > ConvVentCap]['Tempo (dias)'].head(1).to_list()[0])
        }
        try:
            lh = int(df_sim_com_int[df_sim_com_int['Inf. Grave'] > AvailHospBeds]['Tempo (dias)'].head(1).to_list()[0])
        except: 
            lh = 'Não houve colapso'
            
        try:
            lu = int(df_sim_com_int[df_sim_com_int['Inf. Crítico'] > AvailICUBeds]['Tempo (dias)'].head(1).to_list()[0])
        except:
            lu = 'Não houve colapso'
            
        try:
            r = int(df_sim_com_int[df_sim_com_int['Inf. Crítico'] > ConvVentCap]['Tempo (dias)'].head(1).to_list()[0])
        except:
            r = 'Não houve colapso'
            
        dict_com = {
                'Leitos hospitalares': lh,
                'Leitos de UTI':lu,
                'Respiradores':r
            }

        
        st.table(pd.DataFrame(dict_sem, index = ['Sem interveção']).append(pd.DataFrame(dict_com, index = ['Com intervenção'])))
        
        st.title('Auge de infectados')
        
        lista_1=[[int(df_sim_sem_int[df_sim_sem_int['Inf. Leve'] == df_sim_sem_int['Inf. Leve'].max()]['Tempo (dias)'].to_list()[0]),int(df_sim_sem_int['Inf. Leve'].max())],
        [int(df_sim_sem_int[df_sim_sem_int['Inf. Grave'] == df_sim_sem_int['Inf. Grave'].max()]['Tempo (dias)'].to_list()[0]),int(df_sim_sem_int['Inf. Grave'].max())],
        [int(df_sim_sem_int[df_sim_sem_int['Inf. Crítico'] == df_sim_sem_int['Inf. Crítico'].max()]['Tempo (dias)'].to_list()[0]),int(df_sim_sem_int['Inf. Crítico'].max())]
        ]
        
        st.subheader('Sem intervenção')
        st.table(pd.DataFrame(lista_1, columns = ['Dias após primeiro caso','Quantidade de pessoas'],index = ['Casos leves','Casos graves','Casos críticos']))
        
        st.subheader('Com intervenção')
        try:
            lista_2=[[int(df_sim_com_int[df_sim_com_int['Inf. Leve'] == df_sim_com_int['Inf. Leve'].max()]['Tempo (dias)'].to_list()[0]),int(df_sim_com_int['Inf. Leve'].max())],
            [int(df_sim_com_int[df_sim_com_int['Inf. Grave'] == df_sim_com_int['Inf. Grave'].max()]['Tempo (dias)'].to_list()[0]),int(df_sim_com_int['Inf. Grave'].max())],
            [int(df_sim_com_int[df_sim_com_int['Inf. Crítico'] == df_sim_com_int['Inf. Crítico'].max()]['Tempo (dias)'].to_list()[0]),int(df_sim_com_int['Inf. Crítico'].max())]
            ]


            st.table(pd.DataFrame(lista_2, columns = ['Dias após primeiro caso','Quantidade de pessoas'],index = ['Casos leves','Casos graves','Casos críticos']))
        except:
            pass
        st.title('Parâmetros utilizados:')
        st.write('População inicial: ', N)
        st.write('População inicial exposta ao vírus: ', E)
        st.write('Periodo de incubação: ',IncubPeriod)
        st.write('Duração de infecções leves: ',DurMildInf)
        st.write('Duração de infecções graves: ',DurHosp)
        st.write('Duração de infecções críticas:', TimeICUDeath)
        st.write('Duração de infecções assintomáticas: ',DurAsym)
        st.write('Fração de infecções assintomáticas: ',f)
        st.write('Fração de infecções sintomáticas leves: ',round(FracMild*(1 - f),3))
        st.write('Fração de infecções sintomáticas graves:',round(FracSevere*(1 - f),3))
        st.write('Fração de infecções sintomáticas críticas: ',round(FracCritical*(1 - f),3))
        st.write('Taxa de mortalidade de infecções críticas: ',ProbDeath)
        st.write('Taxa de mortalidade geral: ',round(ProbDeath*FracCritical*(1 - f),3))
        st.write('Taxa de propagação de infecções assintomáticas: ',round(b0*N,3))
        st.write('Taxa de propagação de infecções leves: ',round(b1*N,3))
        st.write('Taxa de propagação de infecções graves: ',round(b2*N,3))
        st.write('Taxa de propagação de infecções críticas: ',round(b3*N,3))
        st.subheader('10 melhores familias de parâmetros que prevêem o modelo')
        #erro = erro.sort_values('rmse').head(20).reset_index(drop = True)
        #erro.columns = ['Expostos iniciais','Taxa de infecção assintomática','Taxa de infecção leve','Fração de assintomáticos','Erro RMSE','Delay dos dados reais']
        st.table(parametros)

if __name__ == "__main__":
    main(IncubPeriod)
    

