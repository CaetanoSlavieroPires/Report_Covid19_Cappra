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
        
        return a0, u, g0, g1, g2, g3, p1, p2, f
#Menu dos parâmetros gerais

    
#Simulação com intevenção
def simulacao(TimeStart, TimeEnd, tmax, pop, N, a0, b0, b1, b2 , b3, b0Int, b1Int, b2Int, b3Int, g0, g1, g2, g3, p1, p2, u, names, f,delay):
    TimeStart = TimeStart + delay
    TimeEnd = TimeEnd + delay
    tmax = tmax + delay
    if TimeEnd>TimeStart: #Se há intervenção
            tvec = np.arange(0,TimeStart+1,1) #A simulação sem intervenção termina em t = TimeStart
            sim_sem_int_1 = odeint(seir,pop,tvec,args=(a0,g0,g1,g2,g3,p1,p2,u,b0,b1,b2,b3,f))
            pop = sim_sem_int_1[-1] #Salva a população atual
           #Criando DataFrame
            df_sim_com_int = pd.DataFrame(sim_sem_int_1, columns = names)
            df_sim_com_int['Tempo (dias)'] = tvec
            df_sim_com_int['Simulação'] = 'Com intervenção'            
                #Simulação após o início da intervenção
            tvec=np.arange(TimeStart,TimeEnd+1,1)
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


def main(IncubPeriod):
    pic = "https://images.squarespace-cdn.com/content/5c4ca9b7cef372b39c3d9aab/1575161958793-CFM6738ESA4DNTKF0SQI/CAPPRA_PRIORITARIO_BRANCO.png?content-type=image%2Fpng"
    st.image(pic, use_column_width=False, width=100, caption=None)
    
    #Definindo valores padrões dos parâmetros
    IncubPeriod = 5
    DurMildInf = 6
    TimeICUDeath = 8
    DurHosp = 6
    DurAsym = 6
    tmax = 365
    i = 1
    TimeEnd = tmax
    b2 = 0.1
    b3 = 0.1
    reduc1 = 0.5
    reduc2 = 0.5
    reduc3 = 0.5
    reducasym = 0.5
    
    st.title("Report da simulação do COVID-19")
    st.write("Esse simulador desenvolvido pela [Cappra Institute for Data Science](https://www.cappra.institute/) utiliza o modelo epidêmico SEIR desenvolvido pela cientista [Alison Hill](https://alhill.shinyapps.io/COVID19seir/) para modelar o crescimento do CODIV-19 nas cidades brasileiras, utilizando dados de casos graves, críticos e óbitos divulgados pelas secretarias de saúde.")
    st.write("A subnotificação de casos e o baixo número de testes na população tem sido um problema para realizar a modelagem, obrigando os pesquisadores a fazerem extrapolações de casos para estimar o crescimento do vírus no Brasil. Tendo isso em vista, o modelo é parametrizado utilizando as notificações de casos hospitalizados, internados em UTI e óbitos, por serem os dados mais realistas divulgados pelas secretarias de saude. Com isso, podemos fazer extrapolações de diversos cenários da propagação do vírus, prevendo o que poderá acontecer com o sistema de saúde com e sem ações de distanciamento social e estimando a duração da propagação do vírus em nossas cidades.")
    st.write("Poder realizar essa simulação com qualidade depende inteiramente de uma divulgação transparente e bem estruturada dos dados do COVID-19 e seus resultados são apenas tentativas de se aproximar da realidade do tamanho da infecção no Brasil. Como todo modelo, deve ser contextualizado e não deve ser levado como regra.")
    st.write("Mais análises do COVID-19 feitas pelxs cientistas da Cappra podem ser encontradas em [https://www.cappra.co/lab-covid](https://www.cappra.co/lab-covid).")
    dados = pd.read_csv('dados_cidades.csv', encoding = "ISO-8859-1")
    st.subheader("Selecione a cidade:")
    cidade = st.selectbox("", ['Porto Alegre','São Luís','Curitiba','Passo Fundo','Jaraguá do Sul','Caxias do Sul'])
    if cidade != 'Modelar cidade':
        parametros = pd.read_csv('parametros_cidades.csv', encoding = "ISO-8859-1")
        parametros = parametros[parametros['Cidade'] == cidade].sort_values('rmse')
        dados = dados.set_index('Cidade')
        N = int(dados.loc[cidade,'População'])
        st.table(pd.DataFrame(dados.loc[cidade,:].to_dict(), index = ['']))
    page = 'Report'
    if page == 'Report':
        if cidade == 'Modelar cidade':
            pass#model_city(tmax,TimeStart,TimeEnd,reduc1,reduc2,reduc3,reducasym)
        else:   
            st.title("Monitoramento do COVID-19 em " + cidade)
            dados_casos = pd.read_csv(cidade + '.csv', encoding = "ISO-8859-1")
            dados_casos = dados_casos.reset_index(drop = True).reset_index()  
            slot1 = st.empty()
            st.subheader('Casos registrados em ' + cidade )
            dados_casos['index_aux'] = dados_casos['index']
            dados_casos['Sim'] = 'REAL'
            dados_casos['Casos Log'] = np.log(dados_casos['Casos'])
            dados_casos_aux = dados_casos[['Casos','Recuperados','Obitos']].tail(1)
            dados_casos_aux.index = ['']
            slot1.table(dados_casos_aux)
            fig = px.line(dados_casos, x="Data", y='Casos')
            st.plotly_chart(fig)
            st.subheader('Casos registrados em ' + cidade + ' em escada logarítmica')
            fig = px.line(dados_casos, x="Data", y='Casos Log')
            st.plotly_chart(fig)

            dados_casos['Infectados'] = dados_casos['UTI']
            st.subheader('Casos ativos e internados na UTI em ' + cidade)
            fig = px.line(dados_casos, x="Data", y='Infectados')
            st.plotly_chart(fig)

            st.subheader('Quantidade de testes realizados vs testes negativos')
            dados_casos_aux = dados_casos[['Data','Testes','Negativos']]
            dados_casos_aux = pd.melt(dados_casos_aux,id_vars = ['Data'], var_name='Tipo', value_name='Quantidade')
            fig = px.line(dados_casos_aux, x='Data',y = 'Quantidade', color = 'Tipo')
            st.plotly_chart(fig)
            
            dados_casos['Sim'] = 'Real'
            dados_casos['Inf. Grave'] = dados_casos['Enfermaria']
            dados_casos['Inf. Crítico'] = dados_casos['UTI']
            dados_casos['Mortos'] = dados_casos['Obitos']
            dados_casos['Tempo (dias)'] = dados_casos['index_aux']

            parametros = parametros.sort_values('id')

            b2 = b2/N
            b3 = b2/N
            df = pd.DataFrame()
            for n in parametros['id']:
                E = parametros.iloc[n,0]
                reduc = parametros.iloc[n,10]
                b0 = parametros.iloc[n,1]/N*(1 - reduc)
                b1 = parametros.iloc[n,2]/N*(1 - reduc)
                FracAsym = parametros.iloc[n,3]
                FracSevere = parametros.iloc[n,4]
                FracCritical = parametros.iloc[n,5]
                FracMild = 1 - FracSevere - FracCritical
                delay = parametros.iloc[n,6]
                ProbDeath = parametros.iloc[n,7]
                TimeStart = parametros.iloc[n,11]
                CFR = FracCritical*ProbDeath
                tmax = parametros.iloc[n,14]
                descr = parametros.iloc[n,13]
                
                a0, u, g0, g1, g2, g3, p1, p2, f = params(IncubPeriod, FracMild, FracCritical, FracSevere, TimeICUDeath, CFR, DurMildInf, DurHosp, i, FracAsym, DurAsym, N)
                
                if descr == "Sem distanciamento social":
                    pop = np.zeros(8)
                    pop[0] = N - E
                    pop[1] = E
                    tvec = np.arange(TimeStart,tmax+delay,1)

                elif descr == "Com distanciamento social":
                    if n == 1:
                        pop = soln[TimeStart + delay]
                    else:
                        pop = soln[-1]
                    tvec = np.arange(TimeStart,tmax,1)
                    T_2 = TimeStart

                else:
                    pop = soln_aux[dados_casos['index_aux'].max() - T_2]
                    tvec = np.arange(dados_casos['index_aux'].max(),tmax,1)

               
                soln = odeint(seir,pop,tvec,args=(a0,g0,g1,g2,g3,p1,p2,u,b0,b1,b2,b3,f))
                if descr == "Com distanciamento social":
                    soln_aux = soln

                names = ["Sucetíveis","Expostos","Assintomáticos","Inf. Leve","Inf. Grave","Inf. Crítico","Recuperados","Mortos"]
            
                df_ = pd.DataFrame(soln, columns = names)
                if descr == "Sem distanciamento social":
                    df_['Tempo (dias)'] = tvec - delay
                else:
                    df_['Tempo (dias)'] = tvec

                df_['Sim'] = descr
                df = df.append(df_)

            df = df.append(dados_casos[['Tempo (dias)','Inf. Grave','Inf. Crítico','Mortos','Sim']])
            
            df_ = df[(df['Tempo (dias)'] >= 0) & (df['Tempo (dias)'] < dados_casos['Tempo (dias)'].max() + 20)].copy(deep = True)

            st.title('Curvas de crescimento do COVID-19 em '+ cidade + ' com e sem distanciamento')
            st.write('Comparamos os dados reais com os dados da simulação e aplicamos medidas de intervenção de distanciamento social na simulação.')
            st.write('É importante comparar cenários onde há distanciamento social e onde não há. Dessa forma podemos saber o impacto das medidas de distanciamento.')
            st.subheader('Casos reais versus simulação')
            st.write("Abaixo é possível visualizar as curvas de casos divulgados, separados em críticos e óbitos, que são utilizadas para parametrizar o modelo, e as curvas da simulação sem intervenção e com intervenção, utilizadas para monitorar o crescimento do vírus.")
            
            st.subheader("Regressão de casos críticos:")
            st.write("Casos críticos são casos que devem ser priorizados para internação em UTI e utilizar respiradores.")
            fig = px.line(df_, x="Tempo (dias)", y='Inf. Crítico', color = 'Sim')
            max_ = dados_casos['Inf. Crítico'].max() + 100
            fig.update_layout(yaxis=dict(range=[-5, max_]))

            st.plotly_chart(fig)
            max_ = dados_casos['Obitos'].max() + 100
            st.subheader("Regressão de mortes")
            
            fig = px.line(df_, x="Tempo (dias)", y='Mortos', color = 'Sim')
            fig.update_layout(yaxis=dict(range=[-5, max_]))

            st.plotly_chart(fig)
                    
            
            st.title("Capacidade do sistema de saúde e medidas de intervenção")
            st.write("Um dos principais objetivos da simulação é comparar o cenário atual de crescimento do vírus com um cenário onde há intervenções que reduzam a propagação do vírus, como distanciamento social, quarentena, uso de máscaras, entre outras medidas possíveis. Com reduções na transmissão do vírus, podemos encontrar cenários onde nosso sistema saúde pode ser capaz de lidar com o vírus sem o colapso e esgotamento de leitos hospitalares, de UTI ou respiradores ou cenários piores, onde somos capaz de reduzir a transmissão mas não o suficiente para evitar o colapso do sistema de saúde, necessitando a ampliação deste.")
            st.write("Nessa simulação está sendo considerada a quantidade total de leitos hospitalares e UTI para covid-19 e a quantidade máxima de respiradores disponíveis.")

            AvailHospBeds= int(dados.loc[cidade,'Leitos Hospitalares Adulto'])
            AvailICUBeds=int(dados.loc[cidade,'Leitos UTI Adulto'])
            ConvVentCap=int(dados.loc[cidade,'Número de Respiradores'])
                
            st.subheader('Infecções críticas vs Leitos na UTI')
            df_ = df.copy(deep = True)
            df_['Número de Pessoas'] = df_['Inf. Crítico']
                
            data1 = []
            for x in range(0, tmax):
                data1.append([x,'Leitos da UTI',AvailICUBeds])
                    
            df_ = df_.append(pd.DataFrame(data1, columns = ['Tempo (dias)','Sim','Número de Pessoas']))
            fig = px.line(df_[df_['Tempo (dias)'] >=0], x="Tempo (dias)", y='Número de Pessoas', color = 'Sim')
            max_ = (AvailICUBeds)*2
            fig.update_layout(yaxis=dict(range=[-5, max_]))
            st.plotly_chart(fig)
            
            st.subheader('Infecções críticas vs Número de respiradores')
            df_ = df.copy(deep = True) 
            df_['Número de Pessoas'] = df_['Inf. Crítico'] 
            data1 = []
            for x in range(0, tmax):
                data1.append([x,'Respiradores',ConvVentCap])
         
            df_ = df_.append(pd.DataFrame(data1, columns = ['Tempo (dias)','Sim','Número de Pessoas']))
            
            fig = px.line(df_[df_['Tempo (dias)'] >=0], x="Tempo (dias)", y='Número de Pessoas', color = 'Sim')
            max_ = (ConvVentCap)*2
            fig.update_layout(yaxis=dict(range=[-5, max_]))
            st.plotly_chart(fig)

            df_sim_sem_int = df[df['Sim'] == 'Com o fim do distanciamento social']
            df_sim_com_int = df[df['Sim'] == 'Com distanciamento social']
            st.subheader("Data do colapso do sistema de saúde após o primeiro caso")
            st.write("Comparação do tempo da data de lotação do sistema hospitalar considerando o cenário atual de crescimento do vírus versus o cenário com intervenção.")
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
            st.subheader('Data e quantidade de casos do auge da infecção para cada tipo de infecção sintomática')
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

            df__ = df[df['Sim'] == 'Com o fim do distanciamento social'].copy(deep = True)
            df__ = pd.melt(df__,id_vars = ['Tempo (dias)'], var_name='Tipo', value_name='Número de Pessoas')
            fig = px.line(df__[(~df__['Tipo'].isin(['Sucetíveis','Recuperados','Mortos'])) & (df__['Tempo (dias)'] >= 0)], x="Tempo (dias)", y='Número de Pessoas', color = 'Tipo')
            fig_2 = px.line(df__[(df__['Tipo'].isin(['Sucetíveis','Recuperados','Mortos'])) & (df__['Tempo (dias)'] >= 0)], x="Tempo (dias)", y='Número de Pessoas', color = 'Tipo')
            st.title('Simulação da progressão natural do COVID-19 em ' + cidade + ' se houver o fim do distanciamento social')
            st.write('O modelo é parametrizado baseado na quantidade ativa de casos graves (pacientes hospitalizados), casos críticos (pacientes internados em UTI) e óbitos. Sabemos sobre a subnotificação causada pela baixa quantidade de testes, mas para realizar uma extrapolação de casos leves, assintomáticos e expostos ao vírus, utilizamos dados de hospitalizados, internados em UTI e óbitos para minimizar o erro do modelo.')
            st.subheader('Curvas de infectados do modelo parametrizado:')
            st.plotly_chart(fig)
            st.subheader('Curvas de sucetíveis, recuperados e mortos do modelo parametrizado:')
            st.plotly_chart(fig_2)

            data_aux = dados_casos['index_aux'].max()
            st.title('Estimativa para a progressão do COVID-19 em ' + cidade + ' se houver o fim do distanciamento social')
            st.subheader('Visualizando como estará distribuída a infecção em toda a população da cidade')
            st.subheader('Estimativa para daqui 7 dias:')
            data = (df__[(df__['Tipo'].isin(["Sucetíveis","Expostos",'Recuperados','Mortos','Inf. Crítico','Inf. Grave','Assintomáticos', 'Inf. Leve'])) & (df__['Tempo (dias)'] == data_aux + 7)][['Tipo','Número de Pessoas']].set_index('Tipo')/N*100)
            df_aux = data.copy(deep = True)
            df_aux['Número de Pessoas'] = df_aux['Número de Pessoas'].apply(lambda x: str(round(x,2)) + '%')
            data = data.to_dict()['Número de Pessoas']
            st.table(pd.DataFrame(df_aux.to_dict()['Número de Pessoas'], index = ['Porcentagem da população']))

            fig = plt.figure(
                FigureClass=Waffle, 
                rows=5,
                columns = 20,
                values=data, 
                colors=("#58c736","#a4de26","#0b2fe3","#f5c842","#fc9403","#f73c02","#07e8f0","#ed07bf"),
                title={'label': '', 'loc': 'left'},
                labels=["{0}".format(k) for k, v in data.items()],
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
            data = (df__[(df__['Tipo'].isin(["Sucetíveis","Expostos",'Recuperados','Mortos','Inf. Crítico','Inf. Grave','Assintomáticos', 'Inf. Leve'])) & (df__['Tempo (dias)'] == data_aux + 30)][['Tipo','Número de Pessoas']].set_index('Tipo')/N*100)
            df_aux = data.copy(deep = True)
            df_aux['Número de Pessoas'] = df_aux['Número de Pessoas'].apply(lambda x: str(round(x,2)) + '%')
            data = data.to_dict()['Número de Pessoas']
            st.table(pd.DataFrame(df_aux.to_dict()['Número de Pessoas'], index = ['Porcentagem da população']))
            fig = plt.figure(
                FigureClass=Waffle, 
                rows=5,
                columns = 20,
                values=data, 
                colors=("#58c736","#a4de26","#0b2fe3","#f5c842","#fc9403","#f73c02","#07e8f0","#ed07bf"),
                title={'label': '', 'loc': 'left'},
                labels=["{0}".format(k) for k, v in data.items()],
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
            data = (df__[(df__['Tipo'].isin(["Sucetíveis",'Recuperados','Mortos'])) & (df__['Tempo (dias)'] == df__['Tempo (dias)'].max())][['Tipo','Número de Pessoas']].set_index('Tipo')/N*100)
            df_aux = data.copy(deep = True)
            df_aux['Número de Pessoas'] = df_aux['Número de Pessoas'].apply(lambda x: str(round(x,2)) + '%')
            data = data.to_dict()['Número de Pessoas']
            st.table(pd.DataFrame(df_aux.to_dict()['Número de Pessoas'], index = ['Porcentagem da população']))
            fig = plt.figure(
                FigureClass=Waffle, 
                rows=5,
                columns = 20,
                values=data, 
                colors=("#58c736", "#07e8f0", "#c734ae"),
                title={'label': '', 'loc': 'left'},
                labels=["{0}".format(k) for k, v in data.items()],
                legend={'loc': 'lower left', 'bbox_to_anchor': (0, -0.2), 'ncol': 3, 'framealpha': 10},
                icons='user', icon_size=45, 
                icon_legend=True,
                figsize=(15, 6)
            )
            fig.gca().set_facecolor('#EEEEEE')
            fig.set_facecolor('#EEEEEE')
            plt.show()
            st.pyplot()

            st.title("Sobre a simulação:")
            st.write("A simulação e a modelagem foram desenvolvidas aplicando um simulador mais generalizado que desenvolvemos na [Cappra Institute for Data Science](https://www.cappra.institute/) utiliza o modelo epidêmico SEIR desenvolvido pela cientista [Alison Hill](https://alhill.shinyapps.io/COVID19seir/).")
            st.write("Para acessar o simulador original, que permite a variação de diversos parâmetros, [clique aqui](https://desolate-lake-62973.herokuapp.com/). Mais informações sobre o código do simulador e seu desenvolvimento matemático também estão contido no link.")
            st.write("Para mais informações sobre o desenvolvimento da modelagem, parametrização, análise, dados utilizados, dúvidas em geral, etc, entre em contato com caetano@cappra.com")    

if __name__ == "__main__":
    main(IncubPeriod)