# Importe outras bibliotecas necessárias
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import warnings
from statsmodels.tsa.arima.model import ARIMA
from datetime import date, timedelta
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

st.title('Tech Chellenger 4 - Grupo 02')
st.write("***Integrantes:***")
st.write("*Bárbara Pereira Godoi*")
st.write("*Fernando correia do Nascimento*")
st.write("*Guilherme Gentil Da Silva*")
st.write("*João Vitor Lopes Arruda*")
st.write("*William Fernandes Bento*")

df = pd.read_csv('BasePrecoPetroleo.csv')

font_grafico = {'family':'serif','color':'darkred','size':20}

grafico = plt.figure(figsize=(10, 6))
plt.plot(df)
plt.title("Evolução histórica do preço do petróleo", fontdict=font_grafico, weight='bold', style='italic')
plt.legend(bbox_to_anchor = (1.3, 1.3), ncol = 8)
plt.xlabel("Periodo")
plt.ylabel("Valor Total")
plt.xticks(rotation=35)
st.pyplot(grafico) 

#Trabalhando com dados dos últimos 365 dias
dt_hoje = date.today()
dt_inicio = pd.to_datetime((dt_hoje + timedelta(-365)))
df_b = df[(df.index >= dt_inicio)]

grafico = plt.figure(figsize=(10, 6))
plt.plot(df_b)
plt.title("Evolução do preço do petróleo nos últimos 365 dias", fontdict=font_grafico, weight='bold', style='italic')
plt.legend(bbox_to_anchor = (1.3, 1.3), ncol = 8)
plt.xlabel("Periodo")
plt.ylabel("Valor Total")
plt.xticks(rotation=35)
st.pyplot(grafico) 

# Dividir dados em treinamento e teste
train_size = int(len(df_b) * 0.7)
df_train, df_test = df_b[:train_size], df_b[train_size:]

st.write("---")
st.write("**Dividindo a base em Treino e Teste...**")
st.write("*df_train.shape:* ",df_train.shape," *| df_test.shape:* ",df_test.shape)
st.write("---")

def ehEstacionaria(timeseries):
  dftest = adfuller(timeseries, autolag='AIC')
  dfoutput = pd.Series(dftest[0:4], index=['Estatistica do teste','p-value','O criterio de informação maximizado','Numero de observações usadas'])
  for key,value in dftest[4].items():
    dfoutput['Valor crítico (%s)'%key] = value
  
  if(dfoutput['Estatistica do teste'] < dfoutput['Valor crítico (5%)'] and dfoutput['p-value'] < 0.05):
    st.write(f':black[**É estacionária.**]')
  else:
    st.write(f':red[**Não é estacionária.**]')
    
def test_stationary(timeseries):

  medmov = timeseries.rolling(12).mean() #media movel
  despad = timeseries.rolling(12).std() #desvio movel

  grafico = plt.figure(figsize=(10, 6))
  plt.plot(timeseries, color='blue', label='Valor original')
  plt.plot(medmov, color='red', label='Valor média móvel')
  plt.plot(despad, color='black', label='Desvio móvel')
  plt.legend(loc='best')
  plt.title('Média móvel e desvio padrão', fontdict=font_grafico, weight='bold', style='italic')
  plt.xticks(rotation=35)
  st.pyplot(grafico) 

  st.write("")
  st.write('**Resultado do teste de Dickey Fuller**')
  dftest = adfuller(timeseries, autolag='AIC')
  dfoutput = pd.Series(dftest[0:4], index=['Estatística do teste','p-value','O critério de informação maximizado','Numero de observações usadas'])
  for key,value in dftest[4].items():
    dfoutput['Valor crítico (%s)'%key] = value
  st.write(dfoutput)

  ehEstacionaria(timeseries)
  
st.write("**Verificando se os dados estão em forma estacionária**")
test_stationary(df_train['VALOR'])
st.write("---")

st.write("**Tornando a série estacionária**")
dfdiff = df_train.VALOR.diff()
dfdiff = dfdiff.dropna()
test_stationary(dfdiff)
st.write("---")

st.write("***Determinando os parâmetros ARIMA***")
p = d = q = range(0, 9)
pdq_combinations = list(itertools.product(p, d, q))
menor_aic = float("inf")
melhor_pdq = None
warnings.filterwarnings("ignore")
st.write("Combinações ARIMA(p d q):")

#for pdq in pdq_combinations:
#    try:
#        model = ARIMA(df_train['VALOR'].dropna(), order=pdq)
#        results = model.fit()
#        if results.aic < menor_aic:
#            menor_aic = results.aic
#            melhor_pdq = pdq            
#    except:
#        continue
#    st.write('*Identificado a melhor combinação ARIMA(p d q):* ',melhor_pdq)
    
melhor_pdq = (5, 1, 3)
st.write("**-> Melhor combinação ARIMA(p d q):**", melhor_pdq)
st.write("")
st.write("Ajustando o modelo Arima com o melhor parametro pdq")
modelo_arima_otimizado = ARIMA(df_b['VALOR'], order=(melhor_pdq))
modelo_arima_otimizado_fit = modelo_arima_otimizado.fit()
st.write(modelo_arima_otimizado_fit.summary())
st.write("---")
st.write("")

df_b['DATA'] = df_b.index
data_final = df_b['DATA'].iloc[-1]
datas_futuras = pd.date_range(start=data_final, periods=31, inclusive='right')  # 30 dias após a última data

previsoes_futuras = modelo_arima_otimizado_fit.forecast(steps=len(datas_futuras))

dt_hoje = date.today()
dt_inicio = pd.to_datetime((dt_hoje + timedelta(-30)))

df = df_b[(df_b['DATA'] >= dt_inicio)]
grafico = plt.figure(figsize=(10, 6))
plt.plot(df['DATA'], df['VALOR'], label='Dados Históricos')
plt.plot(datas_futuras, previsoes_futuras, color='red', label='Previsões Futuras')
plt.title('Previsões Futuras de Preço do Petróleo', fontdict=font_grafico, weight='bold', style='italic')
plt.xlabel('Data')
plt.ylabel('Preço')
plt.legend()
plt.xticks(rotation=35)
st.pyplot(grafico) 


previsoes_df = pd.DataFrame({
    'DATA': datas_futuras,
    'VALOR': previsoes_futuras
})

df_petro = df[['DATA','VALOR']]
df_petro['DATA'] = np.datetime_as_string(df_petro['DATA'], unit='D')
df_petro.to_csv('BasePrecoPetroleo.csv', index=False)
previsoes_df.to_csv('previsoes_petroleo.csv', index=False)
