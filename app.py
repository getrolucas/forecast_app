# bibliotecas
import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import datetime

# configurações de nome do arquivo exportado
ano = datetime.date.today().year
mes = datetime.date.today().month                
nome = f'{ano}_{mes}_Streamlit_Prophet.csv'

# configurações do front-end do app
st.set_page_config(layout="wide")

st.title('Forecast com Prophet')        
st.sidebar.title('Informações')    

st.sidebar.write(
    '''
    Para usar o modelo, tenha uma base de dados
    com uma coluna de datas nomeada 'ds' e padrão 'dd/mm/aaaa' e outra 
    com valores nomeada 'y' sem separador de milhar.
    '''
    )

st.sidebar.markdown("---")

col1, col2 = st.columns(2)

# Função para aplicar o modelo
def modelo(dados):
    m = Prophet()
    
    m.fit(dados)
    
    future = m.make_future_dataframe(periods=12)
    
    forecast = m.predict(future)
    forecast = forecast[['ds', 'yhat']]
    forecast['yhat'] = forecast.yhat.astype(int)
    forecast = pd.merge(
        left=forecast, 
        right=dados, 
        on=['ds'],
        how='left'
    )
    return forecast

# Upload de arquivo
st.sidebar.subheader('Faça upload do arquivo .csv')
uploaded_file = st.sidebar.file_uploader("", type=["csv"])

if uploaded_file is not None:
    
    df = pd.read_csv(
        uploaded_file,
        parse_dates=['ds'],
        dayfirst=True,
        on_bad_lines='skip',
        sep=';'
    )

    # Botão rodar o modelp
    if st.sidebar.button('Aplicar Modelo'):
        
        resultado = modelo(df)
        
        # Tabela com resultado
        col1.subheader('Resultado')
        col1.write(resultado)
        
        col2.subheader('Visualização')
        col2.line_chart(
            resultado, 
            x="ds", 
            y=["y", "yhat"]
        )

        # Baixar resultados
        st.download_button(
            label="Baixar Resultados em .csv",
            data=resultado.to_csv(
                index=False, 
                decimal=',', 
                sep=';'
            ).encode(),
            file_name=nome,
            key="download-results"
        )