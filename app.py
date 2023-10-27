# bibliotecas
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
import datetime

# configurações de nome do arquivo exportado
ano = datetime.date.today().year
mes = datetime.date.today().month                
nome = f'{ano}_{mes}_Streamlit_Prophet.csv'

# Função para aplicar o modelo
def modelo(dados):
    m = Prophet()
    
    m.fit(dados)
    
    future = m.make_future_dataframe(periods=12, freq='M')
    
    forecast = m.predict(future)
    forecast = forecast[['ds', 'yhat', 'yhat_upper','yhat_lower']]
    forecast['yhat'] = forecast.yhat.astype(int)
    forecast['yhat_upper'] = forecast.yhat_upper.astype(int)
    forecast['yhat_lower'] = forecast.yhat_lower.astype(int)
    forecast = pd.merge(
        left=forecast, 
        right=dados, 
        on=['ds'],
        how='left'
    )
    forecast.columns = ['Data','Previsão','Upper','Lower','Real']
    forecast['MAPE'] = 1-np.abs((forecast.Real-forecast.Previsão)/forecast.Real)
    return forecast

# configurações do front-end do app
st.set_page_config(layout="wide")

st.title('Forecast com Prophet')        
st.sidebar.title('Informações')    

st.sidebar.write(
    '''
    Para usar o modelo, tenha uma base de dados
    com uma coluna 'ds' contendo datas no padrão 'dd/mm/aaaa' 
    e uma coluna 'y' contendo valores sem separador de milhar.\n
    Exemplo:
    '''
    )
st.sidebar.write(
    pd.DataFrame(
        {
            'ds':['31/01/2023','28/02/2023'],
            'y':['12345','56789']
        }
    )
)
st.sidebar.markdown("---")

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
        
        # Gráfico com resultado
        st.subheader('Visualização')
        fig = px.line(
            resultado, 
            x='Data', 
            y=['Real', 'Previsão'],
            width=1000,
            labels={'value':'Valores'}
        )

        # Adicionar as colunas 'Upper' e 'Lower' com cores diferentes
        fig.add_scatter(
            x=resultado['Data'], 
            y=resultado['Upper'], 
            mode='lines', 
            name='Upper', 
            line=dict(color='#ebeced')
        )
        fig.add_scatter(
            x=resultado['Data'], 
            y=resultado['Lower'], 
            mode='lines', 
            name='Lower', 
            line=dict(color='#ebeced')
        )
        st.plotly_chart(fig)
        
        # Tabela com resultado
        st.subheader('Resultado')
        st.write(resultado)
        
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