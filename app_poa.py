import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px

# -- CONFIGURAÇÃO DA PÁGINA ---

st.set_page_config(page_title="Predição Aluguéis POA", layout="wide")

# -- CABEÇALHO ---

st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Sunset_at_Guaiba_Lake.jpg/800px-Sunset_at_Guaiba_Lake.jpg", height=200)
st.title("🏡 Predição de Aluguéis em Porto Alegre/RS")
st.markdown("""
Esta aplicação utiliza **Regressão Linear Múltipla** para estimar o valor de aluguéis na capital gaúcha.
Os dados baseiam-se em estatísticas de mercado de 2024/2025 (FipeZAP e QuintoAndar), simulando a variabilidade de preços entre bairros como Moinhos de Vento e Cidade Baixa.
""")

# -- 1. GERAÇÃO DE DADOS (Simulando Dataset Real de POA) ---

# Baseado na pesquisa: Média m2 ~R$ 35-50, com bairros nobres custando muito mais.

@st.cache_data
def load_data():
np.random.seed(42)
n_samples = 500

# Bairros e seus fatores de valorização (multiplicadores de preço)bairros_data = {    'Restinga': {'fator': 0.6, 'base': 20},    'Sarandi': {'fator': 0.8, 'base': 28},    'Centro Histórico': {'fator': 1.0, 'base': 35},    'Cidade Baixa': {'fator': 1.1, 'base': 40},    'Menino Deus': {'fator': 1.2, 'base': 45},    'Moinhos de Vento': {'fator': 1.8, 'base': 65},    'Bela Vista': {'fator': 1.7, 'base': 60}}
data = []for _ in range(n_samples):    bairro = np.random.choice(list(bairros_data.keys()))    info = bairros_data[bairro]
    # Gerar características    area = int(np.random.normal(70, 25)) # Média 70m2    area = max(20, area) # Mínimo 20m2
    quartos = 1 if area < 45 else (2 if area < 80 else (3 if area < 120 else 4))
    # Ruído aleatório para simular estado de conservação, vagas, etc.    ruido = np.random.normal(0, 300)
    # Fórmula do preço simulada (Regra de negócio oculta que o ML vai descobrir)    # Preço = (Area * Valor_m2_Bairro) + (Quartos * 150) + Ruído    preco_base = (area * info['base']) + (quartos * 150) + ruido    preco_final = max(400, round(preco_base, 2))
    data.append([bairro, area, quartos, preco_final])
df = pd.read_csv(pd.DataFrame(data, columns=['Bairro', 'Area_m2', 'Quartos', 'Preco_Aluguel']).to_csv(index=False)) # Hack para simular CSVdf = pd.DataFrame(data, columns=['Bairro', 'Area_m2', 'Quartos', 'Preco_Aluguel'])return df
df = load_data()

# -- 2. TREINAMENTO DO MODELO ---

# Preparação dos dados (One-Hot Encoding para o Bairro)

df_model = pd.get_dummies(df, columns=['Bairro'], drop_first=False)
X = df_model.drop('Preco_Aluguel', axis=1)
y = df_model['Preco_Aluguel']

model = LinearRegression()
model.fit(X, y)

# -- 3. INTERFACE LATERAL (INPUTS) ---

st.sidebar.header("Parâmetros do Imóvel")
st.sidebar.markdown("Defina as características para simular o preço:")

bairro_input = st.sidebar.selectbox("Bairro", df['Bairro'].unique())
area_input = st.sidebar.slider("Área (m²)", 20, 250, 60)
quartos_input = st.sidebar.slider("Número de Quartos", 1, 5, 2)

# -- 4. PREDIÇÃO ---

# Criar dataframe de entrada com a mesma estrutura do treino

input_data = pd.DataFrame({'Area_m2': [area_input], 'Quartos': [quartos_input], 'Bairro': [bairro_input]})
input_dummies = pd.get_dummies(input_data, columns=['Bairro'])

# Garantir que todas as colunas de bairros existam (mesmo as não selecionadas)

for col in X.columns:
if col not in input_dummies.columns:
input_dummies[col] = 0

# Reordenar colunas para garantir match com o modelo

input_dummies = input_dummies[X.columns]

prediction = model.predict(input_dummies)[0]

# -- 5. EXIBIÇÃO DOS RESULTADOS ---

col1, col2 = st.columns([1, 2])

with col1:
st.markdown("### 💰 Valor Estimado")
st.metric(label="Aluguel Mensal", value=f"R$ {prediction:,.2f}")
st.info(f"""**Análise do Modelo:**O modelo identificou que imóveis no **{bairro_input}** possuem um fator de valorização específico.A cada m² extra, o preço tende a subir significativamente.""")

with col2:
st.markdown("### 📊 Comparativo de Mercado")
# Gráfico de dispersão mostrando onde o imóvel se encaixa
fig = px.scatter(
df,
x='Area_m2',
y='Preco_Aluguel',
color='Bairro',
title="Relação Área x Preço por Bairro (Dados Históricos)",
labels={'Area_m2': 'Área (m²)', 'Preco_Aluguel': 'Aluguel (R$)'},
opacity=0.6
)

# Adicionar o ponto predito
fig.add_scatter(
    x=[area_input],
    y=[prediction],
    mode='markers',
    marker=dict(size=15, color='red', symbol='x'),
    name='Sua Simulação'
)

st.plotly_chart(fig, use_container_width=True)

# -- 6. VISUALIZAÇÃO DE DADOS ---

st.divider()
st.subheader("Amostra dos Dados Utilizados (Porto Alegre)")
st.dataframe(df.sample(5))

st.markdown("---")
st.caption("Desenvolvido com Streamlit por Luciano Martins Fagundes • Modelo: Regressão Linear Múltipla • Dados Simulados baseados no FipeZAP 2024/2025")
