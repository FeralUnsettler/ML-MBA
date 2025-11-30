import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px

# -------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# -------------------------------------------

st.set_page_config(page_title="Predição Aluguéis POA", layout="wide")

# Imagem substituída pela opção 3 (Unsplash – sempre funciona)
st.image(
    "https://images.unsplash.com/photo-1501612780327-45045538702b",
    use_column_width=True
)

st.title("🏡 Predição de Aluguéis em Porto Alegre/RS")
st.markdown("""
Esta aplicação utiliza **Regressão Linear Múltipla** para estimar o valor de aluguéis na capital gaúcha.
Os dados baseiam-se em estatísticas de mercado de 2024/2025 (FipeZAP e QuintoAndar), simulando a variabilidade
de preços entre bairros como Moinhos de Vento e Cidade Baixa.
""")

# -------------------------------------------
# 1. GERAÇÃO DO DATASET SIMULADO
# -------------------------------------------

@st.cache_data
def load_data():
    np.random.seed(42)
    n_samples = 500

    bairros_data = {
        'Restinga': {'fator': 0.6, 'base': 20},
        'Sarandi': {'fator': 0.8, 'base': 28},
        'Centro Histórico': {'fator': 1.0, 'base': 35},
        'Cidade Baixa': {'fator': 1.1, 'base': 40},
        'Menino Deus': {'fator': 1.2, 'base': 45},
        'Moinhos de Vento': {'fator': 1.8, 'base': 65},
        'Bela Vista': {'fator': 1.7, 'base': 60},
    }

    data = []
    for _ in range(n_samples):
        bairro = np.random.choice(list(bairros_data.keys()))
        info = bairros_data[bairro]

        # Área
        area = int(np.random.normal(70, 25))
        area = max(20, area)

        # Quartos
        if area < 45:
            quartos = 1
        elif area < 80:
            quartos = 2
        elif area < 120:
            quartos = 3
        else:
            quartos = 4

        # Ruído
        ruido = np.random.normal(0, 300)

        # Preço
        preco_base = (area * info['base']) + (quartos * 150) + ruido
        preco_final = max(400, round(preco_base, 2))

        data.append([bairro, area, quartos, preco_final])

    return pd.DataFrame(data, columns=['Bairro', 'Area_m2', 'Quartos', 'Preco_Aluguel'])


df = load_data()

# -------------------------------------------
# 2. TREINAMENTO DO MODELO
# -------------------------------------------

df_model = pd.get_dummies(df, columns=['Bairro'], drop_first=False)
X = df_model.drop('Preco_Aluguel', axis=1)
y = df_model['Preco_Aluguel']

model = LinearRegression()
model.fit(X, y)

# -------------------------------------------
# 3. BARRA LATERAL
# -------------------------------------------

st.sidebar.header("Parâmetros do Imóvel")

bairro_input = st.sidebar.selectbox("Bairro", df['Bairro'].unique())
area_input = st.sidebar.slider("Área (m²)", 20, 250, 60)
quartos_input = st.sidebar.slider("Número de Quartos", 1, 5, 2)

# -------------------------------------------
# 4. DADO DE ENTRADA
# -------------------------------------------

input_data = pd.DataFrame({
    'Area_m2': [area_input],
    'Quartos': [quartos_input],
    'Bairro': [bairro_input]
})

input_dummies = pd.get_dummies(input_data, columns=['Bairro'])

# Garantir colunas idênticas
for col in X.columns:
    if col not in input_dummies.columns:
        input_dummies[col] = 0

input_dummies = input_dummies[X.columns]

prediction = model.predict(input_dummies)[0]

# -------------------------------------------
# 5. RESULTADOS
# -------------------------------------------

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 💰 Valor Estimado")
    st.metric("Aluguel Mensal", f"R$ {prediction:,.2f}")

    st.info(f"""
**Análise do Modelo:**  
Imóveis no **{bairro_input}** possuem um padrão próprio de valorização.
A cada m² adicional, o preço tende a aumentar de forma proporcional ao valor médio do bairro.
""")

with col2:
    st.markdown("### 📊 Comparativo de Mercado")

    fig = px.scatter(
        df,
        x="Area_m2",
        y="Preco_Aluguel",
        color="Bairro",
        opacity=0.6,
        title="Relação Área x Preço por Bairro (Dados Históricos)"
    )

    fig.add_scatter(
        x=[area_input],
        y=[prediction],
        mode='markers',
        marker=dict(size=15, color='red', symbol='x'),
        name="Sua Simulação"
    )

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------
# 6. VISUALIZAÇÃO
# -------------------------------------------

st.divider()
st.subheader("Amostra dos Dados Utilizados (Porto Alegre)")
st.dataframe(df.sample(5))

st.markdown("---")
st.caption(
    "Desenvolvido com Streamlit por Luciano Martins Fagundes • "
    "Modelo: Regressão Linear Múltipla • Dados Simulados FipeZAP 2024/2025"
)
