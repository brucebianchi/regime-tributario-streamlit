import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Geração de dados fictícios
np.random.seed(42)
n = 200
faturamento = np.random.randint(100, 5000, n)  # em mil R$
cnae = np.random.randint(1, 6, n)              # CNAE de 1 a 5
funcionarios = np.random.randint(1, 50, n)
folha_percentual = np.random.randint(5, 60, n)

# Regra para determinar regime tributário
regimes = []
for f, c, func, folha in zip(faturamento, cnae, funcionarios, folha_percentual):
    if f <= 4800 and c in [1, 2] and folha < 30:
        regimes.append("Simples Nacional")
    elif f <= 7800 and folha < 40:
        regimes.append("Lucro Presumido")
    else:
        regimes.append("Lucro Real")

# Dataset
df = pd.DataFrame({
    "Faturamento": faturamento,
    "CNAE": cnae,
    "Funcionários": funcionarios,
    "Folha (%)": folha_percentual,
    "Regime": regimes
})

# Modelo
X = df[["Faturamento", "CNAE", "Funcionários", "Folha (%)"]]
y = df["Regime"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

modelo = DecisionTreeClassifier(max_depth=4)
modelo.fit(X_train, y_train)

# Interface Streamlit
st.title("Calculadora de Regime Tributário 📊")

st.sidebar.header("Preencha os dados da empresa:")
fat = st.sidebar.number_input("Faturamento (mil R$)", min_value=0, max_value=10000, value=1000)
cnae = st.sidebar.selectbox("CNAE (categoria simplificada)", options=[1, 2, 3, 4, 5])
func = st.sidebar.slider("Número de Funcionários", 0, 100, 5)
folha = st.sidebar.slider("Percentual da Folha de Pagamento (%)", 0, 100, 20)

if st.sidebar.button("Calcular Regime"):
    entrada = np.array([[fat, cnae, func, folha]])
    previsao = modelo.predict(entrada)
    st.success(f"📌 Regime tributário sugerido: **{previsao[0]}**")

# Acurácia do modelo
st.markdown("### Acurácia do modelo")
acuracia = accuracy_score(y_test, modelo.predict(X_test))
st.write(f"Acurácia: **{acuracia:.2f}**")

# Exibir árvore de decisão
st.markdown("### Árvore de Decisão")
fig, ax = plt.subplots(figsize=(16, 8))
plot_tree(modelo, feature_names=X.columns, class_names=modelo.classes_, filled=True, ax=ax)
st.pyplot(fig)
