import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, PrecisionRecallDisplay
)

# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================
st.set_page_config(
    page_title="ChurnGuard | Retenção & Crescimento",
    page_icon="📉",
    layout="wide",
)

# =========================
# DARK PREMIUM CSS
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg,#070A10 0%,#0B0F17 100%);
    color:white;
}
.block-container { max-width:1200px; }
.card {
    background: rgba(255,255,255,0.05);
    padding:18px;
    border-radius:16px;
    border:1px solid rgba(255,255,255,0.1);
}
.kpi { font-size:1.8rem; font-weight:800; }
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODELO
# =========================
model = joblib.load("modelo_churn.pkl")
model_columns = joblib.load("colunas_modelo.pkl")

# =========================
# LOAD DATASET ORIGINAL
# =========================
df_telco = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df_telco["TotalCharges"] = pd.to_numeric(df_telco["TotalCharges"], errors="coerce")
df_telco.dropna(subset=["TotalCharges"], inplace=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("ChurnGuard")
threshold = st.sidebar.slider("Limiar de decisão", 0.1, 0.9, 0.5, 0.05)

menu = st.sidebar.radio("Navegação", [
    "📊 Visão Executiva",
    "🎯 Priorização Inteligente",
    "🧪 Simulação Individual",
    "💰 Impacto Financeiro",
    "📈 Análise Estratégica"
])

# =========================
# FUNÇÕES
# =========================
def encode_input(df):
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)
    return df_encoded

def predict(df):
    X = encode_input(df)
    return model.predict_proba(X)[:,1]

# =========================
# VISÃO EXECUTIVA
# =========================
if menu == "📊 Visão Executiva":
    st.title("📊 ChurnGuard — Revenue Intelligence")

    col1,col2,col3 = st.columns(3)
    with col1:
        st.markdown('<div class="card"><div class="kpi">Modelo</div>Logistic Regression</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="card"><div class="kpi">{threshold}</div>Limiar atual</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card"><div class="kpi">ROI</div>Simulador ativo</div>', unsafe_allow_html=True)

# =========================
# PRIORIZAÇÃO INTELIGENTE
# =========================
if menu == "🎯 Priorização Inteligente":

    st.header("Ranking de Prioridade")

    df = df_telco.copy()
    y = (df["Churn"]=="Yes").astype(int)
    X = df.drop(columns=["Churn","customerID"])

    probs = predict(X)
    df["Probabilidade"] = probs
    df["Risco"] = np.where(probs>=threshold,"Alto","Baixo")

    df_sorted = df.sort_values("Probabilidade",ascending=False)

    st.dataframe(df_sorted.head(50), use_container_width=True)

# =========================
# SIMULAÇÃO INDIVIDUAL
# =========================
if menu == "🧪 Simulação Individual":

    st.header("Simulação de Cliente")

    gender = st.selectbox("Sexo",["Male","Female"])
    tenure = st.number_input("Tempo como cliente (meses)",0,100,12)
    monthly = st.number_input("Mensalidade",0.0,500.0,70.0)

    input_df = pd.DataFrame([{
        "gender":gender,
        "tenure":tenure,
        "MonthlyCharges":monthly,
        "SeniorCitizen":0,
        "Partner":"No",
        "Dependents":"No",
        "PhoneService":"Yes",
        "MultipleLines":"No",
        "InternetService":"Fiber optic",
        "OnlineSecurity":"No",
        "OnlineBackup":"No",
        "DeviceProtection":"No",
        "TechSupport":"No",
        "StreamingTV":"No",
        "StreamingMovies":"No",
        "Contract":"Month-to-month",
        "PaperlessBilling":"Yes",
        "PaymentMethod":"Electronic check",
        "TotalCharges":monthly*tenure
    }])

    if st.button("Calcular Risco"):
        prob = predict(input_df)[0]
        st.metric("Probabilidade de Churn",f"{prob*100:.2f}%")

# =========================
# IMPACTO FINANCEIRO
# =========================
if menu == "💰 Impacto Financeiro":

    st.header("Simulador de ROI")

    horizonte = st.number_input("Meses",1,24,12)
    sucesso = st.slider("Taxa de sucesso retenção",0.0,1.0,0.2)
    custo = st.number_input("Custo por ação",0.0,100.0,10.0)

    df = df_telco.copy()
    probs = predict(df.drop(columns=["Churn","customerID"]))

    df["Prob"] = probs
    foco = df[df["Prob"]>=threshold]

    receita = foco["MonthlyCharges"].sum()*horizonte*sucesso
    custo_total = len(foco)*custo
    roi = (receita-custo_total)/(custo_total+1e-9)

    st.metric("Receita Salva Estimada",f"R$ {receita:,.0f}")
    st.metric("ROI",f"{roi:.2f}x")

# =========================
# ANÁLISE ESTRATÉGICA
# =========================
if menu == "📈 Análise Estratégica":

    st.header("Métricas do Modelo")

    df = df_telco.copy()
    y = (df["Churn"]=="Yes").astype(int)
    X = df.drop(columns=["Churn","customerID"])

    probs = predict(X)
    y_pred = (probs>=threshold).astype(int)

    precision = precision_score(y,y_pred)
    recall = recall_score(y,y_pred)
    f1 = f1_score(y,y_pred)

    col1,col2,col3 = st.columns(3)
    col1.metric("Precisão",f"{precision:.2f}")
    col2.metric("Recall",f"{recall:.2f}")
    col3.metric("F1 Score",f"{f1:.2f}")

    cm = confusion_matrix(y,y_pred)
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title("Matriz de Confusão")
    st.pyplot(fig)

    st.subheader("Curva Precision-Recall")
    fig2, ax2 = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y,probs,ax=ax2)
    st.pyplot(fig2)
