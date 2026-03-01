import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# =========================================================
# Configurações gerais
# =========================================================
st.set_page_config(
    page_title="ChurnGuard • Retenção & Prioridades",
    page_icon="📉",
    layout="wide",
)

st.markdown(
    """
    <style>
      .small-muted { color: #6b7280; font-size: 0.92rem; }
      .hr { border-top: 1px solid #e5e7eb; margin: 1rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Carregar artefatos do modelo
# =========================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("modelo_churn.pkl")
    cols = joblib.load("colunas_modelo.pkl")
    return model, cols

model, model_columns = load_artifacts()

# =========================================================
# Dataset para dashboard (opcional)
# =========================================================
@st.cache_data
def load_dataset(path="WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).copy()
    df["Churn_num"] = df["Churn"].map({"No": 0, "Yes": 1})
    return df

try:
    df_data = load_dataset()
    dataset_loaded = True
except Exception:
    df_data = None
    dataset_loaded = False

# =========================================================
# Estado do app
# =========================================================
if "history" not in st.session_state:
    st.session_state.history = []

if "last_batch" not in st.session_state:
    st.session_state.last_batch = None

# =========================================================
# Helpers
# =========================================================
FEATURES_RAW = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges"
]

def safe_to_numeric(series, default=np.nan):
    out = pd.to_numeric(series, errors="coerce")
    return out.fillna(default)

def build_encoded(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_encoded = pd.get_dummies(df_raw, drop_first=True)
    df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)
    return df_encoded

def risk_bucket(prob: float) -> str:
    if prob < 0.30:
        return "Baixo"
    if prob < 0.60:
        return "Médio"
    return "Alto"

def risk_emoji(bucket: str) -> str:
    return {"Baixo": "🟢", "Médio": "🟡", "Alto": "🔴"}.get(bucket, "🟡")

def ensure_required_columns(df: pd.DataFrame):
    missing = [c for c in FEATURES_RAW if c not in df.columns]
    return (len(missing) == 0, missing)

def priority_score(prob: float, tenure: float, contract: str, monthly: float) -> float:
    score = prob * 100
    if tenure <= 6: score += 10
    elif tenure <= 12: score += 6
    elif tenure <= 24: score += 3

    if contract == "Month-to-month": score += 8
    elif contract == "One year": score += 2

    if monthly >= 90: score += 6
    elif monthly >= 70: score += 3

    return float(np.clip(score, 0, 100))

def playbook_row(prob: float, contract: str, tenure: int, monthly: float, payment: str) -> dict:
    bucket = risk_bucket(prob)

    action = "Acompanhar"
    offer = "Comunicação padrão"
    channel = "E-mail / In-app"
    reason = "Risco baixo"

    if bucket == "Médio":
        action = "Contato proativo"
        channel = "WhatsApp / Telefone"
        offer = "Benefício leve (upgrade/bonus)"
        reason = "Risco médio — agir antes do cancelamento"
    elif bucket == "Alto":
        action = "Retenção imediata"
        channel = "Telefone (prioridade) + WhatsApp"
        offer = "Oferta direcionada (desconto/upgrade/contrato)"
        reason = "Risco alto — evitar churn nas próximas semanas"

    if contract == "Month-to-month" and bucket in ["Médio", "Alto"]:
        offer = "Incentivo para migrar para contrato anual (reduz churn)"
        reason += " • contrato mês a mês"
    if tenure <= 6 and bucket in ["Médio", "Alto"]:
        reason += " • cliente novo (tenure baixo)"
    if monthly >= 80 and bucket in ["Médio", "Alto"]:
        reason += " • cobrança alta"
    if payment == "Electronic check" and bucket in ["Médio", "Alto"]:
        reason += " • método com churn alto (Electronic check)"

    return {
        "acao": action,
        "canal": channel,
        "oferta_sugerida": offer,
        "racional": reason
    }

def explain_logreg_top_factors(X_row: pd.DataFrame, top_n: int = 6):
    if not hasattr(model, "coef_"):
        return pd.DataFrame(columns=["Fator", "Impacto (aprox.)"])
    coefs = pd.Series(model.coef_[0], index=model_columns)
    row = X_row.iloc[0]
    contrib = (row * coefs).sort_values(ascending=False)
    contrib = contrib[contrib != 0].head(top_n)
    if contrib.empty:
        return pd.DataFrame(columns=["Fator", "Impacto (aprox.)"])
    return pd.DataFrame({
        "Fator": contrib.index.str.replace("_", " ").str.replace("-", " "),
        "Impacto (aprox.)": contrib.values.round(4)
    })

def template_csv_bytes():
    df_tpl = pd.DataFrame(columns=["customerID"] + FEATURES_RAW)
    df_tpl.loc[0] = [
        "CUST-0001",
        "Female", 0, "Yes", "No", 12,
        "Yes", "No", "DSL",
        "Yes", "Yes", "Yes", "Yes",
        "No", "No", "Month-to-month", "Yes",
        "Electronic check", 75.0, 1200.0
    ]
    return df_tpl.to_csv(index=False).encode("utf-8")

# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    try:
        st.image("logo.png", use_container_width=True)  # imagem ainda aceita use_container_width
    except Exception:
        pass

    st.markdown("## 📉 ChurnGuard")
    st.caption("Retenção • Priorização • Batch scoring")

    threshold = st.slider(
        "Limiar (ponto de corte)",
        min_value=0.10, max_value=0.90, value=0.50, step=0.05,
        help="Abaixo do limiar = não churn. Acima = churn."
    )

    st.markdown("---")
    menu = st.radio(
        "Navegação",
        ["🏠 Visão geral", "📥 Batch scoring (CSV)", "📌 Plano de Ação", "🧪 Simulador (1 cliente)", "📊 Dashboard", "🗂 Histórico"],
    )

    st.markdown("---")
    st.download_button(
        "⬇️ Baixar CSV modelo (template)",
        data=template_csv_bytes(),
        file_name="template_clientes_churn.csv",
        mime="text/csv",
        width="stretch"
    )

# =========================================================
# Header
# =========================================================
st.markdown("# ChurnGuard — Previsão de Cancelamento")
st.markdown('<div class="small-muted">Aplicação operacional: upload de base → churn → ranking → playbook → export para retenção.</div>', unsafe_allow_html=True)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# =========================================================
# 1) Visão geral
# =========================================================
if menu == "🏠 Visão geral":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Threshold", f"{threshold:.2f}")
    c2.metric("Módulo", "Batch")
    c3.metric("Saída", "CSV pronto")
    c4.metric("Playbook", "Ativo")

    st.subheader("O que este produto entrega")
    st.write(
        "✅ Modelo treinado + app interativo\n"
        "✅ Upload de CSV de clientes e scoring em lote\n"
        "✅ Ranking Top 50 por prioridade de ação\n"
        "✅ Filtros operacionais para o time de retenção\n"
        "✅ Playbook de retenção (ação, canal, oferta)\n"
        "✅ Export do resultado (top50 e completo)\n"
        "✅ Plano de Ação (contatos do dia + copiar lista + impacto estimado + export alto risco)"
    )

# =========================================================
# 2) Batch scoring
# =========================================================
elif menu == "📥 Batch scoring (CSV)":
    st.subheader("📥 Batch scoring — upload de base e priorização automática")

    with st.expander("✅ Formato esperado do CSV (colunas obrigatórias)"):
        st.code(", ".join(FEATURES_RAW), language="text")
        st.caption("Você pode incluir `customerID` (opcional). Colunas extras ok.")

    uploaded = st.file_uploader("Upload do CSV de clientes", type=["csv"])

    if uploaded is None:
        st.info("Envie um CSV para calcular churn e gerar ranking.")
        st.stop()

    df_in = pd.read_csv(uploaded).copy()

    has_id = "customerID" in df_in.columns
    if has_id:
        df_ids = df_in[["customerID"]].copy()
    else:
        df_ids = pd.DataFrame({"customerID": [f"CLIENTE-{i+1:05d}" for i in range(len(df_in))]})

    if "Churn" in df_in.columns:
        df_in = df_in.drop(columns=["Churn"])

    ok, missing = ensure_required_columns(df_in)
    if not ok:
        st.error("Seu CSV está faltando colunas obrigatórias:")
        st.write(missing)
        st.stop()

    df_in["TotalCharges"] = safe_to_numeric(df_in["TotalCharges"], default=np.nan)
    before = len(df_in)
    df_scoring = df_in.dropna(subset=["TotalCharges"]).copy()
    removed = before - len(df_scoring)
    if removed > 0:
        st.warning(f"Removi {removed} linhas com TotalCharges inválido/vazio (igual no treino).")

    df_scoring["SeniorCitizen"] = safe_to_numeric(df_scoring["SeniorCitizen"], default=0).astype(int)
    df_scoring["tenure"] = safe_to_numeric(df_scoring["tenure"], default=0).astype(int)
    df_scoring["MonthlyCharges"] = safe_to_numeric(df_scoring["MonthlyCharges"], default=0.0).astype(float)
    df_scoring["TotalCharges"] = safe_to_numeric(df_scoring["TotalCharges"], default=np.nan).astype(float)

    X = build_encoded(df_scoring[FEATURES_RAW])
    probs = model.predict_proba(X)[:, 1]

    df_out = df_scoring.copy()
    df_out.insert(0, "customerID", df_ids.loc[df_scoring.index, "customerID"].values)

    df_out["prob_churn"] = probs
    df_out["classe_predita"] = (df_out["prob_churn"] >= threshold).astype(int)
    df_out["risco"] = df_out["prob_churn"].apply(risk_bucket)

    df_out["prioridade_score"] = df_out.apply(
        lambda r: priority_score(r["prob_churn"], r["tenure"], r["Contract"], r["MonthlyCharges"]),
        axis=1
    )

    pb = df_out.apply(
        lambda r: playbook_row(r["prob_churn"], r["Contract"], int(r["tenure"]), float(r["MonthlyCharges"]), r["PaymentMethod"]),
        axis=1
    )
    df_out["acao"] = pb.apply(lambda d: d["acao"])
    df_out["canal"] = pb.apply(lambda d: d["canal"])
    df_out["oferta_sugerida"] = pb.apply(lambda d: d["oferta_sugerida"])
    df_out["racional"] = pb.apply(lambda d: d["racional"])

    ordem = {"Alto": 0, "Médio": 1, "Baixo": 2}
    df_out["_ordem"] = df_out["risco"].map(ordem)
    df_out = df_out.sort_values(by=["_ordem", "prioridade_score", "prob_churn"], ascending=[True, False, False]).drop(columns=["_ordem"])

    st.session_state.last_batch = df_out.copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clientes avaliados", f"{len(df_out):,}".replace(",", "."))
    c2.metric("Alto risco", int((df_out["risco"] == "Alto").sum()))
    c3.metric("Médio risco", int((df_out["risco"] == "Médio").sum()))
    c4.metric("Baixo risco", int((df_out["risco"] == "Baixo").sum()))

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.markdown("### 🏆 Ranking de Prioridade — Top 50")
    df_top50 = df_out.head(50).copy()
    df_top50.insert(1, "badge_risco", df_top50["risco"].apply(lambda r: f"{risk_emoji(r)} {r}"))

    cols_show = [
        "customerID", "badge_risco", "prioridade_score", "prob_churn",
        "Contract", "tenure", "MonthlyCharges", "PaymentMethod",
        "acao", "canal", "oferta_sugerida"
    ]
    st.dataframe(df_top50[cols_show], width="stretch", hide_index=True)

    st.markdown("### ⬇️ Exportações")
    cexp1, cexp2 = st.columns(2)
    with cexp1:
        st.download_button(
            "⬇️ Baixar TOP 50 (CSV)",
            data=df_top50.to_csv(index=False).encode("utf-8"),
            file_name=f"top50_prioridade_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            width="stretch"
        )
    with cexp2:
        st.download_button(
            "⬇️ Baixar BASE completa com risco (CSV)",
            data=df_out.to_csv(index=False).encode("utf-8"),
            file_name=f"clientes_risco_completo_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            width="stretch"
        )

    st.success("Batch scoring concluído ✅ Agora vá em **Plano de Ação** para gerar a lista do dia.")

# =========================================================
# 3) Plano de Ação
# =========================================================
elif menu == "📌 Plano de Ação":
    st.subheader("📌 Plano de Ação — Lista do dia para o time de retenção")
    st.caption("Baseado no último CSV processado no Batch Scoring.")

    if st.session_state.last_batch is None:
        st.warning("Ainda não há batch carregado. Vá em Batch Scoring e faça upload de uma base.")
        st.stop()

    df = st.session_state.last_batch.copy()

    left, right = st.columns([2, 1])
    with right:
        capacidade = st.number_input("Capacidade diária (contatos)", min_value=5, max_value=500, value=50, step=5)
        foco = st.multiselect("Foco de risco", ["Alto", "Médio", "Baixo"], default=["Alto"])
        taxa_conversao = st.slider("Taxa estimada de retenção (%)", 5, 80, 30)
        st.info("Sugestão: começar por **Alto** e depois preencher com **Médio** conforme capacidade.")

    df_f = df[df["risco"].isin(foco)].copy()
    df_f = df_f.sort_values(by=["prioridade_score", "prob_churn"], ascending=[False, False])

    total_alto = int((df["risco"] == "Alto").sum())
    total_medio = int((df["risco"] == "Médio").sum())
    total_baixo = int((df["risco"] == "Baixo").sum())

    df_today = df_f.head(int(capacidade)).copy()

    receita_media = float(df_today["MonthlyCharges"].mean()) if len(df_today) > 0 else 0.0
    churns_previstos = float(df_today["prob_churn"].sum()) if len(df_today) > 0 else 0.0
    churns_evitar = churns_previstos * (taxa_conversao / 100)
    receita_preservada = churns_evitar * receita_media

    with left:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Alto risco total", total_alto)
        k2.metric("Contatos hoje", len(df_today))
        k3.metric("Churn estimado no grupo", f"{churns_previstos:.1f}")
        k4.metric("Receita potencial preservada", f"R$ {receita_preservada:,.0f}".replace(",", "."))

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.markdown("### 🏆 Lista do dia (prioridade máxima)")
    if len(df_today) == 0:
        st.warning("Não há clientes no filtro atual. Ajuste o foco de risco ou aumente a base.")
    else:
        df_today.insert(1, "badge_risco", df_today["risco"].apply(lambda r: f"{risk_emoji(r)} {r}"))
        cols_show = [
            "customerID", "badge_risco", "prioridade_score", "prob_churn",
            "Contract", "tenure", "MonthlyCharges", "PaymentMethod",
            "acao", "canal", "oferta_sugerida", "racional"
        ]
        st.dataframe(df_today[cols_show], width="stretch", hide_index=True)

        st.markdown("### 📞 Lista resumida para CRM / WhatsApp")
        texto_lista = ""
        for _, row in df_today.iterrows():
            texto_lista += (
                f"{row['customerID']} | Risco: {row['risco']} | "
                f"Ação: {row['acao']} | Oferta: {row['oferta_sugerida']}\n"
            )

        st.text_area("Copiar e colar no CRM:", value=texto_lista, height=220)

        st.markdown("### ⬇️ Exportações")
        cexp1, cexp2 = st.columns(2)

        with cexp1:
            st.download_button(
                "⬇️ Baixar LISTA DO DIA (CSV)",
                data=df_today.to_csv(index=False).encode("utf-8"),
                file_name=f"lista_do_dia_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                width="stretch"
            )

        with cexp2:
            df_high = df[df["risco"] == "Alto"].copy()
            st.download_button(
                "⬇️ Baixar SOMENTE ALTO RISCO",
                data=df_high.to_csv(index=False).encode("utf-8"),
                file_name=f"alto_risco_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                width="stretch"
            )

        st.success("Plano de ação pronto ✅ Lista gerada com impacto estimado.")

# =========================================================
# 4) Simulador (1 cliente)
# =========================================================
elif menu == "🧪 Simulador (1 cliente)":
    st.subheader("🧪 Simulador — cliente individual (demo)")

    gender = st.selectbox("gender", ["Female", "Male"])
    senior = st.selectbox("SeniorCitizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("tenure", min_value=0, max_value=120, value=12)
    phone = st.selectbox("PhoneService", ["Yes", "No"])
    multilines = st.selectbox("MultipleLines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("OnlineSecurity", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("OnlineBackup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("DeviceProtection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("TechSupport", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("StreamingTV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("StreamingMovies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("PaperlessBilling", ["Yes", "No"])
    payment = st.selectbox(
        "PaymentMethod",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    monthly = st.number_input("MonthlyCharges", min_value=0.0, max_value=200.0, value=75.0, step=0.5)
    total = st.number_input("TotalCharges", min_value=0.0, max_value=100000.0, value=1200.0, step=10.0)

    if st.button("✅ Calcular", width="stretch"):
        raw = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": int(senior),
            "Partner": partner,
            "Dependents": dependents,
            "tenure": int(tenure),
            "PhoneService": phone,
            "MultipleLines": multilines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": float(monthly),
            "TotalCharges": float(total),
        }])

        X = build_encoded(raw)
        prob = float(model.predict_proba(X)[0][1])
        bucket = risk_bucket(prob)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Prob. churn", f"{prob*100:.1f}%")
        c2.metric("Risco", f"{risk_emoji(bucket)} {bucket}")
        c3.metric("Classe (threshold)", "Churn" if prob >= threshold else "Não churn")
        c4.metric("Prioridade score", f"{priority_score(prob, tenure, contract, monthly):.1f}")

        pb = playbook_row(prob, contract, int(tenure), float(monthly), payment)
        st.info(f"**Playbook:** {pb['acao']} • {pb['canal']} • {pb['oferta_sugerida']} \n\n**Racional:** {pb['racional']}")

        with st.expander("🧠 Top fatores (se for Regressão Logística)"):
            df_factors = explain_logreg_top_factors(X, top_n=8)
            if df_factors.empty:
                st.write("Este modelo não expõe coeficientes para explicação (ou não é Regressão Logística).")
            else:
                st.dataframe(df_factors, width="stretch", hide_index=True)

# =========================================================
# 5) Dashboard
# =========================================================
elif menu == "📊 Dashboard":
    st.subheader("📊 Dashboard — drivers do churn (EDA)")

    if not dataset_loaded:
        st.warning("Não consegui carregar o dataset. Coloque o CSV na pasta do app.")
    else:
        churn_rate = df_data["Churn_num"].mean()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Clientes", f"{len(df_data):,}".replace(",", "."))
        k2.metric("Taxa de churn", f"{churn_rate*100:.1f}%")
        k3.metric("Tenure médio", f"{df_data['tenure'].mean():.1f} meses")
        k4.metric("MonthlyCharges médio", f"{df_data['MonthlyCharges'].mean():.2f}")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Churn por contrato")
            churn_by_contract = df_data.groupby("Contract")["Churn_num"].mean().sort_values(ascending=False).reset_index()
            churn_by_contract["churn_%"] = churn_by_contract["Churn_num"] * 100
            st.dataframe(churn_by_contract[["Contract", "churn_%"]], width="stretch", hide_index=True)
            st.bar_chart(churn_by_contract.set_index("Contract")["churn_%"])
        with c2:
            st.markdown("### Churn por faixa de tenure")
            bins = [-1, 6, 12, 24, 48, 1000]
            labels = ["0-6", "7-12", "13-24", "25-48", "49+"]
            tmp = df_data.copy()
            tmp["tenure_faixa"] = pd.cut(tmp["tenure"], bins=bins, labels=labels)
            churn_by_tenure = tmp.groupby("tenure_faixa")["Churn_num"].mean().reset_index()
            churn_by_tenure["churn_%"] = churn_by_tenure["Churn_num"] * 100
            st.dataframe(churn_by_tenure, width="stretch", hide_index=True)
            st.line_chart(churn_by_tenure.set_index("tenure_faixa")["churn_%"])

# =========================================================
# 6) Histórico
# =========================================================
elif menu == "🗂 Histórico":
    st.subheader("🗂 Histórico de simulações")

    if len(st.session_state.history) == 0:
        st.write("Sem histórico ainda. Use o Simulador.")
    else:
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist, width="stretch", hide_index=True)
        st.download_button(
            "⬇️ Baixar histórico (CSV)",
            data=df_hist.to_csv(index=False).encode("utf-8"),
            file_name="historico_simulacoes.csv",
            mime="text/csv",
            width="stretch"
        )
        if st.button("🧹 Limpar histórico", width="stretch"):
            st.session_state.history = []
            st.success("Histórico limpo ✅")