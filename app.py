import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# =========================
# Configuração visual básica
# =========================
st.set_page_config(
    page_title="ChurnGuard | Retenção & Crescimento",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

def inject_css():
    st.markdown(
        """
        <style>
          :root {
            --card-bg: rgba(255,255,255,0.04);
            --card-border: rgba(255,255,255,0.10);
            --muted: rgba(255,255,255,0.70);
          }
          .block-container { padding-top: 1.2rem; }
          .small-muted { color: var(--muted); font-size: 0.95rem; }
          .hr { border-top: 1px solid rgba(255,255,255,0.10); margin: 1rem 0; }

          .card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 16px 16px;
          }
          .kpi-title { font-size: 0.85rem; color: var(--muted); margin-bottom: 6px; }
          .kpi-value { font-size: 1.8rem; font-weight: 700; line-height: 1; }
          .kpi-sub { font-size: 0.9rem; color: var(--muted); margin-top: 6px; }

          .badge {
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.15);
            background: rgba(255,255,255,0.04);
            font-size: 0.85rem;
            color: var(--muted);
          }

          .section-title {
            font-size: 1.25rem;
            font-weight: 700;
            margin: 0.2rem 0 0.8rem 0;
          }

          /* deixa os radio/inputs com cara mais “produto” */
          div[data-baseweb="select"] > div,
          div[data-baseweb="input"] > div,
          div[data-baseweb="textarea"] > div {
            border-radius: 12px !important;
          }
          .stButton>button {
            border-radius: 12px;
            padding: 0.6rem 1rem;
            font-weight: 600;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_css()

# =========================
# Carregamento de recursos
# =========================
@st.cache_resource
def load_model_and_columns():
    model = joblib.load("modelo_churn.pkl")
    cols = joblib.load("colunas_modelo.pkl")  # lista de colunas após o preprocess
    return model, cols

@st.cache_data
def load_telco_dataset():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).copy()
    return df

def safe_load():
    try:
        model, model_cols = load_model_and_columns()
    except Exception as e:
        st.error("Não consegui carregar o modelo/colunas. Verifique se 'modelo_churn.pkl' e 'colunas_modelo.pkl' estão no repositório.")
        st.code(str(e))
        st.stop()

    try:
        base = load_telco_dataset()
    except Exception as e:
        st.warning("Não consegui carregar o CSV Telco automaticamente. O app continua, mas algumas páginas (métricas/EDA) vão ficar limitadas.")
        base = None

    return model, model_cols, base

model, model_cols, telco_df = safe_load()

# =========================
# Utilitários de produto
# =========================
def kpi_card(title, value, sub=""):
    st.markdown(
        f"""
        <div class="card">
          <div class="kpi-title">{title}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def to_model_frame(raw_df: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    """
    Ajusta o dataframe para ter exatamente as colunas esperadas pelo modelo,
    cria colunas faltantes com 0, remove extras e garante que não existam NaN/Inf.
    """
    df = raw_df.copy()

    # cria faltantes
    for c in expected_cols:
        if c not in df.columns:
            df[c] = 0

    # mantém só as esperadas
    df = df[expected_cols]

    # garante numérico + remove NaN/Inf
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    return df

def extract_feature_importance(model, feature_names):
    """
    Tenta obter top features para modelos com:
    - feature_importances_ (árvores)
    - coef_ (regressão logística)
    """
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        s = pd.Series(imp, index=feature_names).sort_values(ascending=False)
        return s
    if hasattr(model, "coef_"):
        coef = model.coef_
        if len(coef.shape) == 2 and coef.shape[0] == 1:
            coef = coef[0]
        s = pd.Series(np.abs(coef), index=feature_names).sort_values(ascending=False)
        return s
    return None

def predict_proba_from_input(df_input_processed: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(df_input_processed)[:, 1]
    if hasattr(model, "decision_function"):
        z = model.decision_function(df_input_processed)
        return 1 / (1 + np.exp(-z))
    return model.predict(df_input_processed).astype(float)

def drop_leak_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Remove colunas que não devem entrar na pontuação."""
    df2 = df.copy()
    for col in ["Churn", "customerID"]:
        if col in df2.columns:
            df2 = df2.drop(columns=[col])
    return df2

def coerce_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Garante conversão dos numéricos principais, quando existirem."""
    df2 = df.copy()
    for c in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")
    return df2

# =========================
# Traduções (UI PT -> valores EN que o modelo espera)
# =========================
UI = {
    "Sexo": {"Feminino": "Female", "Masculino": "Male"},
    "Idoso (Senior)": {"Não": 0, "Sim": 1},
    "Tem parceiro(a)": {"Não": "No", "Sim": "Yes"},
    "Tem dependentes": {"Não": "No", "Sim": "Yes"},
    "Serviço de telefone": {"Não": "No", "Sim": "Yes"},
    "Múltiplas linhas": {"Não": "No", "Sim": "Yes", "Sem telefone": "No phone service"},
    "Internet": {"DSL": "DSL", "Fibra": "Fiber optic", "Sem internet": "No"},
    "Segurança online": {"Não": "No", "Sim": "Yes", "Sem internet": "No internet service"},
    "Backup online": {"Não": "No", "Sim": "Yes", "Sem internet": "No internet service"},
    "Proteção do dispositivo": {"Não": "No", "Sim": "Yes", "Sem internet": "No internet service"},
    "Suporte técnico": {"Não": "No", "Sim": "Yes", "Sem internet": "No internet service"},
    "TV por streaming": {"Não": "No", "Sim": "Yes", "Sem internet": "No internet service"},
    "Filmes por streaming": {"Não": "No", "Sim": "Yes", "Sem internet": "No internet service"},
    "Contrato": {"Mensal": "Month-to-month", "1 ano": "One year", "2 anos": "Two year"},
    "Fatura digital": {"Não": "No", "Sim": "Yes"},
    "Forma de pagamento": {
        "Boleto eletrônico": "Electronic check",
        "Cheque enviado": "Mailed check",
        "Cartão (automático)": "Credit card (automatic)",
        "Transferência (automática)": "Bank transfer (automatic)"
    }
}

def build_single_customer_input():
    col1, col2 = st.columns(2)

    with col1:
        gender_pt = st.selectbox("Sexo", list(UI["Sexo"].keys()))
        senior_pt = st.selectbox("Idoso (Senior)", list(UI["Idoso (Senior)"].keys()))
        partner_pt = st.selectbox("Tem parceiro(a)", list(UI["Tem parceiro(a)"].keys()))
        dependents_pt = st.selectbox("Tem dependentes", list(UI["Tem dependentes"].keys()))
        tenure = st.number_input("Tempo de casa (tenure em meses)", min_value=0, max_value=200, value=12, step=1)

    with col2:
        phone_pt = st.selectbox("Serviço de telefone", list(UI["Serviço de telefone"].keys()))
        multilines_pt = st.selectbox("Múltiplas linhas", list(UI["Múltiplas linhas"].keys()))
        internet_pt = st.selectbox("Internet", list(UI["Internet"].keys()))
        contract_pt = st.selectbox("Contrato", list(UI["Contrato"].keys()))
        paperless_pt = st.selectbox("Fatura digital", list(UI["Fatura digital"].keys()))

    col3, col4 = st.columns(2)
    with col3:
        onsec_pt = st.selectbox("Segurança online", list(UI["Segurança online"].keys()))
        onbackup_pt = st.selectbox("Backup online", list(UI["Backup online"].keys()))
        devprot_pt = st.selectbox("Proteção do dispositivo", list(UI["Proteção do dispositivo"].keys()))
        tech_pt = st.selectbox("Suporte técnico", list(UI["Suporte técnico"].keys()))

    with col4:
        tv_pt = st.selectbox("TV por streaming", list(UI["TV por streaming"].keys()))
        movies_pt = st.selectbox("Filmes por streaming", list(UI["Filmes por streaming"].keys()))
        pay_pt = st.selectbox("Forma de pagamento", list(UI["Forma de pagamento"].keys()))
        monthly = st.number_input("Mensalidade (MonthlyCharges)", min_value=0.0, max_value=500.0, value=75.0, step=1.0)
        total = st.number_input("Total pago (TotalCharges)", min_value=0.0, max_value=100000.0, value=1200.0, step=10.0)

    record = {
        "gender": UI["Sexo"][gender_pt],
        "SeniorCitizen": UI["Idoso (Senior)"][senior_pt],
        "Partner": UI["Tem parceiro(a)"][partner_pt],
        "Dependents": UI["Tem dependentes"][dependents_pt],
        "tenure": tenure,
        "PhoneService": UI["Serviço de telefone"][phone_pt],
        "MultipleLines": UI["Múltiplas linhas"][multilines_pt],
        "InternetService": UI["Internet"][internet_pt],
        "OnlineSecurity": UI["Segurança online"][onsec_pt],
        "OnlineBackup": UI["Backup online"][onbackup_pt],
        "DeviceProtection": UI["Proteção do dispositivo"][devprot_pt],
        "TechSupport": UI["Suporte técnico"][tech_pt],
        "StreamingTV": UI["TV por streaming"][tv_pt],
        "StreamingMovies": UI["Filmes por streaming"][movies_pt],
        "Contract": UI["Contrato"][contract_pt],
        "PaperlessBilling": UI["Fatura digital"][paperless_pt],
        "PaymentMethod": UI["Forma de pagamento"][pay_pt],
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }
    return pd.DataFrame([record])

# =========================
# Estado (memória do app)
# =========================
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.50

if "last_batch_scored" not in st.session_state:
    st.session_state.last_batch_scored = None

if "sim_history" not in st.session_state:
    st.session_state.sim_history = []

# =========================
# Sidebar - Identidade + Navegação
# =========================
with st.sidebar:
    st.markdown("### ChurnGuard")
    st.markdown('<span class="badge">Retenção • Priorização • Crescimento</span>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.markdown("**Limiar de decisão (ponto de corte)**")
    st.session_state.threshold = st.slider("", 0.05, 0.95, float(st.session_state.threshold), 0.01)
    st.caption("Quanto maior o limiar, mais ‘rigoroso’ você fica para chamar alguém de risco.")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    page = st.radio(
        "Navegação",
        [
            "📊 Visão Executiva",
            "📥 Upload de Base",
            "🎯 Priorização Inteligente",
            "🧪 Simulação Individual",
            "💰 Impacto Financeiro",
            "📈 Análise Estratégica",
            "🗂 Histórico",
        ],
        label_visibility="visible"
    )

# =========================
# Páginas
# =========================
def page_executiva():
    st.title("ChurnGuard — Retenção & Crescimento")
    st.markdown('<p class="small-muted">Da previsão de churn à priorização e impacto financeiro — com narrativa executiva.</p>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Limiar atual", f"{st.session_state.threshold:.2f}", "Ponto de corte para classificar risco")
    with c2:
        kpi_card("Modo", "Produto", "Fluxo: Upload → Ranking → Ação → ROI")
    with c3:
        status = "Pronto" if st.session_state.last_batch_scored is not None else "Aguardando"
        kpi_card("Base carregada", status, "Use Upload de Base para começar")
    with c4:
        kpi_card("Saída", "CSV + Ações", "Ranking com recomendações")

    st.markdown("### O que este produto entrega")
    st.markdown(
        """
        - **Prioriza clientes por risco** usando modelo treinado  
        - **Transforma score em ação** (segmento, canal, oferta sugerida)  
        - **Compara cenários** (limiar x precisão x ROI)  
        - **Mostra impacto financeiro estimado** (receita salva)  
        """
    )

def page_upload():
    st.header("📥 Upload de Base — entrada operacional")
    st.markdown('<p class="small-muted">Envie um CSV com as colunas do Telco (pode ter extras). A gente calcula o risco e guarda para priorização.</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Envie seu CSV", type=["csv"])
    if uploaded is None:
        st.info("Dica: se você não tiver CSV agora, use o dataset Telco do repositório para testar na página Análise Estratégica.")
        return

    df = pd.read_csv(uploaded)
    df = coerce_numeric_cols(df)

    st.write("Prévia da base:")
    st.dataframe(df.head(20), use_container_width=True)

    st.success("Base carregada. Agora vá em **🎯 Priorização Inteligente** para gerar ranking e ações.")
    st.session_state.upload_df = df

def page_priorizacao():
    st.header("🎯 Priorização Inteligente — ranking + playbook")
    st.markdown('<p class="small-muted">Transforme probabilidade de churn em lista de ação (quem atacar primeiro e como).</p>', unsafe_allow_html=True)

    if "upload_df" not in st.session_state or st.session_state.upload_df is None:
        st.warning("Você ainda não enviou uma base. Vá em **📥 Upload de Base**.")
        return

    df = st.session_state.upload_df.copy()
    df = coerce_numeric_cols(df)
    df_feat = drop_leak_cols(df)

    # One-hot (mesmo estilo usado no app anterior)
    cat_cols = df_feat.select_dtypes(include=["object"]).columns.tolist()
    df_enc = pd.get_dummies(df_feat, columns=cat_cols, drop_first=False)

    X = to_model_frame(df_enc, model_cols)
    proba = predict_proba_from_input(X)

    out = df.copy()
    out["risco_churn"] = proba
    out["classificacao"] = np.where(out["risco_churn"] >= st.session_state.threshold, "ALTO RISCO", "baixo risco")

    def suggest_action(row):
        risk = float(row["risco_churn"])
        contract = row["Contract"] if "Contract" in row else ""
        if risk >= 0.80:
            return "Oferta forte + contato humano (prioridade máxima)"
        if risk >= 0.65:
            return "Retenção: upgrade/benefício + contato (WhatsApp/telefone)"
        if risk >= st.session_state.threshold:
            if contract == "Month-to-month":
                return "Incentivo para migrar para contrato anual"
            return "Ajuste de plano + benefício leve"
        return "Acompanhamento (nurturing) / sem ação"

    out["acao_recomendada"] = out.apply(suggest_action, axis=1)
    out = out.sort_values("risco_churn", ascending=False).reset_index(drop=True)

    st.session_state.last_batch_scored = out

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Clientes na base", f"{len(out):,}".replace(",", "."), "Total avaliados")
    with c2:
        high = (out["classificacao"] == "ALTO RISCO").sum()
        kpi_card("Alto risco", f"{high:,}".replace(",", "."), "Acima do limiar")
    with c3:
        kpi_card("Top prioridade", "Top 50", "Lista pronta para operação")

    st.markdown("### Ranking (Top 50)")
    st.dataframe(out.head(50), use_container_width=True)

    st.download_button(
        "⬇️ Baixar ranking completo (CSV)",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="churnguard_ranking.csv",
        mime="text/csv"
    )

def page_simulacao():
    st.header("🧪 Simulação Individual — cliente único")
    st.markdown('<p class="small-muted">Preencha os dados do cliente e veja risco + recomendação. Tudo em português.</p>', unsafe_allow_html=True)

    df_one = build_single_customer_input()
    df_one = coerce_numeric_cols(df_one)

    df_enc = pd.get_dummies(
        df_one,
        columns=df_one.select_dtypes(include=["object"]).columns.tolist(),
        drop_first=False
    )
    X = to_model_frame(df_enc, model_cols)

    if st.button("✅ Calcular risco"):
        proba = float(predict_proba_from_input(X)[0])
        label = "ALTO RISCO" if proba >= st.session_state.threshold else "baixo risco"

        c1, c2, c3 = st.columns(3)
        with c1:
            kpi_card("Risco de churn", f"{proba:.2%}", "Probabilidade estimada")
        with c2:
            kpi_card("Classificação", label, f"Limiar: {st.session_state.threshold:.2f}")
        with c3:
            kpi_card("Recomendação", "Ação sugerida", "Baseada no nível de risco")

        if proba >= 0.80:
            st.info("Ação sugerida: **contato humano + oferta forte + resolver fricções** (prioridade máxima).")
        elif proba >= st.session_state.threshold:
            st.info("Ação sugerida: **benefício / upgrade / migração de contrato** (prioridade alta).")
        else:
            st.info("Ação sugerida: **nurturing** (acompanhar, ofertas leves, melhoria de experiência).")

        st.session_state.sim_history.append({
            "risco_churn": proba,
            "classificacao": label,
            "limiar": st.session_state.threshold
        })

def page_impacto():
    st.header("💰 Impacto Financeiro — estimativa de receita salva")
    st.markdown('<p class="small-muted">Aqui você traduz modelo em dinheiro.</p>', unsafe_allow_html=True)

    if st.session_state.last_batch_scored is None:
        st.warning("Você ainda não gerou um ranking. Vá em **🎯 Priorização Inteligente**.")
        return

    out = st.session_state.last_batch_scored.copy()

    st.markdown("### Premissas")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        horizonte_meses = st.number_input("Horizonte (meses)", 1, 36, 12, 1)
    with c2:
        taxa_sucesso = st.number_input("Taxa de sucesso da ação (%)", 1, 100, 20, 1) / 100
    with c3:
        custo_por_acao = st.number_input("Custo por ação (R$)", 0.0, 500.0, 12.0, 1.0)
    with c4:
        qtd_acoes = st.number_input("Qtd. de clientes que você consegue acionar", 1, 5000, 300, 10)

    alto = out[out["classificacao"] == "ALTO RISCO"].head(int(qtd_acoes)).copy()
    if "MonthlyCharges" not in alto.columns:
        st.error("Sua base não tem 'MonthlyCharges'. Sem isso não dá para estimar receita salva.")
        return

    receita_potencial = (alto["MonthlyCharges"].fillna(0).sum() * horizonte_meses) * taxa_sucesso
    custo_total = len(alto) * custo_por_acao
    roi = (receita_potencial - custo_total) / (custo_total + 1e-9)

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Receita salva estimada", f"R$ {receita_potencial:,.0f}".replace(",", "."), f"{len(alto)} ações • {int(taxa_sucesso*100)}% sucesso")
    with c2:
        kpi_card("Custo total estimado", f"R$ {custo_total:,.0f}".replace(",", "."), "Custo operacional / campanhas")
    with c3:
        kpi_card("ROI estimado", f"{roi:.1f}x", "Retorno sobre o investimento")

    st.markdown("### Lista de ações (amostra)")
    st.dataframe(alto.head(30), use_container_width=True)

def page_analise():
    st.header("📈 Análise Estratégica — métricas + drivers + cenários")
    st.markdown('<p class="small-muted">Qualidade do modelo + explicabilidade + trade-off do limiar.</p>', unsafe_allow_html=True)

    if telco_df is None:
        st.warning("Sem o CSV Telco carregado. Confirme se 'WA_Fn-UseC_-Telco-Customer-Churn.csv' está no repo.")
        return

    df = telco_df.copy()
    if "Churn" not in df.columns:
        st.error("O dataset precisa da coluna 'Churn'.")
        return

    y = (df["Churn"] == "Yes").astype(int)

    X_raw = drop_leak_cols(df)  # remove Churn/customerID
    X_raw = coerce_numeric_cols(X_raw)

    cat_cols = X_raw.select_dtypes(include=["object"]).columns.tolist()
    X_enc = pd.get_dummies(X_raw, columns=cat_cols, drop_first=False)
    X = to_model_frame(X_enc, model_cols)

    proba = predict_proba_from_input(X)
    y_pred = (proba >= st.session_state.threshold).astype(int)

    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    cm = confusion_matrix(y, y_pred)

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Precision", f"{precision:.2f}", "Dos marcados como risco, quantos eram churn")
    with c2: kpi_card("Recall", f"{recall:.2f}", "Dos churns reais, quantos o modelo pegou")
    with c3: kpi_card("F1-score", f"{f1:.2f}", "Equilíbrio entre precision e recall")
    with c4: kpi_card("Limiar", f"{st.session_state.threshold:.2f}", "Trade-off principal")

    st.markdown("### Matriz de confusão")
    cm_df = pd.DataFrame(
        cm,
        index=["Real: Não churn", "Real: Churn"],
        columns=["Prev: Não churn", "Prev: Churn"]
    )
    st.dataframe(cm_df, use_container_width=True)

    with st.expander("Ver relatório completo (classification report)"):
        st.text(classification_report(y, y_pred, zero_division=0))

    st.markdown("### Top drivers do modelo (features)")
    imp = extract_feature_importance(model, model_cols)
    if imp is None:
        st.info("Seu modelo não expõe feature importance/coeficientes.")
    else:
        top = imp.head(15).reset_index()
        top.columns = ["feature", "importancia"]
        st.dataframe(top, use_container_width=True)
        st.bar_chart(top.set_index("feature")["importancia"])

    st.markdown("### Comparador de cenários — limiar x ROI")
    colA, colB, colC = st.columns(3)
    with colA:
        horizonte = st.number_input("Horizonte (meses) [cenários]", 1, 36, 12, 1, key="hz2")
    with colB:
        sucesso = st.number_input("Sucesso da retenção (%) [cenários]", 1, 100, 20, 1, key="sx2") / 100
    with colC:
        custo_acao = st.number_input("Custo por ação (R$) [cenários]", 0.0, 500.0, 12.0, 1.0, key="ca2")

    thresholds = [0.30, 0.50, 0.70]
    rows = []
    monthly = X_raw["MonthlyCharges"].fillna(0).values if "MonthlyCharges" in X_raw.columns else np.zeros(len(X_raw))

    for t in thresholds:
        yp = (proba >= t).astype(int)
        prec_t = precision_score(y, yp, zero_division=0)
        rec_t = recall_score(y, yp, zero_division=0)
        f1_t = f1_score(y, yp, zero_division=0)

        acoes = int(yp.sum())
        receita = monthly[yp == 1].sum() * horizonte * sucesso
        custo = acoes * custo_acao
        roi_t = (receita - custo) / (custo + 1e-9)

        rows.append({
            "limiar": t,
            "precision": round(prec_t, 3),
            "recall": round(rec_t, 3),
            "f1": round(f1_t, 3),
            "ações": acoes,
            "receita_salva_est": round(receita, 0),
            "custo_est": round(custo, 0),
            "ROI(x)": round(roi_t, 2),
        })

    scen = pd.DataFrame(rows).sort_values("limiar")
    st.dataframe(scen, use_container_width=True)

def page_historico():
    st.header("🗂 Histórico — simulações e execuções")
    st.markdown('<p class="small-muted">Registre evidências: o que foi simulado, quais decisões, qual limiar.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Simulações individuais")
        if len(st.session_state.sim_history) == 0:
            st.info("Sem histórico ainda. Use **🧪 Simulação Individual**.")
        else:
            hist = pd.DataFrame(st.session_state.sim_history)
            st.dataframe(hist.tail(30), use_container_width=True)

    with col2:
        st.markdown("### Último ranking (batch)")
        if st.session_state.last_batch_scored is None:
            st.info("Nenhum ranking ainda. Use **🎯 Priorização Inteligente**.")
        else:
            st.dataframe(st.session_state.last_batch_scored.head(20), use_container_width=True)

# =========================
# Render
# =========================
if page == "📊 Visão Executiva":
    page_executiva()
elif page == "📥 Upload de Base":
    page_upload()
elif page == "🎯 Priorização Inteligente":
    page_priorizacao()
elif page == "🧪 Simulação Individual":
    page_simulacao()
elif page == "💰 Impacto Financeiro":
    page_impacto()
elif page == "📈 Análise Estratégica":
    page_analise()
elif page == "🗂 Histórico":
    page_historico()
