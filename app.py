import os
import io
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
# Page config
# =========================
st.set_page_config(
    page_title="ChurnGuard | Retenção & Crescimento",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Premium Dark CSS
# =========================
def inject_css():
    st.markdown(
        """
        <style>
          :root{
            --bg: #0B0F17;
            --panel: rgba(255,255,255,0.055);
            --panel2: rgba(255,255,255,0.035);
            --stroke: rgba(255,255,255,0.10);
            --text: rgba(255,255,255,0.92);
            --muted: rgba(255,255,255,0.72);
            --muted2: rgba(255,255,255,0.55);
            --brand: #7C3AED;   /* roxo premium */
            --brand2: #22D3EE;  /* ciano detalhe */
            --danger: #FB7185;
            --warn: #FBBF24;
            --ok: #34D399;
            --shadow2: 0 10px 26px rgba(0,0,0,0.28);
            --radius: 18px;
          }

          .stApp {
            background:
              radial-gradient(1100px 520px at 12% 10%, rgba(124,58,237,0.20), transparent 55%),
              radial-gradient(900px 460px at 86% 18%, rgba(34,211,238,0.12), transparent 52%),
              linear-gradient(180deg, #070A10 0%, var(--bg) 35%, #070A10 100%);
            color: var(--text);
          }

          .block-container { padding-top: 1.2rem; padding-bottom: 2.4rem; max-width: 1280px; }

          section[data-testid="stSidebar"]{
            background: linear-gradient(180deg, rgba(255,255,255,0.040) 0%, rgba(255,255,255,0.020) 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
          }

          h1, h2, h3, h4 { letter-spacing: -0.02em; }

          .small-muted { color: var(--muted); font-size: 0.98rem; }
          .tiny-muted  { color: var(--muted2); font-size: 0.88rem; }
          .hr { border-top: 1px solid rgba(255,255,255,0.10); margin: 1rem 0; }

          .hero {
            background: linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
            border: 1px solid rgba(255,255,255,0.11);
            border-radius: 22px;
            padding: 18px 18px;
            box-shadow: var(--shadow2);
            position: relative;
            overflow: hidden;
          }
          .hero:before{
            content:"";
            position:absolute; inset:-2px;
            background: radial-gradient(680px 260px at 15% 20%, rgba(124,58,237,0.22), transparent 55%);
            pointer-events:none;
          }
          .hero .title { font-size: 1.55rem; font-weight: 850; margin:0; }
          .hero .subtitle { margin-top: 6px; color: var(--muted); }

          .badge {
            display:inline-flex; align-items:center; gap:8px;
            padding: 6px 12px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.14);
            background: rgba(255,255,255,0.04);
            font-size: 0.86rem;
            color: var(--muted);
          }
          .dot { width:8px; height:8px; border-radius: 999px; background: var(--brand); box-shadow: 0 0 16px rgba(124,58,237,0.55); }

          .card {
            background: linear-gradient(180deg, var(--panel), var(--panel2));
            border: 1px solid var(--stroke);
            border-radius: var(--radius);
            padding: 16px 16px;
            box-shadow: var(--shadow2);
          }
          .card:hover{
            border-color: rgba(255,255,255,0.16);
            transform: translateY(-1px);
            transition: 160ms ease;
          }
          .kpi-title { font-size: 0.86rem; color: var(--muted); margin-bottom: 6px; }
          .kpi-value { font-size: 1.9rem; font-weight: 850; line-height: 1.05; }
          .kpi-sub   { font-size: 0.92rem; color: var(--muted2); margin-top: 8px; }

          div[data-baseweb="select"] > div,
          div[data-baseweb="input"] > div,
          div[data-baseweb="textarea"] > div {
            border-radius: 14px !important;
            border-color: rgba(255,255,255,0.14) !important;
            background: rgba(255,255,255,0.03) !important;
          }

          .stButton>button {
            border-radius: 14px;
            padding: 0.62rem 1.05rem;
            font-weight: 760;
            border: 1px solid rgba(255,255,255,0.14);
            background: linear-gradient(180deg, rgba(124,58,237,0.95), rgba(124,58,237,0.72));
            box-shadow: 0 10px 26px rgba(124,58,237,0.28);
          }
          .stButton>button:hover {
            filter: brightness(1.06);
            box-shadow: 0 12px 30px rgba(124,58,237,0.35);
          }

          div[data-testid="stDataFrame"]{
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.10);
            box-shadow: var(--shadow2);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_css()

# =========================
# Load artifacts
# =========================
@st.cache_resource
def load_artifacts():
    model = joblib.load("modelo_churn.pkl")
    model_columns = joblib.load("colunas_modelo.pkl")
    return model, model_columns

model, model_columns = load_artifacts()

@st.cache_data
def load_telco(path="WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    df = pd.read_csv(path)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).copy()
    return df

try:
    telco_df = load_telco()
except Exception:
    telco_df = None

# =========================
# Helpers: encode to model columns (auto-detect drop_first)
# =========================
def _try_match_drop_first(df_raw: pd.DataFrame, model_cols: list[str]) -> bool:
    """
    Testa se o padrão drop_first=True ou False casa melhor com colunas_modelo.pkl
    """
    # candidate True
    enc_true = pd.get_dummies(df_raw, drop_first=True)
    s_true = set(enc_true.columns)
    # candidate False
    enc_false = pd.get_dummies(df_raw, drop_first=False)
    s_false = set(enc_false.columns)

    target = set(model_cols)

    # score = overlap - missing
    score_true = len(target.intersection(s_true)) - len(target.difference(s_true))
    score_false = len(target.intersection(s_false)) - len(target.difference(s_false))

    return score_true >= score_false  # True = drop_first True, else False

@st.cache_data
def detect_drop_first_from_telco(model_cols: list[str]) -> bool:
    """
    Usa uma amostra do Telco para inferir qual get_dummies foi usado no treino.
    Se não tiver Telco, assume True (mais comum), mas o app segue funcionando.
    """
    if telco_df is None:
        return True

    df = telco_df.copy()
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # pega uma amostra pequena
    df_sample = df.head(200).copy()
    return _try_match_drop_first(df_sample, model_cols)

DROP_FIRST = detect_drop_first_from_telco(model_columns)

def build_input_encoded(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_encoded = pd.get_dummies(df_raw, drop_first=DROP_FIRST)
    df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)
    return df_encoded

def predict_proba_df(df_encoded: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(df_encoded)[:, 1]
    if hasattr(model, "decision_function"):
        z = model.decision_function(df_encoded)
        return 1 / (1 + np.exp(-z))
    return model.predict(df_encoded).astype(float)

def risk_bucket(prob: float, threshold: float) -> str:
    if prob < threshold:
        return "Baixo"
    if prob < min(threshold + 0.20, 0.95):
        return "Médio"
    return "Alto"

def risk_badge(bucket: str) -> str:
    return {"Alto": "🔴 Alto", "Médio": "🟡 Médio", "Baixo": "🟢 Baixo"}[bucket]

def feature_importance(model, cols: list[str]) -> pd.DataFrame | None:
    if hasattr(model, "coef_"):
        coef = model.coef_[0] if len(model.coef_.shape) == 2 else model.coef_
        imp = pd.DataFrame({"feature": cols, "importancia": np.abs(coef)})
        return imp.sort_values("importancia", ascending=False)
    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({"feature": cols, "importancia": model.feature_importances_})
        return imp.sort_values("importancia", ascending=False)
    return None

# =========================
# PT-BR UI mappings (para valores do dataset)
# =========================
MAP_SEXO = {"Feminino": "Female", "Masculino": "Male"}
MAP_SIMNAO = {"Sim": "Yes", "Não": "No"}
MAP_CONTRATO = {"Mensal": "Month-to-month", "1 ano": "One year", "2 anos": "Two year"}
MAP_INTERNET = {"DSL": "DSL", "Fibra óptica": "Fiber optic", "Sem internet": "No"}
MAP_MULTILINHAS = {"Não": "No", "Sim": "Yes", "Sem telefone": "No phone service"}
MAP_SEM_INTERNET = {"Sim": "Yes", "Não": "No", "Sem internet": "No internet service"}
MAP_PAGAMENTO = {
    "Boleto eletrônico": "Electronic check",
    "Cheque enviado": "Mailed check",
    "Transferência (automática)": "Bank transfer (automatic)",
    "Cartão (automático)": "Credit card (automatic)",
}

# =========================
# Upload: colunas em PT (UX), conversão para o formato esperado
# =========================
PT_TEMPLATE_COLS = [
    "Sexo","Idoso","Tem parceiro","Tem dependentes","Tempo como cliente (meses)",
    "Serviço de telefone","Múltiplas linhas","Internet",
    "Segurança online","Backup online","Proteção do dispositivo","Suporte técnico",
    "TV por streaming","Filmes por streaming",
    "Contrato","Fatura digital","Forma de pagamento",
    "Mensalidade (R$)","Total pago (R$)"
]

def make_pt_template():
    return pd.DataFrame([{
        "Sexo": "Feminino",
        "Idoso": "Não",
        "Tem parceiro": "Sim",
        "Tem dependentes": "Sim",
        "Tempo como cliente (meses)": 12,
        "Serviço de telefone": "Sim",
        "Múltiplas linhas": "Não",
        "Internet": "DSL",
        "Segurança online": "Sim",
        "Backup online": "Sim",
        "Proteção do dispositivo": "Sim",
        "Suporte técnico": "Sim",
        "TV por streaming": "Sim",
        "Filmes por streaming": "Sim",
        "Contrato": "Mensal",
        "Fatura digital": "Sim",
        "Forma de pagamento": "Boleto eletrônico",
        "Mensalidade (R$)": 75.0,
        "Total pago (R$)": 1200.0,
    }])[PT_TEMPLATE_COLS]

def normalize_pt_upload_to_en(df_pt: pd.DataFrame) -> pd.DataFrame:
    # garante colunas
    missing = [c for c in PT_TEMPLATE_COLS if c not in df_pt.columns]
    if missing:
        raise ValueError("Faltando colunas no CSV: " + ", ".join(missing))

    df = df_pt.copy()

    # converte tipos numéricos
    df["Tempo como cliente (meses)"] = pd.to_numeric(df["Tempo como cliente (meses)"], errors="coerce")
    df["Mensalidade (R$)"] = pd.to_numeric(df["Mensalidade (R$)"], errors="coerce")
    df["Total pago (R$)"] = pd.to_numeric(df["Total pago (R$)"], errors="coerce")

    # mapeia para o formato Telco (EN)
    out = pd.DataFrame({
        "gender": df["Sexo"].map(MAP_SEXO),
        "SeniorCitizen": df["Idoso"].map({"Não": 0, "Sim": 1}),
        "Partner": df["Tem parceiro"].map(MAP_SIMNAO),
        "Dependents": df["Tem dependentes"].map(MAP_SIMNAO),
        "tenure": df["Tempo como cliente (meses)"].astype("Int64"),
        "PhoneService": df["Serviço de telefone"].map(MAP_SIMNAO),
        "MultipleLines": df["Múltiplas linhas"].map(MAP_MULTILINHAS),
        "InternetService": df["Internet"].map(MAP_INTERNET),
        "OnlineSecurity": df["Segurança online"].map(MAP_SEM_INTERNET),
        "OnlineBackup": df["Backup online"].map(MAP_SEM_INTERNET),
        "DeviceProtection": df["Proteção do dispositivo"].map(MAP_SEM_INTERNET),
        "TechSupport": df["Suporte técnico"].map(MAP_SEM_INTERNET),
        "StreamingTV": df["TV por streaming"].map(MAP_SEM_INTERNET),
        "StreamingMovies": df["Filmes por streaming"].map(MAP_SEM_INTERNET),
        "Contract": df["Contrato"].map(MAP_CONTRATO),
        "PaperlessBilling": df["Fatura digital"].map(MAP_SIMNAO),
        "PaymentMethod": df["Forma de pagamento"].map(MAP_PAGAMENTO),
        "MonthlyCharges": df["Mensalidade (R$)"],
        "TotalCharges": df["Total pago (R$)"],
    })

    # validações simples
    if out.isna().any().any():
        # identifica primeiras colunas com NaN
        cols_nan = out.columns[out.isna().any()].tolist()
        raise ValueError(
            "Há valores inválidos (não reconhecidos) no CSV. Verifique principalmente: "
            + ", ".join(cols_nan)
        )

    return out

# =========================
# App State
# =========================
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.50
if "batch_scored" not in st.session_state:
    st.session_state.batch_scored = None
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# Sidebar (produto)
# =========================
with st.sidebar:
    st.markdown("### ChurnGuard")
    st.markdown('<span class="badge"><span class="dot"></span> Retenção • Crescimento • ROI</span>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.markdown("**Limiar de decisão (ponto de corte)**")
    st.session_state.threshold = st.slider("", 0.05, 0.95, float(st.session_state.threshold), 0.01)
    st.caption("Limiar menor = pega mais churn (mais ações). Limiar maior = menos ações (mais seletivo).")

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
        ]
    )

# =========================
# UI building blocks
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
        unsafe_allow_html=True,
    )

def hero(title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
          <div class="title">{title}</div>
          <div class="subtitle">{subtitle}</div>
          <div style="margin-top:10px" class="badge"><span class="dot"></span> Produto • Analytics • Ação • ROI</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")

def playbook_action(row_en: dict, bucket: str) -> str:
    contract = row_en.get("Contract", "")
    tenure = float(row_en.get("tenure", 0) or 0)
    monthly = float(row_en.get("MonthlyCharges", 0) or 0)
    payment = row_en.get("PaymentMethod", "")

    if bucket == "Baixo":
        return "Acompanhar normalmente + reforçar proposta de valor (benefícios e educação do cliente)."

    if bucket == "Médio":
        msg = "Contato proativo + oferta leve (benefício/upgrade). "
        if contract == "Month-to-month":
            msg += "Sugerir migração para plano anual com incentivo. "
        if tenure <= 12:
            msg += "Reforçar onboarding + suporte para reduzir fricções. "
        if monthly >= 70:
            msg += "Ajustar plano/pacote para melhorar custo-benefício percebido. "
        return msg.strip()

    # Alto
    msg = "Ação imediata (até 24h): contato humano + oferta direcionada. "
    if contract == "Month-to-month":
        msg += "Priorizar migração para 1 ano/2 anos com incentivo. "
    if payment == "Electronic check":
        msg += "Oferecer troca para pagamento automático com bônus. "
    if tenure <= 6:
        msg += "Check-in de experiência e remoção rápida de atritos. "
    return msg.strip()

# =========================
# PAGES
# =========================
def page_executiva():
    hero("ChurnGuard — Revenue Intelligence", "Do score de churn à priorização, ação e ROI (pronto para operação comercial).")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Limiar atual", f"{st.session_state.threshold:.2f}", "Ponto de corte para classificar risco")
    with c2:
        kpi_card("Codificação", "Auto-detect", f"get_dummies(drop_first={DROP_FIRST})")
    with c3:
        status = "Pronto ✅" if st.session_state.batch_scored is not None else "Aguardando"
        kpi_card("Base processada", status, "Faça upload e gere ranking")
    with c4:
        kpi_card("Saída", "Ranking + ROI", "Export CSV para time comercial")

    st.markdown("### Entregáveis do produto")
    st.markdown(
        """
        - **Pontuação em lote (base de clientes)** com risco e recomendação  
        - **Priorização Top 50/Top N** para operação de retenção  
        - **Impacto financeiro (ROI)** para justificar investimento  
        - **Métricas do modelo** (Precisão, Recall, F1, Matriz + Curva PR)  
        - **Drivers do churn** (top features) e **análise por segmentos** (heatmap)  
        """
    )

def page_upload():
    st.header("📥 Upload de Base")
    st.markdown('<p class="small-muted">Aqui o usuário trabalha 100% em português. O sistema converte para o padrão do modelo por baixo.</p>', unsafe_allow_html=True)

    template = make_pt_template()
    st.download_button(
        "⬇️ Baixar template (CSV em português)",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="template_clientes_churn_ptbr.csv",
        mime="text/csv",
    )

    up = st.file_uploader("Envie seu CSV preenchido (em português)", type=["csv"])
    if up is None:
        st.info("Dica: baixe o template acima, preencha e envie aqui.")
        return

    df_pt = pd.read_csv(up)
    st.write("Prévia do arquivo enviado:")
    st.dataframe(df_pt.head(20), use_container_width=True)

    try:
        df_en = normalize_pt_upload_to_en(df_pt)
    except Exception as e:
        st.error("Não consegui converter seu CSV para o padrão do modelo.")
        st.code(str(e))
        return

    st.success("Conversão OK ✅ Agora podemos pontuar e gerar ranking.")
    st.session_state.upload_df_en = df_en

def page_priorizacao():
    st.header("🎯 Priorização Inteligente")
    st.markdown('<p class="small-muted">Gera ranking de clientes com risco, classificação e playbook recomendado.</p>', unsafe_allow_html=True)

    if "upload_df_en" not in st.session_state:
        st.warning("Você ainda não enviou uma base. Vá em **📥 Upload de Base**.")
        return

    df_en = st.session_state.upload_df_en.copy()

    X = build_input_encoded(df_en)
    proba = predict_proba_df(X)

    out = df_en.copy()
    out["prob_churn"] = proba
    out["risco"] = out["prob_churn"].apply(lambda p: risk_bucket(float(p), st.session_state.threshold))
    out["badge"] = out["risco"].map(risk_badge)
    out["acao_recomendada"] = out.apply(lambda r: playbook_action(r.to_dict(), r["risco"]), axis=1)

    out = out.sort_values("prob_churn", ascending=False).reset_index(drop=True)
    st.session_state.batch_scored = out

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Clientes avaliados", f"{len(out):,}".replace(",", "."), "Total na base")
    with c2:
        kpi_card("Alto risco", f"{(out['risco']=='Alto').sum():,}".replace(",", "."), "Prioridade máxima")
    with c3:
        kpi_card("Médio risco", f"{(out['risco']=='Médio').sum():,}".replace(",", "."), "Ação proativa")
    with c4:
        kpi_card("Baixo risco", f"{(out['risco']=='Baixo').sum():,}".replace(",", "."), "Acompanhar")

    topn = st.selectbox("Mostrar ranking Top", [25, 50, 100, 200], index=1)
    st.markdown("### Ranking (prioridade)")
    st.dataframe(out.head(int(topn)), use_container_width=True)

    st.download_button(
        "⬇️ Baixar ranking completo (CSV)",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="churnguard_ranking.csv",
        mime="text/csv",
    )

def page_simulacao():
    st.header("🧪 Simulação Individual")
    st.markdown('<p class="small-muted">Preencha os dados e veja risco + recomendação. Tudo em português.</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        sexo = st.selectbox("Sexo", list(MAP_SEXO.keys()))
        idoso = st.selectbox("Idoso", ["Não", "Sim"])
        parceiro = st.selectbox("Tem parceiro(a)?", list(MAP_SIMNAO.keys()))
        dependentes = st.selectbox("Tem dependentes?", list(MAP_SIMNAO.keys()))
        tenure = st.number_input("Tempo como cliente (meses)", 0, 200, 12, 1)

    with col2:
        telefone = st.selectbox("Serviço de telefone", list(MAP_SIMNAO.keys()))
        multilinhas = st.selectbox("Múltiplas linhas", list(MAP_MULTILINHAS.keys()))
        internet = st.selectbox("Internet", list(MAP_INTERNET.keys()))
        contrato = st.selectbox("Contrato", list(MAP_CONTRATO.keys()))
        fatura = st.selectbox("Fatura digital", list(MAP_SIMNAO.keys()))

    with col3:
        seg = st.selectbox("Segurança online", list(MAP_SEM_INTERNET.keys()))
        backup = st.selectbox("Backup online", list(MAP_SEM_INTERNET.keys()))
        prot = st.selectbox("Proteção do dispositivo", list(MAP_SEM_INTERNET.keys()))
        suporte = st.selectbox("Suporte técnico", list(MAP_SEM_INTERNET.keys()))
        tv = st.selectbox("TV por streaming", list(MAP_SEM_INTERNET.keys()))
        filmes = st.selectbox("Filmes por streaming", list(MAP_SEM_INTERNET.keys()))
        pagamento = st.selectbox("Forma de pagamento", list(MAP_PAGAMENTO.keys()))
        mensal = st.number_input("Mensalidade (R$)", 0.0, 500.0, 75.0, 1.0)
        total = st.number_input("Total pago (R$)", 0.0, 100000.0, 1200.0, 10.0)

    df_en = pd.DataFrame([{
        "gender": MAP_SEXO[sexo],
        "SeniorCitizen": 1 if idoso == "Sim" else 0,
        "Partner": MAP_SIMNAO[parceiro],
        "Dependents": MAP_SIMNAO[dependentes],
        "tenure": int(tenure),
        "PhoneService": MAP_SIMNAO[telefone],
        "MultipleLines": MAP_MULTILINHAS[multilinhas],
        "InternetService": MAP_INTERNET[internet],
        "OnlineSecurity": MAP_SEM_INTERNET[seg],
        "OnlineBackup": MAP_SEM_INTERNET[backup],
        "DeviceProtection": MAP_SEM_INTERNET[prot],
        "TechSupport": MAP_SEM_INTERNET[suporte],
        "StreamingTV": MAP_SEM_INTERNET[tv],
        "StreamingMovies": MAP_SEM_INTERNET[filmes],
        "Contract": MAP_CONTRATO[contrato],
        "PaperlessBilling": MAP_SIMNAO[fatura],
        "PaymentMethod": MAP_PAGAMENTO[pagamento],
        "MonthlyCharges": float(mensal),
        "TotalCharges": float(total),
    }])

    if st.button("✅ Calcular risco"):
        X = build_input_encoded(df_en)
        prob = float(predict_proba_df(X)[0])
        bucket = risk_bucket(prob, st.session_state.threshold)
        rec = playbook_action(df_en.iloc[0].to_dict(), bucket)

        c1, c2, c3 = st.columns(3)
        with c1:
            kpi_card("Probabilidade de churn", f"{prob*100:.1f}%", "Estimativa do modelo")
        with c2:
            kpi_card("Classificação", risk_badge(bucket), f"Limiar: {st.session_state.threshold:.2f}")
        with c3:
            kpi_card("Ação sugerida", "Playbook", "Pronto para o time comercial")

        st.progress(min(max(prob, 0.0), 1.0))
        st.info(rec)

        st.session_state.history.insert(0, {
            "prob_churn": prob,
            "risco": bucket,
            "limiar": float(st.session_state.threshold)
        })

def page_impacto():
    st.header("💰 Impacto Financeiro (ROI de Retenção)")
    st.markdown('<p class="small-muted">Aqui você traduz modelo em dinheiro — o que mais impressiona em entrevista.</p>', unsafe_allow_html=True)

    if st.session_state.batch_scored is None:
        st.warning("Você ainda não gerou o ranking. Vá em **🎯 Priorização Inteligente**.")
        return

    out = st.session_state.batch_scored.copy()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        horizonte = st.number_input("Horizonte (meses)", 1, 36, 12, 1)
    with c2:
        sucesso = st.number_input("Sucesso da retenção (%)", 1, 100, 20, 1) / 100
    with c3:
        custo = st.number_input("Custo por ação (R$)", 0.0, 500.0, 12.0, 1.0)
    with c4:
        topn = st.number_input("Qtd. de clientes acionados (Top N)", 10, 5000, 300, 10)

    foco = out.sort_values("prob_churn", ascending=False).head(int(topn)).copy()

    receita_salva = (foco["MonthlyCharges"].sum() * horizonte) * sucesso
    custo_total = len(foco) * custo
    roi = (receita_salva - custo_total) / (custo_total + 1e-9)

    a, b, c_ = st.columns(3)
    with a:
        kpi_card("Receita salva estimada", f"R$ {receita_salva:,.0f}".replace(",", "."), f"{int(sucesso*100)}% sucesso • {horizonte} meses")
    with b:
        kpi_card("Custo total estimado", f"R$ {custo_total:,.0f}".replace(",", "."), f"{len(foco)} ações")
    with c_:
        kpi_card("ROI estimado", f"{roi:.2f}x", "Retorno sobre investimento")

    st.markdown("### Lista de ações (amostra)")
    st.dataframe(foco.head(50), use_container_width=True)

def page_analise():
    st.header("📈 Análise Estratégica (Métricas + Drivers + Segmentos)")
    st.markdown('<p class="small-muted">Prova de maturidade: qualidade do modelo, trade-offs e segmentação por valor.</p>', unsafe_allow_html=True)

    if telco_df is None:
        st.warning("Não encontrei o dataset Telco no repositório. Para esta página, confirme se o arquivo CSV está no repo.")
        return

    df = telco_df.copy()
    if "Churn" not in df.columns:
        st.error("O dataset precisa da coluna 'Churn'.")
        return

    y = (df["Churn"] == "Yes").astype(int)
    X_raw = df.drop(columns=["Churn"]).copy()
    if "customerID" in X_raw.columns:
        X_raw = X_raw.drop(columns=["customerID"])

    X = build_input_encoded(X_raw)
    proba = predict_proba_df(X)
    y_pred = (proba >= st.session_state.threshold).astype(int)

    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    cm = confusion_matrix(y, y_pred)

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Precisão", f"{precision:.2f}", "Dos marcados como risco, quantos eram churn")
    with c2: kpi_card("Recall (Cobertura)", f"{recall:.2f}", "Dos churns reais, quantos o modelo capturou")
    with c3: kpi_card("F1-score", f"{f1:.2f}", "Equilíbrio entre precisão e recall")
    with c4: kpi_card("Limiar", f"{st.session_state.threshold:.2f}", "Trade-off principal")

    st.markdown("### Matriz de confusão")
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title("Matriz de confusão")
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_xticklabels(["Não churn","Churn"])
    ax.set_yticklabels(["Não churn","Churn"])
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha="center", va="center")
    st.pyplot(fig, clear_figure=True)

    st.markdown("### Curva Precision-Recall")
    fig2, ax2 = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y, proba, ax=ax2)
    ax2.set_title("Precision-Recall (quanto menor o limiar, maior o recall e maior o volume de ações)")
    st.pyplot(fig2, clear_figure=True)

    st.markdown("### Top features (drivers do churn)")
    imp = feature_importance(model, model_columns)
    if imp is None:
        st.info("Seu modelo não expõe coeficientes/importância. Se quiser, depois eu te ajudo a gerar um modelo explicável mantendo o deploy.")
    else:
        top = imp.head(15).copy()
        st.dataframe(top, use_container_width=True)
        st.bar_chart(top.set_index("feature")["importancia"])

    st.markdown("### Heatmap: churn por contrato x tempo de casa")
    # bins de tenure
    tenure = pd.to_numeric(df["tenure"], errors="coerce").fillna(0)
    bins = [0, 6, 12, 24, 48, 72, 999]
    labels = ["0-6", "7-12", "13-24", "25-48", "49-72", "73+"]
    df_h = df.copy()
    df_h["tenure_faixa"] = pd.cut(tenure, bins=bins, labels=labels, include_lowest=True)
    df_h["churn_num"] = (df_h["Churn"] == "Yes").astype(int)

    pivot = df_h.pivot_table(index="Contract", columns="tenure_faixa", values="churn_num", aggfunc="mean").fillna(0)

    fig3, ax3 = plt.subplots(figsize=(8, 3))
    im = ax3.imshow(pivot.values, aspect="auto")
    ax3.set_title("Taxa média de churn")
    ax3.set_yticks(range(len(pivot.index)))
    ax3.set_yticklabels(pivot.index.tolist())
    ax3.set_xticks(range(len(pivot.columns)))
    ax3.set_xticklabels(pivot.columns.tolist())
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax3.text(j, i, f"{pivot.values[i,j]*100:.0f}%", ha="center", va="center")
    st.pyplot(fig3, clear_figure=True)

    st.markdown("### Comparador de cenários: limiar x ROI")
    st.caption("Mostra claramente o trade-off: limiar baixo = mais ações (mais custo) e maior cobertura; limiar alto = menos ações (mais seletivo).")

    colA, colB, colC = st.columns(3)
    with colA:
        hz = st.number_input("Horizonte (meses) — cenários", 1, 36, 12, 1, key="hz_scen")
    with colB:
        sx = st.number_input("Sucesso (%) — cenários", 1, 100, 20, 1, key="sx_scen") / 100
    with colC:
        ca = st.number_input("Custo por ação (R$) — cenários", 0.0, 500.0, 12.0, 1.0, key="ca_scen")

    thresholds = [0.30, 0.50, 0.70]
    rows = []
    monthly = pd.to_numeric(df.get("MonthlyCharges", pd.Series(np.zeros(len(df)))), errors="coerce").fillna(0).values

    for t in thresholds:
        yp = (proba >= t).astype(int)
        acoes = int(yp.sum())
        receita = monthly[yp == 1].sum() * hz * sx
        custo_total = acoes * ca
        roi = (receita - custo_total) / (custo_total + 1e-9)

        rows.append({
            "Limiar": t,
            "Ações (clientes acionados)": acoes,
            "Receita salva estimada (R$)": round(receita, 0),
            "Custo estimado (R$)": round(custo_total, 0),
            "ROI (x)": round(roi, 2),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

def page_historico():
    st.header("🗂 Histórico")
    st.markdown('<p class="small-muted">Registro das simulações feitas no app.</p>', unsafe_allow_html=True)

    if len(st.session_state.history) == 0:
        st.info("Sem histórico ainda. Use **🧪 Simulação Individual**.")
        return

    st.dataframe(pd.DataFrame(st.session_state.history).head(50), use_container_width=True)

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
