# ChurnGuard — Previsão de Churn + Plano de Ação (Streamlit)

Aplicação completa (com cara de produto) para **prever churn**, gerar **ranking de prioridade**, sugerir um **playbook de retenção** e exportar listas prontas para o time (CSV).

> Projeto construído para demonstrar capacidade técnica em um contexto real de negócio: dados + modelo + app operacional.

---

## ✅ O que este projeto entrega

### 1) Batch Scoring (CSV)
- Upload de um CSV com clientes
- Cálculo de **probabilidade de churn**
- Classificação de risco (**Baixo / Médio / Alto**)
- **Prioridade score** para ordenar quem deve ser atendido primeiro
- Playbook automático: **ação / canal / oferta sugerida / racional**
- Export CSV completo + export do Top 50

### 2) Plano de Ação (operacional)
- Define **capacidade diária** (quantos contatos o time consegue fazer)
- Gera a **lista do dia** (Top N)
- Campo “copiar e colar” para CRM/WhatsApp
- Métricas de impacto (estimativas):
  - churn esperado no grupo
  - churn evitável (com taxa estimada de retenção)
  - **receita potencial preservada** (aprox.)

### 3) Simulador (1 cliente)
- Ideal para demo em reunião
- Permite simular perfis e ver o risco

### 4) Dashboard (EDA)
- Gráficos e métricas para justificar decisões:
  - churn por contrato
  - churn por faixa de tenure

---

## 🧠 Dataset
Base: **Telco Customer Churn** (Kaggle)  
Arquivo utilizado localmente:
`WA_Fn-UseC_-Telco-Customer-Churn.csv`

---

## 📦 Estrutura do projeto
