import streamlit as st
import numpy as np
import pandas as pd
from modelo import (
    carregar_dados,
    treinar_modelos,
    normalizar_texto
)

def prever(anamnese, modelos, le_mob, le_app, palavras_chave, features, features_eutanasia):
    texto_norm = normalizar_texto(anamnese)

    idade = 8
    peso = 18
    temperatura = 38.8
    gravidade = 5
    dor = 5

    if any(p in texto_norm for p in ["sem apetite", "não come"]):
        apetite = le_app.transform(["nenhum"])[0] if "nenhum" in le_app.classes_ else 0
    else:
        apetite = le_app.transform(["normal"])[0] if "normal" in le_app.classes_ else 0

    if any(p in texto_norm for p in ["não anda", "mobilidade limitada"]):
        mobilidade = le_mob.transform(["limitada"])[0] if "limitada" in le_mob.classes_ else 0
    else:
        mobilidade = le_mob.transform(["normal"])[0] if "normal" in le_mob.classes_ else 0

    doencas_detectadas = [p for p in palavras_chave if p in texto_norm]
    tem_doenca_letal = int(len(doencas_detectadas) > 0)

    entrada = pd.DataFrame([[idade, peso, gravidade, dor, mobilidade, apetite, temperatura, tem_doenca_letal]],
                           columns=features_eutanasia)

    modelo_eutanasia, modelo_alta, modelo_internar, modelo_dias = modelos[:4]

    prob_eutanasia = modelo_eutanasia.predict_proba(entrada)[0][1]
    prob_internar = modelo_internar.predict_proba(entrada[features])[0][1]

    alta = modelo_alta.predict(entrada[features])[0]
    internar = 1 if prob_internar > 0.4 else 0
    dias = int(round(modelo_dias.predict(entrada[features])[0])) if internar == 1 else 0

    return {
        "Alta": "Sim" if alta else "Não",
        "Internar": "Sim" if internar else "Não",
        "Dias Internado": dias,
        "Chance de Eutanásia (%)": round(prob_eutanasia * 100, 1),
        "Doenças Detectadas": doencas_detectadas or ["Nenhuma grave"]
    }

# === Streamlit Interface ===

st.set_page_config(page_title="Análise Veterinária", layout="centered")
st.title("🐾 Análise Clínica Veterinária com IA")
st.markdown("Insira a anamnese do paciente para prever os cuidados clínicos e analisar doenças mencionadas.")

try:
    df, df_doencas = carregar_dados()
    features = ['Idade', 'Peso', 'Gravidade', 'Dor', 'Mobilidade', 'Apetite', 'Temperatura']
    features_eutanasia = features + ['tem_doenca_letal']
    modelos = treinar_modelos(df, features, features_eutanasia, df_doencas)
except Exception as e:
    st.error(f"Erro ao carregar dados ou treinar modelos: {e}")
    st.stop()

texto = st.text_area("✍️ Digite a anamnese do paciente:")

if st.button("🔍 Analisar"):
    if texto.strip() == "":
        st.warning("Digite uma anamnese para analisar.")
    else:
        resultado = prever(texto, modelos, modelos[4], modelos[5], modelos[6], features, features_eutanasia)
        st.subheader("📋 Resultado da Análise")
        for k, v in resultado.items():
            st.write(f"**{k}**: {v}")

