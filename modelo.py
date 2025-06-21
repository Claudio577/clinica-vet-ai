import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import unicodedata
import re

def normalizar_texto(texto):
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8').lower()
    texto = re.sub(r'[^\w\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def carregar_dados():
    df = pd.read_csv("data/Casos_Cl_nicos_Simulados.csv")
    df_doencas = pd.read_csv("data/doencas_caninas_eutanasia_expandidas.csv")
    return df, df_doencas

def treinar_modelos(df, features, features_eutanasia, df_doencas):
    le_mob = LabelEncoder()
    le_app = LabelEncoder()

    df['Mobilidade'] = le_mob.fit_transform(df['Mobilidade'].str.lower().str.strip())
    df['Apetite'] = le_app.fit_transform(df['Apetite'].str.lower().str.strip())

    palavras_chave = [
        unicodedata.normalize('NFKD', d).encode('ASCII', 'ignore').decode('utf-8').lower().strip()
        for d in df_doencas['Doença'].dropna().unique()
    ]

    df['tem_doenca_letal'] = df['Doença'].fillna("").apply(
        lambda d: int(any(p in unicodedata.normalize('NFKD', d).encode('ASCII', 'ignore').decode('utf-8').lower()
                          for p in palavras_chave))
    )

    X_eutanasia = df[features_eutanasia]
    y_eutanasia = df['Eutanasia']

    modelo_eutanasia = RandomForestClassifier(class_weight='balanced', random_state=42)
    modelo_eutanasia.fit(X_eutanasia, y_eutanasia)

    modelo_alta = RandomForestClassifier().fit(df[features], df['Alta'])
    modelo_internar = RandomForestClassifier(class_weight='balanced', random_state=42)
    modelo_internar.fit(df[features], df['Internar'])

    modelo_dias = RandomForestRegressor().fit(df[df['Internar'] == 1][features], df[df['Internar'] == 1]['Dias Internado'])

    return modelo_eutanasia, modelo_alta, modelo_internar, modelo_dias, le_mob, le_app, palavras_chave
