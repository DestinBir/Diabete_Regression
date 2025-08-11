import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import os
from datetime import datetime


# 1. Chargement du modèle et préprocesseurs

try:
    model = joblib.load('model_plk')
    transformation = joblib.load('transformation.pkl')
    normalisation = joblib.load('normalisation.pkl')
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.stop()


# 2. Configuration de la page

st.set_page_config(page_title="Prédiction de Diabète", page_icon="🩺", layout="wide")

st.title("Prédiction du Risque de Diabète")

# Tabs pour organisation
tabs = st.tabs(["Formulaire patient", "Résultats", "Informations"])


# 3. Formulaire

with tabs[0]:
    with st.form("patient_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Démographie")
            gender = st.selectbox('Genre', ['Male', 'Female'])
            age = st.slider('Âge', 0, 120, 40)

            st.subheader("Mode de vie")
            smoking_history = st.selectbox('Historique tabagique', ['never', 'No Info', 'current', 'former', 'not current', 'ever'])
            bmi = st.number_input('IMC', min_value=10.0, max_value=50.0, value=25.0)

        with col2:
            st.subheader("Santé cardiovasculaire")
            hypertension = st.radio('Hypertension', [0, 1], format_func=lambda x: 'Oui' if x == 1 else 'Non')
            heart_disease = st.radio('Maladie cardiaque', [0, 1], format_func=lambda x: 'Oui' if x == 1 else 'Non')

            st.subheader("Biométrie")
            hba1c = st.slider('Niveau HbA1c (%)', 3.0, 9.0, 5.0, step=0.1)
            glucose = st.slider('Glucose sanguin (mg/dL)', 80, 300, 120)

        submitted = st.form_submit_button("Prédire")


# 4. Résultats et explications

with tabs[1]:
    if submitted:
        input_data = pd.DataFrame({
            'gender': [gender],
            'age': [age],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'smoking_history': [smoking_history],
            'bmi': [bmi],
            'HbA1c_level': [hba1c],
            'blood_glucose_level': [glucose]
        })

        try:
            # Transformation et normalisation
            transformed = transformation.transform(input_data)
            normalized = normalisation.transform(transformed)

            prediction = model.predict(normalized)[0]
            probability = model.predict_proba(normalized)[0][1]

            # Affichage du résultat
            if prediction == 1:
                st.error(f"Risque de diabète ÉLEVÉ ({probability:.1%})")
                st.markdown("### Recommandations")
                st.write("- Consultez un médecin rapidement")
                st.write("- Adoptez un régime équilibré et faites de l'exercice")
                st.write("- Surveillez régulièrement votre glycémie")
            else:
                st.success(f"Risque de diabète FAIBLE ({probability:.1%})")
                st.markdown("### Conseils de prévention")
                st.write("- Maintenez un poids santé et une alimentation saine")
                st.write("- Faites au moins 150 minutes d'exercice par semaine")
                st.write("- Limitez les sucres ajoutés et l'alcool")

            # Graphique radar des facteurs
            factors = {
                "Âge": age / 120,
                "IMC": bmi / 50,
                "HbA1c": hba1c / 9,
                "Glucose": glucose / 300
            }

            radar_df = pd.DataFrame({
                'Facteur': list(factors.keys()),
                'Valeur normalisée': list(factors.values())
            })

            fig = px.line_polar(radar_df, r='Valeur normalisée', theta='Facteur', line_close=True)
            fig.update_traces(fill='toself')
            st.plotly_chart(fig, use_container_width=True)

            # Sauvegarde historique
            hist_file = "historique.csv"
            new_entry = input_data.copy()
            new_entry["probability"] = probability
            new_entry["prediction"] = prediction
            new_entry["date"] = datetime.now()

            if os.path.exists(hist_file):
                pd.concat([pd.read_csv(hist_file), new_entry]).to_csv(hist_file, index=False)
            else:
                new_entry.to_csv(hist_file, index=False)

        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")


# 5. Informations

with tabs[2]:
    st.markdown("## Comprendre les facteurs de risque")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**HbA1c**")
        st.markdown("> Mesure la moyenne du sucre dans le sang sur 2-3 mois. >6.5% indique souvent un diabète.")
    with col2:
        st.info("**Glucose**")
        st.markdown("> Taux de sucre sanguin à jeun. >126 mg/dL = diabète.")
    with col3:
        st.info("**IMC**")
        st.markdown("> >30 = obésité, augmente fortement le risque de diabète.")

    st.markdown("---")
    st.caption("© 2025 - Application de prédiction de diabète | Michel B.")
