import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# 1. Charger TOUS les préprocesseurs nécessaires
try:
    model = joblib.load('model_plk')
    transformation = joblib.load('transformation.pkl')  # Charger le ColumnTransformer
    normalisation = joblib.load('normalisation.pkl')    # Charger le MinMaxScaler
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {str(e)}")
    st.stop()

# 2. Configuration de la page
st.set_page_config(
    page_title="Prédiction de Diabète",
    page_icon="🩺",
    layout="wide"
)

# 3. Interface utilisateur améliorée
st.title('Prédiction de Diabète')
st.markdown("""
Cette application prédit le risque de diabète en fonction des caractéristiques du patient.
""")

# Section d'information
with st.expander("À propos de cette application"):
    st.markdown("""
    - **Précision**: 95.7%
    - **Données utilisées**: Diabetes Prediction Dataset
    """)

# 4. Organisation des inputs dans des colonnes et sections
with st.form("patient_form"):
    st.header("Informations du patient")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Démographie")
        gender = st.selectbox('Genre', ['Male', 'Female'])
        age = st.slider('Âge', 0, 140, 40)
        
        st.subheader("Mode de vie")
        smoking_history = st.selectbox('Historique tabagique', 
                                      ['never', 'No Info', 'current', 'former', 'not current', 'ever'])
        bmi = st.number_input('IMC', min_value=10.0, max_value=50.0, value=25.0)
    
    with col2:
        st.subheader("Santé cardiovasculaire")
        hypertension = st.radio('Hypertension', [0, 1], format_func=lambda x: 'Oui' if x == 1 else 'Non')
        heart_disease = st.radio('Maladie cardiaque', [0, 1], format_func=lambda x: 'Oui' if x == 1 else 'Non')
        
        st.subheader("Biométrie")
        hba1c = st.slider('Niveau HbA1c (%)', 3.0, 9.0, 5.0, step=0.1)
        glucose = st.slider('Glucose sanguin (mg/dL)', 80, 300, 120)
    
    # Bouton de soumission
    submitted = st.form_submit_button('Prédire le risque de diabète')

# 5. Gestion robuste des prédictions
if submitted:
    # Création du DataFrame
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
        # Transformation des données
        input_transformed = transformation.transform(input_data)
        input_normalized = normalisation.transform(input_transformed)
        
        # Prédiction
        prediction = model.predict(input_normalized)[0]
        probability = model.predict_proba(input_normalized)[0][1]
        
        # Affichage des résultats
        st.header("Résultats de la prédiction")
        
        if prediction == 1:
            st.error(f'Risque de diabète élevé (Probabilité: {probability:.1%})')
            with st.expander("Recommandations médicales"):
                st.markdown("""
                - Consultez un médecin rapidement
                - Adoptez un régime alimentaire équilibré
                - Faites de l'exercice régulièrement
                - Surveillez votre glycémie quotidiennement
                - Évitez les sucres ajoutés et les glucides raffinés
                """)
        else:
            st.success(f'Risque de diabète faible (Probabilité: {probability:.1%})')
            with st.expander("Conseils de prévention"):
                st.markdown("""
                - Maintenez un poids santé
                - Faites au moins 150 minutes d'exercice par semaine
                - Adoptez une alimentation riche en fibres
                - Limitez votre consommation d'alcool
                - Faites des contrôles réguliers
                """)
        
        # Visualisation de la probabilité
        st.subheader("Niveau de risque")
        st.progress(float(probability))
        st.caption(f"Probabilité de diabète: {probability:.1%}")
        
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {str(e)}")

# 6. Section supplémentaire avec des explications
st.markdown("---")
st.header("Comprendre les facteurs de risque")

cols = st.columns(3)
with cols[0]:
    st.info("**HbA1c**")
    st.markdown("Niveau moyen de sucre dans le sang sur 2-3 mois. Un niveau > 6.5% indique un diabète.")
    
with cols[1]:
    st.info("**Glucose sanguin**")
    st.markdown("Niveau de sucre dans le sang à jeun. Un niveau > 126 mg/dL est considéré comme diabétique.")
    
with cols[2]:
    st.info("**IMC**")
    st.markdown("Indice de masse corporelle. Un IMC > 30 augmente significativement le risque de diabète.")

# 7. Pied de page
st.markdown("---")
st.caption("© 2025 - Application de prédiction de diabète | Développé par [Michel B]")