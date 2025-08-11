import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Charger le modèle et les préprocesseurs
model = joblib.load('model_plk')
# Note: Vous devrez aussi sauvegarder/charger vos transformations (ColumnTransformer et MinMaxScaler)

# Interface utilisateur
st.title('🔮 Prédiction de Diabète')
st.subheader('Entrez les informations du patient :')

# Création des champs de saisie
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox('Genre', ['Male', 'Female'])
    age = st.slider('Âge', 0, 100, 40)
    hypertension = st.radio('Hypertension', [0, 1])
    heart_disease = st.radio('Maladie cardiaque', [0, 1])
    
with col2:
    smoking_history = st.selectbox('Historique tabagique', 
                                  ['never', 'No Info', 'current', 'former'])
    bmi = st.number_input('Indice de la masse corporelle (IMC)', min_value=10.0, max_value=50.0, value=25.0)
    hba1c = st.slider('Niveau HbA1c', 3.0, 9.0, 5.0)
    glucose = st.slider('Glucose sanguin', 80, 300, 120)

# Bouton de prédiction
if st.button('Prédire le risque de diabète'):
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
    
    # Faire la prédiction
    prediction = model.predict(input_data)[0]  # Remplacer par input_normalized
    probability = model.predict_proba(input_data)[0][1]  # Remplacer par input_normalized
    
    # Afficher résultat
    st.subheader('Résultats :')
    if prediction == 1:
        st.error(f'Risque de diabète élevé ({probability:.1%} probabilité)')
        st.markdown('Recommandations : ')
        st.markdown('- Consultation médicale urgente')
        st.markdown('- Régime alimentaire contrôlé')
        st.markdown('- Surveillance régulière de la glycémie')
    else:
        st.success(f'Risque de diabète faible ({probability:.1%} probabilité)')
        st.markdown('💡 Conseils de prévention :')
        st.markdown('- Exercice régulier')
        st.markdown('- Alimentation équilibrée')
        st.markdown('- Contrôles annuels')

# Information supplémentaire
st.markdown("---")
st.info("""
Note technique :
- Précision : 95.7%
- F1-score : 72.3%
""")