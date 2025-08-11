# 🩺 Prédiction de Diabète avec Régression Logistique

## 📌 Description

Ce projet utilise un modèle de **régression logistique** pour prédire le risque de diabète à partir de données médicales et de mode de vie d'un patient. Il comprend :

* Un **Notebook Jupyter** pour l'entraînement, l'évaluation et la sauvegarde du modèle.
* Une **application Streamlit** interactive permettant à l'utilisateur de saisir ses informations et d'obtenir une prédiction en temps réel.

---

## 📂 Structure du projet

```
📁 Projet_Diabete
 ├── diabetes_prediction_nb.ipynb     # Notebook d'entraînement
 ├── app.py                           # Application Streamlit
 ├── model_plk                        # Modèle entraîné
 ├── transformation.pkl               # OneHotEncoder sauvegardé
 ├── normalisation.pkl                 # MinMaxScaler sauvegardé
 ├── historique.csv                   # Historique des prédictions (auto-généré)
 ├── diabetes_prediction_dataset.csv  # Jeu de données utilisé
 └── requirements.txt                 # Dépendances Python
```

---

## ⚙️ Installation

1. **Cloner le dépôt**

```bash
git clone <url_du_repo>
cd Diabete_Regression
```

2. **Créer un environnement virtuel**

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

3. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

---

## 📊 Utilisation

### 1. Entraîner le modèle

Ouvrir le **notebook** dans Jupyter et exécuter toutes les cellules :

```bash
jupyter notebook diabetes_prediction_nb.ipynb
```

Cela va :

* Nettoyer les données
* Entraîner la régression logistique
* Sauvegarder `model_plk`, `transformation.pkl` et `normalisation.pkl`

### 2. Lancer l'application Streamlit

```bash
streamlit run app.py
```

Puis ouvrir l'URL générée (généralement `http://localhost:8501`).

---

## 🖥 Fonctionnalités de l'application

* Formulaire clair avec sections **Démographie**, **Mode de vie**, **Santé cardiovasculaire** et **Biométrie**
* Prédiction du risque avec **probabilité**
* **Graphique radar interactif** pour visualiser la position par rapport aux seuils
* Sauvegarde automatique des prédictions dans un fichier CSV
* Explications médicales vulgarisées pour les principaux indicateurs

---

## 📈 Performances

* **Précision entraînement** : \~95%
* **Précision test** : \~95%
* **F1-score** : \~71%

*(Les performances peuvent varier selon les données et le prétraitement appliqué)*

---

## 📌 Notes importantes

* Ce modèle est à visée **éducative** et ne remplace pas un diagnostic médical.
* Les recommandations affichées sont **génériques** et ne constituent pas un avis médical personnalisé.

---

## 👤 Auteurs

Projet développé par **\Michel B. et revu par Destin B.** — 2025
