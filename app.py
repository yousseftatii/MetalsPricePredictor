import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Metals Price Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Comic+Neue:wght@400;700&family=Press+Start+2P&family=Roboto+Mono:wght@400;700&display=swap');
    /* Main background */
    .stApp {
        background-color: #1c110c; /* Dark brown */
        color: #F5DEB3; /* Light brown text */
    }
    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background-color: #30200a; /* Light brown */
        border-right: 2px solid #8B7355; /* Darker brown border */
    }
            
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    text-align: center;}
            
    /* General text */
    * {
        font-family: 'Roboto Mono', monospace;
        color: #FFFFFF; /* Light brown text */;
    }
            
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Press Start 2P', cursive;
        color: #F5DEB3; /* Light brown text */
        animation: glow 2s infinite alternate;
        text-align: center;
    }
    
    @keyframes glow {
        0% {
            text-shadow: 0 0 5px #8B7355, 0 0 10px #8B7355, 0 0 20px #8B7355;
        }
        100% {
            text-shadow: 0 0 10px #D2B48C, 0 0 20px #D2B48C, 0 0 30px #D2B48C;
        }
    }
    /* Buttons */
    .stButton>button {
        background-color: #8B7355; /* Dark brown */
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 10px #D2B48C, 0 0 20px #D2B48C;
    }
    /* Metric boxes */
    .metric-box {
        background-color: #D2B48C; /* Light brown */
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-box:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    /* Model results */
    .model-results {
        border-left: 3px solid #8B7355; /* Dark brown */
        padding-left: 15px;
        margin: 20px 0;
        transition: border-color 0.3s ease;
    }
    .model-results:hover {
        border-left-color: #D2B48C; /* Light brown */
    }
    /* Sliders and number inputs */
    .stSlider, .stNumberInput {
        margin-bottom: 20px;
    }
    /* Footer */
    .footer {
        text-align: center;
        padding: 10px;
        background-color: #D2B48C; /* Light brown */
        border-radius: 10px;
        margin-top: 20px;
        animation: fadeIn 2s ease-in-out;
    }
    @keyframes fadeIn {
        0% {
            opacity: 0;
        }
        100% {
            opacity: 1;
        }
    }
            
    /* Sidebar button hover effect */
    .stSidebar .stButton>button:hover {
        background-color: #D2B48C; /* Light brown */
        color: #8B7355; /* Dark brown */
    }
    </style>
    """, unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("clean_data.csv")  # Ensure the CSV is in the same directory as app.py
    return df

df = load_data()

# Define the correct path to the Models directory
MODELS_DIR = Path(".")  # Current directory (where app.py is located)

# Model performance data for the best models
BEST_MODEL_METRICS = {
    "Prix Or": {
        "Model" : "RandomForestRegressor",
        "MAE": 36.7712,
        "MSE": 2425.7423,
        "RMSE": 49.2518,
        "R²": 0.9857,
    },
    "PrixArgent": {
        "Model" : "RandomForestRegressor",
        "MAE": 164.7938,
        "MSE": 70212.5697,
        "RMSE": 264.9765,
        "R²": 0.9070,
    }
}

# Define the features used during training for each target
FEATURES_USED_IN_TRAINING = {
    "Prix Or": ['PrixArgent', 'Réserve extérieur', 'Prix Gaz naturel', 'Indice des prix à la consommation'],
    "PrixArgent": ['Prix Or', 'Prix Gaz naturel', 'Export']
}

# Input ranges for user-defined features (based on df.describe())
INPUT_RANGES = {
    "Prix Or": {
        "PrixArgent": {"min": 786.38, "max": 4279.79, "default": 2039.01},
        "Réserve extérieur": {"min": 65063.0, "max": 153075.0, "default": 109130.21},
        "Prix Gaz naturel": {"min": 1.08, "max": 2.64, "default": 1.78},
        "Indice des prix à la consommation": {"min": 198.1, "max": 234.85, "default": 217.63}
    },
    "PrixArgent": {
        "Prix Or": {"min": 476.67, "max": 1770.95, "default": 1114.43},
        "Prix Gaz naturel": {"min": 1.08, "max": 2.64, "default": 1.78},
        "Export": {"min": 107.6, "max": 135.3, "default": 123.4}
    }
}

# Sidebar configuration
st.sidebar.title("Configuration")
target = st.sidebar.selectbox("Sélectionner la Variable Cible", list(BEST_MODEL_METRICS.keys()))

# Add a refresh button in the sidebar
if st.sidebar.button("Actualiser les Données et Modèles"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("Données et modèles actualisés avec succès!")

# Main app
st.title("Metals Price Predictor")
st.markdown("""
Plongez dans l'univers fascinant de la prédiction des prix des métaux précieux grâce à des modèles de machine learning de pointe. 
Cette application révolutionnaire vous permet d'explorer les tendances du marché de l'or et de l'argent avec une précision inégalée. 
Que vous soyez un investisseur chevronné, un analyste financier ou simplement curieux des dynamiques économiques, 
**Metals Price Predictor** vous offre des outils puissants pour anticiper les fluctuations des prix et prendre des décisions éclairées.
""")

# Project Overview and Model Performance Table (Side by Side)
col1, col2 = st.columns([2, 2])
with col1:
    st.subheader("📌 Aperçu du Projet")
    st.write("""
    **Metals Price Predictor** est une application innovante conçue pour prédire les prix de l'or (`Prix Or`) et de l'argent (`PrixArgent`) 
    en utilisant des modèles de machine learning entraînés sur des données historiques. Les performances des modèles sont évaluées 
    à l'aide de métriques robustes.
    """)
    st.write("**Variables Clés :**")
    st.write("""
    - **Prix Or** : Influencé par des facteurs macroéconomiques tels que l'inflation, les réserves étrangères et les prix de l'argent.
    - **PrixArgent** : Dépend des prix de l'or, des exportations industrielles et des coûts énergétiques (prix du gaz naturel).
    """)
    st.write("**Guide d'Utilisation :**")
    st.write("""
    1. **Sélectionnez une variable cible** : Choisissez entre l'or ou l'argent.
    2. **Ajustez les curseurs** : Modifiez les valeurs des variables clés pour explorer différents scénarios.
    3. **Générez la prédiction** : Obtenez des prévisions précises en un clic.
    """)

with col2:
    st.subheader("📊 Performance des Modèles")
    metrics_df = pd.DataFrame(BEST_MODEL_METRICS).T
    st.dataframe(metrics_df)

    st.subheader("📂 Aperçu des Données")
    st.dataframe(df.sample(5))


# Prediction Section (Input Sliders and Prediction Results Side by Side)
col3, col4 = st.columns([3, 3])
with col3:
    with st.form("prediction_form"):
        st.subheader(f"Prédire {target}")
        inputs = {}
        features = FEATURES_USED_IN_TRAINING[target]
        
        for feature in features:
            if feature in INPUT_RANGES[target]:
                inputs[feature] = st.slider(
                    label=feature,
                    min_value=float(INPUT_RANGES[target][feature]["min"]),
                    max_value=float(INPUT_RANGES[target][feature]["max"]),
                    value=float(INPUT_RANGES[target][feature]["default"]),
                    step=0.1
                )
            else:
                st.warning(f"La variable '{feature}' est manquante dans INPUT_RANGES. Utilisation d'une plage par défaut.")
                inputs[feature] = st.slider(
                    label=feature,
                    min_value=0.0,
                    max_value=100.0,
                    value=50.0,
                    step=0.1
                )
        
        submitted = st.form_submit_button("Générer la Prédiction")

with col4:
    if submitted:
        try:
            # Prepare input data as a DataFrame with the exact features used during training
            input_data = pd.DataFrame([inputs], columns=features)
            
            # Load and use scaler
            scaler_path = MODELS_DIR / f"{target.lower()}_scaler_standard_new.pkl"
            if not scaler_path.exists():
                st.error(f"Scaler file for {target} not found at {scaler_path}.")
                st.stop()
            scaler = joblib.load(scaler_path)
            
            # Scale the input data
            scaled_input = scaler.transform(input_data)
            
            # Load model
            model_path = MODELS_DIR / f"{target.lower()}_random_forest_model_new.pkl"
            if not model_path.exists():
                st.error(f"Model file for {target} not found at {model_path}.")
                st.stop()
            model = joblib.load(model_path)
            
            # Make prediction
            prediction = model.predict(scaled_input)[0]
            
            # Display prediction
            st.subheader("Résultats de la Prédiction")
            st.markdown(f"**{target} Prédit:**")
            st.markdown(f"<div class='metric-box' style='font-size: 24px; color: #000000;'>{prediction:,.2f}</div>", unsafe_allow_html=True)
            
            # Prediction Insights
            st.subheader("💡 Analyse de la Prédiction")
            st.write(f"La valeur prédite pour {target} est **{prediction:,.2f}**.")
            st.write("Cette prédiction est basée sur les entrées suivantes :")
            st.write(inputs)
        
        except Exception as e:
            st.error(f"Erreur de prédiction : {str(e)}")



# Custom CSS for timeline and animations
st.markdown("""
<style>
@keyframes slideIn {
  from { transform: translateX(100%); opacity: 0; }
  to { transform: translateX(0); opacity: 1; }
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

.timeline-container {
  position: relative;
  padding: 20px 0;
  max-height: 600px;
  overflow-y: auto;
}

.timeline-step {
  position: relative;
  padding: 25px;
  margin: 0 0 30px 30px;
  border-radius: 10px;
  background: #2a1a0f;
  border-left: 4px solid #8B7355;
  animation: slideIn 0.8s ease-out;
  transition: all 0.3s ease;
  color: white; /* Set all text to white */
}

.timeline-step h3, .timeline-step h4, .timeline-step p, .timeline-step strong, .timeline-step span {
  color: white; /* Ensure all text inside is white */
}

.timeline-step:hover {
  transform: translateY(-5px);
  box-shadow: 0 5px 15px rgba(139, 115, 85, 0.3);
}

.timeline-icon {
  position: absolute;
  left: -45px;
  top: 15px;
  font-size: 24px;
  background: #8B7355;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  animation: fadeIn 1s ease-in;
}

.progress-bar {
  position: fixed;
  bottom: 0;
  left: 0;
  height: 6px; /* Increased height for better visibility */
  background: linear-gradient(90deg, #8B7355, #D2B48C); /* Gradient for visual appeal */
  z-index: 999;
  transition: width 0.3s ease;
  box-shadow: 0 2px 5px rgba(139, 115, 85, 0.5); /* Added shadow for prominence */
}

.progress-bar::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(255, 255, 255, 0.2);
  animation: progress-glow 2s infinite;
}

@keyframes progress-glow {
  0% { opacity: 0.5; }
  50% { opacity: 1; }
  100% { opacity: 0.5; }
}
</style>
""", unsafe_allow_html=True)

# Timeline Container
with st.container():
    st.markdown('<div class="timeline-container">', unsafe_allow_html=True)
    
# Section Documentation : Parcours du Projet
st.markdown("---")
st.subheader("🚀 Parcours Scientifique du Projet")
st.write("Découvrez comment nous avons transformé des données brutes en insights prédictifs")

# Conteneur de la timeline
with st.container():
    st.markdown('<div class="timeline-container">', unsafe_allow_html=True)
    
    # 1. Compréhension du Problème
    st.markdown("""
    <div class="timeline-step">
        <div class="timeline-icon">🔍</div>
        <h3 style='margin-top:0'>1. Cadrage du Problème</h3>
        <div class="timeline-content">
            <p><strong>Enjeux Économiques :</strong><br>
            Les prix de l'or et de l'argent sont des indicateurs clés pour les marchés financiers,
            influençant les stratégies d'investissement et les politiques monétaires.</p>
            <p><strong>Défis Identifiés :</strong><br>
            <span>• Prévisions manuelles peu fiables<br>
            • Complexité des interactions marché<br>
            • Volatilité des matières premières</span></p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Nettoyage des Données
    st.markdown(f"""
    <div class="timeline-step">
        <div class="timeline-icon">🧹</div>
        <h3 style='margin-top:0'>2. Prétraitement des Données</h3>
        <div class="timeline-content">
            <div style='display:grid; grid-template-columns:1fr 1fr; gap:15px'>
                <div style='padding:15px; background:#1c110c; border-radius:8px'>
                    <h4 style='margin:0 0 10px 0'>Données Brutes</h4>
                    <p>• {len(df.columns)+3} variables initiales<br>
                    • Valeurs aberrantes.. <br>
                    • Incohérences temporelles</p>
                </div>
                <div style='padding:15px; background:#1c110c; border-radius:8px'>
                    <h4 style='margin:0 0 10px 0'>Données Nettoyées</h4>
                    <p>• {len(df.columns)} variables pertinentes<br>
                    • Dates standardisées (jour/mois/année)<br>
                    • Correction de l'asymétrie :
                      <span>
                      <br>• Prix Gaz naturel (log)
                      <br>• Prix Café Arabica (log)
                      <br>• Production Industrielle (√)
                      </span>
                    </p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3. Sélection des Variables
    st.markdown("""
    <div class="timeline-step">
        <div class="timeline-icon">🎯</div>
        <h3 style='margin-top:0'>3. Sélection Stratégique des Variables</h3>
        <div class="timeline-content">
            <div style='display:grid; grid-template-columns:1fr 1fr; gap:15px'>
                <div style='padding:15px; background:#1c110c; border-radius:8px'>
                    <h4 style='margin:0 0 10px 0'>Modèle Or</h4>
                    • Prix de l'Argent<br>
                    • Réserves Étrangères<br>
                    • Prix Gaz Naturel<br>
                    • Indice des Prix à la Consommation
                </div>
                <div style='padding:15px; background:#1c110c; border-radius:8px'>
                    <h4 style='margin:0 0 10px 0'>Modèle Argent</h4>
                    • Prix de l'Or<br>
                    • Prix Gaz Naturel<br>
                    • Exportations
                </div>
            </div>
            <p style='margin-top:15px'>
            Sélection basée sur :<br>
            • Analyse de corrélation (Pearson > 0.85)<br>
            • Contexte économique (demande industrielle, inflation)
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4. Analyse des Relations Non-Linéaires
    st.markdown("""
    <div class="timeline-step">
        <div class="timeline-icon">📊</div>
        <h3 style='margin-top:0'>4. Analyse des Relations Complexes</h3>
        <div class="timeline-content">
            <div style='display:grid; grid-template-columns:1fr 1fr; gap:15px'>
                <div style='padding:15px; background:#1c110c; border-radius:8px'>
                    <h4 style='margin:0 0 10px 0'>Or vs Exportations</h4>
                    <p>Relation non-linéaire<br>
                    R² = 0.78</p>
                </div>
                <div style='padding:15px; background:#1c110c; border-radius:8px'>
                    <h4 style='margin:0 0 10px 0'>Argent vs Exportations</h4>
                    <p>Relation non-linéaire<br>
                    R² = 0.82</p>
                </div>
            </div>
            <p style='margin-top:10px'>
            ➔ Les relations entre les exportations et les prix de l'or/argent sont non-linéaires, 
            nécessitant des modèles capables de capturer ces complexités.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 5. Développement des Modèles
    st.markdown("""
    <div class="timeline-step">
        <div class="timeline-icon">🧠</div>
        <h3 style='margin-top:0'>5. Évolution des Modèles</h3>
        <div class="timeline-content">
            <div style='display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:20px'>
                <div style='padding:15px; background:#1c110c; border-radius:8px; text-align:center'>
                    <h4 style='margin:0'>Linéaire</h4>
                    <p style='font-size:0.9em'>R² Or: 0.97<br>Argent: 0.89</p>
                </div>
                <div style='padding:15px; background:#1c110c; border-radius:8px; text-align:center'>
                    <h4 style='margin:0'>Ridge</h4>
                    <p style='font-size:0.9em'>R² Or: 0.97<br>Argent: 0.90</p>
                </div>
                <div style='padding:15px; background:#1c110c; border-radius:8px; text-align:center'>
                    <h4 style='margin:0'>Random Forest</h4>
                    <p style='font-size:0.9em'>R² Or: 0.99<br>Argent: 0.91</p>
                </div>
                <div style='padding:15px; background:#1c110c; border-radius:8px; text-align:center'>
                    <h4 style='margin:0'>XGBoost</h4>
                    <p style='font-size:0.9em'>R² Or: 0.98<br>Argent: 0.91</p>
                </div>
            </div>
            {plot}
        </div>
    </div>
    """.format(plot=px.line(pd.DataFrame({
        "Modèle": ["Linéaire", "Ridge", "Random Forest", "XGBoost"],
        "Or": [0.9738, 0.9730, 0.9857, 0.9841],
        "Argent": [0.8926, 0.8973, 0.9070, 0.9057]
    }), x="Modèle", y=["Or", "Argent"], title="Évolution des Performances").update_layout(
        plot_bgcolor='#1c110c',
        paper_bgcolor='#1c110c',
        font_color='white'  # Set plot text to white
    ).to_html()), unsafe_allow_html=True)



    # 7. Conclusions Scientifiques
    st.markdown("""
    <div class="timeline-step">
        <div class="timeline-icon">🏆</div>
        <h3 style='margin-top:0'>6. Conclusions Clés</h3>
        <div class="timeline-content">
            <div style='padding:15px; background:#1c110c; border-radius:8px'>
                <p><strong>Insights Principaux :</strong><br>
                • Relations non-linéaires dominantes (R² > 0.85)<br>
                • Robustesse aux multicollinéarités (VIF < 5)<br>
                • Performance optimale avec Random Forest</p>
                <p><strong>Implications :</strong><br>
                • Outil fiable pour stratégies de couverture<br>
                • Détection précoce des tendances marché<br>
                • Optimisation des portefeuilles matières premières</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Barre de progression animée
st.markdown('<div class="progress-bar" style="width:100%"></div>', unsafe_allow_html=True)


# References Section
st.markdown("---")
st.write("**Références**")
st.write("- Jeu de Données: [Lien vers le jeu de données](#)")
st.write("- Outils: Streamlit, Plotly, Scikit-learn")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ By | Youssef TATI & Karim Maktouf")
