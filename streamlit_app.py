import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import base64
import time

st.markdown("""
    <style>
    .reportview-container {
        background: #f0f0f5;
    }
    .sidebar .sidebar-content {
        background: #f0f0f5;
    }
    h1 {
        color: #FF5733;
        text-align: center;
        margin-top: 50px;
        margin-left: 100px; /* Centrer le titre tout en laissant de l'espace pour le logo */
    }
    .header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 20px; /* Espace entre le logo et le titre */
    }
    .logo {
        position: relative;
        top: 0;
        left: 0;
        width: 100px;
        margin-right: 20px;
    }
    .prediction-box {
        border: 2px solid #FF5733;
        padding: 10px;
        border-radius: 5px;
        width: 400px; /* Définir une largeur fixe pour la case de prédiction */
        margin-left: 50px; /* Ajuster la marge si nécessaire */
        word-wrap: break-word; /* Pour permettre de couper le texte si nécessaire */
    }
    .button-container {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-top: 20px;
    }
    .footer {
        color: #FF5733;
        text-align: center;
        font-size: 18px;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Fonction pour convertir l'image en base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Ajouter un logo
logo_path = r'C:\Users\souley razak\Desktop\orange-solo.png'  # Chemin vers ton logo
if os.path.exists(logo_path):
    logo_base64 = get_base64_image(logo_path)
else:
    st.error("Logo image not found at the specified path.")
# Afficher le logo et le titre sur la même ligne
st.markdown(f"""
    <div class="header">
        <img src="data:image/png;base64,{logo_base64}" class="logo">
        <h1>Analyse de l'état de la box Wi-Fi</h1>
    </div>
    """, unsafe_allow_html=True)

# Charger le modèle
model_path = r'C:\Users\souley razak\PycharmProjects\Orange Box analysis\myAI5_model.keras'
model = tf.keras.models.load_model(model_path)

# Fonction de prédiction
def predict_box_state(image):
    image = np.array(image) / 255.0  # Normaliser l'image
    image = np.expand_dims(image, axis=0)  # Ajouter une dimension batch
    prediction = model.predict(image)
    class_index = np.argmax(prediction, axis=1)[0]
    return class_index

# Ajouter un conteneur pour le reste du contenu
with st.container():
    uploaded_file = st.file_uploader("Téléchargez une photo de votre box Wi-Fi", type="jpg")

    if uploaded_file:
        image = Image.open(uploaded_file)

        # Créer un layout avec deux colonnes : une pour l'image et une pour la prédiction
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(image, caption="Image téléchargée", use_column_width=True)

        # Prédiction après le téléchargement de l'image
        class_index = predict_box_state(np.array(image))

        with col2:
            # Afficher la classe prédite dans un rectangle de bordure orange
            st.markdown(f"""
                <div class="prediction-box">
                    <p>
                        {' Votre modem est éteint, veuillez effectuer les manipulations suivantes :<br>'
                         'a. Vérifier l’interrupteur du routeur s’il est bien sur « ON »<br>'
                         'b. Débrancher / Rebrancher l’alimentation<br>'
                         'c. Tester sur une autre prise<br>'
                         'Si les voyants du routeur ne s’allument toujours pas: veuillez vous rendre à la boutique pour échanger votre modem' if class_index == 0 else
                        'Votre routeur est bien connecté à internet' if class_index == 1 else
                        ' Vous avez un problème de connexion à internet, veuillez effectuer les manipulations suivantes :<br>'
                        'a. Redémarrer le routeur<br>'
                        'b. Débrancher/rebrancher le câble optique<br>'
                        'Vérifier si le câble optique n’est pas coupé<br>'
                        'Si votre connexion n’est pas rétablie, un technicien va vous contacter dans les prochaines 24h' if class_index == 2 else
                        'Erreur : La classification n\'a pas pu être effectuée correctement.'}
                    </p>
                </div>
                """, unsafe_allow_html=True)

# Gérer le bouton "Quitter"
if 'quit' not in st.session_state:
    st.session_state['quit'] = False

# Bouton "Quitter"
if st.button("Quitter"):
    st.session_state['quit'] = True
    st.success("En espérant avoir résolu votre problème, à bientôt!")
    time.sleep(3)  # Attendre 3 secondes avant de fermer l'application
    st.stop()  # Arrêter l'exécution de Streamlit

# Afficher le message après avoir cliqué sur "Quitter"
if st.session_state['quit']:
    st.success("En espérant avoir résolu votre problème, à bientôt!")

