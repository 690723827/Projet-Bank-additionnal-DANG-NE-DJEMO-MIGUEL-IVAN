import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from utils import add_bg_from_local, load_and_preprocess_data, train_model
from sklearn.metrics import classification_report, confusion_matrix

# Page Configuration
st.set_page_config(
    page_title="Classification des Données Bancaires",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'a_propos'

# Enable dark theme for Altair
alt.themes.enable("dark")

# Sidebar Navigation
def set_page_selection(page):
    st.session_state.page_selection = page

with st.sidebar:
    st.title('🏦 Classification des Données Bancaires')
    
    st.subheader("Sections")
    for page in ['À Propos', 'Jeu de Données', 'Analyse Exploratoire', 'Prédiction', 'Conclusion']:
        if st.button(page, use_container_width=True, 
                    on_click=set_page_selection, 
                    args=(page.lower().replace(' ', '_'),)):
            pass

    st.subheader("Résumé")
    st.markdown("""
        Un tableau de bord interactif pour explorer et classifier les données 
        d'une campagne marketing bancaire.

        - [Jeu de Données](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
        - [Notebook Google Colab](https://colab.research.google.com/drive/1KJDBrx3akSPUW42Kbeepj64ZisHFD-NV)
        - [Dépôt GitHub](https://github.com/teguegni/bank-additionnal-full)

        Auteur : DANGNE DJEMO MIGUEL IVAN
    """)

# Load data
try:
    df = load_and_preprocess_data('bank-additional-full.csv')
except Exception as e:
    st.error(f"Erreur lors du chargement des données: {e}")
    st.stop()

# Page Content
if st.session_state.page_selection == 'a_propos':
    st.title("️ À Propos")
    st.markdown("""
        Cette application explore le jeu de données Bank Marketing et propose :

        - Une exploration visuelle des données
        - Un prétraitement et nettoyage des données
        - La construction et l'évaluation de modèles d'apprentissage automatique
        - Une interface interactive pour prédire si un client souscrira à un produit

        Technologies utilisées :
        - Python (Streamlit, Altair, Pandas)
        - Machine Learning (Scikit-learn)

        Auteur : DANG-NE DJEMO MIGUEL IVAN
        ✉️ Contact : dangneivanmiguel@gmail.com
    """)

elif st.session_state.page_selection == 'jeu_de_donnees':
    st.title("📊 Jeu de Données")
    
    if st.checkbox("Afficher le DataFrame"):
        nb_rows = st.slider("Nombre de lignes à afficher :", 
                           min_value=5, max_value=len(df), value=10)
        st.write(df.head(nb_rows))

    if st.checkbox("Afficher les statistiques descriptives"):
        st.write(df.describe())

    # Interactive visualization section
    with st.expander("Options de Visualisation"):
        col1, col2 = st.columns(2)
        with col1:
            selected_x = st.selectbox("Axe X", df.select_dtypes(include=['number']).columns)
        with col2:
            selected_y = st.selectbox("Axe Y", df.select_dtypes(include=['number']).columns)
        
        plot_type = st.radio("Type de graphique", 
                            ["Nuage de points", "Histogramme", "Boxplot"])
    
    # Create dynamic charts based on selection
    if plot_type == "Nuage de points":
        chart = alt.Chart(df).mark_circle(size=60).encode(
            x=selected_x,
            y=selected_y,
            color='y',
            tooltip=list(df.columns)
        ).interactive()
        st.altair_chart(chart, use_container_width=True)

elif st.session_state.page_selection == 'analyse_exploratoire':
    st.title("🔍 Analyse Exploratoire")
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for column in categorical_cols:
        st.subheader(f"Distribution de {column}")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df, x=column, hue='y', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close()

elif st.session_state.page_selection == 'prediction':
    st.title("🔮 Prédiction")
    
    # Input form
    age = st.number_input("Âge du client", min_value=18, max_value=120, value=30)
    duration = st.number_input("Durée du contact (seconds)", min_value=0, value=60)
    campaign = st.number_input("Nombre de contacts", min_value=1, value=1)
    
    if st.button("Prédire"):
        try:
            model, _, _, _, _ = train_model(df)
            prediction = model.predict([[age, duration, campaign]])
            result = "Oui" if prediction[0] == 1 else "Non"
            st.success(f"Prédiction de souscription: {result}")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction: {e}")

elif st.session_state.page_selection == 'conclusion':
    st.title("📝 Conclusion")
    st.markdown("""
        Un traitement minutieux et réfléchi du DataFrame bank-additional-full est 
        fondamental pour maximiser la précision et la fiabilité du modèle de prédiction. 
        En combinant explorations, prétraitements adéquats, et évaluations rigoureuses, 
        on peut développer un modèle robuste qui est mieux équipé pour prédire les 
        comportements des clients envers la souscription à un produit.
    """)
import base64
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def add_bg_from_local(image_file):
    """Add background image to the app"""
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset"""
    df = pd.read_csv(file_path, delimiter=';')
    
    # Replace unknown values with mode
    for column in df.columns:
        if df[column].dtype == 'object':
            mode_value = df[column].mode()[0]
            df[column] = df[column].replace('unknown', mode_value)
    
    # Encode categorical variables
    label_encoder = LabelEncoder()
    categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 
                         'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])
        
    return df

def train_model(df):
    """Train the machine learning model"""
    X = df[['age', 'duration', 'campaign']]
    y = df['y'].map({'yes': 1, 'no': 0})
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test
