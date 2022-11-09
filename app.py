# import module
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import pickle
import random
from preprocess import preprocess_text, ADJECTIVES_LIST
from textblob import TextBlob

dataset_df = pd.read_csv("./dataset_cleaned.csv")
features_nmf_df = pd.read_csv("./features_nmf.csv").set_index('Unnamed: 0')
TOPIC_INTERPRETATION = { "topic0": "TABLES/PERSONNEL DE SERVICE",
                            "topic1": "GOUTS DES PLATS",
                            "topic2": "LIVRAISON ET QUALITE DES PIZZAS",
                            "topic3": "STAFF/MANAGEMENT",
                            "topic4": "QUALITE DE LA NOURRITURE ET DU SERVICE",
                            "topic5": "MENU BURGERS",
                            "topic6": "COMMANDES ET LIVRAISONS",
                            "topic7": "TEMPS D'ATTENTE",
                            "topic8": "QUALITE DES PLATS DE POULETS",
                            "topic9": "QUALITE DU BAR ET DES BOISSONS",
                            "topic10": "RAPPORT QUANTITE/PRIX",
                            "topic11": "MENUS SUSHIS",
                            "topic12": "MENUS SANDWICHS",
                            "topic13": "ILS SONT DEJAS VENUS",
                            "topic14": "CADRE/ENVIRONNEMENT"}

VECTORIZER_PATH = "./tfidf.vec"
NMF_MODEL_PATH = "./nmf.model15"

vectorizer = pickle.load(open(VECTORIZER_PATH, 'rb'))
model = pickle.load(open(NMF_MODEL_PATH, 'rb'))


def predict_topic_from_text(text, n_topic=1):
    polarity = TextBlob(text).polarity
    topic_label_list = list()

    if polarity < 0.5:
        preprocessed_text = preprocess_text(text=text)
        preprocessed_text = " ".join([word for word in preprocessed_text.split() if word not in ADJECTIVES_LIST])

        # Transform the TF-IDF
        X = vectorizer.transform([preprocessed_text])

        # Transform the TF-IDF: nmf_features
        nmf_features = model.transform(X)

        # Get topic ID
        # topic_id = pd.DataFrame(nmf_features).idxmax(axis=1)
        topic_id = nmf_features.argmax(axis=1)[0]
        topic_id_list = nmf_features.argsort(axis=1).tolist()[0][::-1][:n_topic]
        # print(topic_id_list, [TOPIC_INTERPRETATION["topic{}".format(topic_id)] for topic_id in topic_id_list])
        topic_label_list = [TOPIC_INTERPRETATION["topic{}".format(topic_id)] for topic_id in topic_id_list]

    return topic_label_list, polarity


def predict_topic_from_dataset(text, n_topic=1):
    
    topic_id = features_nmf_df.loc[text].idxmax()
    topic_id_list = np.array(features_nmf_df.loc[text].values).argsort()[::-1][:n_topic]
    # print(topic_id_list, [TOPIC_INTERPRETATION["topic{}".format(topic_id)] for topic_id in topic_id_list])
    topic_label_list = [TOPIC_INTERPRETATION["topic{}".format(topic_id)] for topic_id in topic_id_list]

    return topic_label_list

# Title
st.title("ReviewAnalyzer")
img = Image.open("topic_modeling.png")
st.image(img, width=600)

# Selection box

# first argument takes the titleof the selectionbox
# second argument takes options
#review_id = st.selectbox("ID: ", dataset_df.index.tolist())

global review_text_index
type = st.sidebar.radio("Quel texte analyser ?", ('Avis dataset', 'Texte libre'))
with st.sidebar.beta_expander("Prediction des avis du dataset"):

    review_text = random.choice(features_nmf_df.index.tolist())
    review_id = st.number_input("Numéro d'index", min_value=0, max_value=9999, value=0)

    review_text_index = features_nmf_df.index[review_id]

    if (st.button("Choix aléatoire d'un avis")):
        #st.write("Contenu de l'avis: ", review_text)
        st.info("Contenu de l'avis: {}".format(review_text))

    if (st.button("Choix d'un avis suivant le numéro d'index")):

        #st.write("Contenu de l'avis: ", review_text)
        st.info("Contenu de l'avis: {}".format(review_text_index))

with st.sidebar.beta_expander("Prediction de nouveaux avis"):
    review_text_free = st.text_area('Entrez un texte')

# Create a button, that when clicked, shows a text
n_topic = st.number_input("Nombre de topics", min_value=1, max_value=15, value=1)
if (st.button("Détecter le sujet d'insatisfaction")):

    if type == "Avis dataset":
        st.success("TOPIC: {}".format(predict_topic_from_dataset(text=review_text_index, n_topic=n_topic)))
    elif type == "Texte libre":
        topic_labels, polarity = predict_topic_from_text(text=review_text_free, n_topic=n_topic)
        if polarity > 0.5:
            st.success("POLARITE: {} (COMMENTAIRE POSTIF)".format(round(polarity, 2)))
        else:
            st.error("POLARITE: {} (COMMENTAIRE NEGATIF)".format(round(polarity, 2)))
            st.error("TOPIC: {}".format(topic_labels))
    #st.success("TOPIC: {}".format(predict_topic_from_dataset(text=review_text)))

#st.write("TOPIC: ", predict_topic(text=review_text_cleaned))