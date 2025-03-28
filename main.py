from fastapi import FastAPI, HTTPException
import requests
import os
from pydantic import BaseModel
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from duckduckgo_search import DDGS

# 📌 Définir un chemin local pour télécharger les ressources NLTK
NLTK_DIR = "/opt/render/nltk_data"
os.makedirs(NLTK_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DIR)

# 📌 Vérifier et télécharger les ressources NLTK manquantes
nltk_resources = ['punkt', 'wordnet', 'vader_lexicon']
for resource in nltk_resources:
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
    except LookupError:
        print(f"📥 Téléchargement de {resource}...")
        nltk.download(resource, download_dir=NLTK_DIR)

# 📌 Charger le token Hugging Face depuis la variable d’environnement
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("🚨 Erreur : La variable d'environnement 'HF_TOKEN' est manquante.")

# 📌 URL de l'API Hugging Face pour ton modèle PsyBot
HF_MODEL_URL = "https://api-inference.huggingface.co/models/fatmata/psybot"

# 📌 Initialisation de FastAPI
app = FastAPI()

# 📌 Initialisation de l'analyseur VADER
analyzer = SentimentIntensityAnalyzer()

# 📌 Mots-clés pour la détection des recherches
search_keywords = {
    "what", "who", "define", "explain", "is", "how", "causes", "symptoms",
    "treatment", "history", "types", "effects", "meaning", "scientific", "study", "research"
}

# 📌 Modèle pour recevoir l'entrée utilisateur
class UserInput(BaseModel):
    user_input: str

# 📌 Fonction pour générer une réponse avec l'API Hugging Face
def generate_response(user_input):
    prompt = f"<|startoftext|><|user|> {user_input} <|bot|>"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 50,
            "do_sample": True,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.2
        }
    }
    try:
        response = requests.post(HF_MODEL_URL, headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()

        # ✅ Correction de l'extraction du texte généré
        if isinstance(response_json, list) and len(response_json) > 0:
            generated_text = response_json[0].get('generated_text', '')
            return generated_text.split("<|bot|>")[-1].strip() if "<|bot|>" in generated_text else generated_text
        return "Désolé, je ne peux pas répondre pour le moment."
    
    except requests.exceptions.RequestException as e:
        return f"Erreur lors de la communication avec le modèle : {str(e)}"

# 📌 Fonction de recherche avec DuckDuckGo
def search_duckduckgo(query, max_results=3):
    try:
        search_results = list(DDGS().text(query, max_results=max_results))
        return [result["body"] for result in search_results if "body" in result] or ["Je n'ai pas trouvé d'informations sur ce sujet."]
    except Exception as e:
        return [f"Erreur de recherche : {str(e)}"]

# 📌 Fonction de classification et réponse
def classify_and_respond(text):
    try:
        print(f"🔍 Message reçu : {text}")

        # 🔹 Vérifier la tokenization
        try:
            tokens = set(word_tokenize(text.lower()))
            print(f"✅ Tokens : {tokens}")
        except Exception as e:
            print(f"❌ Erreur tokenisation : {e}")
            return ["⚠️ Erreur tokenisation"]
        
        # 🔹 Vérifier si c'est une recherche
        if tokens.intersection(search_keywords) or text.endswith('?'):
            return search_duckduckgo(text)
        
        # 🔹 Analyse du sentiment avec VADER
        vader_score = analyzer.polarity_scores(text)["compound"]
        print(f"🧠 Score VADER : {vader_score}")
        
        # 🔹 Bloquer les messages violents
        violent_keywords = {"punch", "hit", "hurt", "kill", "destroy", "break", "explode", "attack"}
        if any(word in text.lower() for word in violent_keywords):
            return ["🔴 Non Accepté: Essayez de vous calmer. La violence ne résout rien."]
        
        # 🔹 Si la requête est acceptable, utiliser GPT
        response = generate_response(text)
        print(f"🤖 Réponse GPT : {response}")
        return [f"🟢 Accepté: {response}"]
    
    except Exception as e:
        print(f"❌ Erreur classification : {e}")
        return ["⚠️ Une erreur est survenue dans la classification du message."]

# 📌 Endpoint principal de l'API
@app.post("/chat/")
async def chat_with_bot(user_input: UserInput):
    try:
        response = classify_and_respond(user_input.user_input)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 📌 Endpoint de test pour voir si l'API tourne bien
@app.get("/")
async def home():
    return {"message": "PsyBot API is running!"}
