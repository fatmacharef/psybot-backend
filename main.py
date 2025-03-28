from fastapi import FastAPI, HTTPException
import requests
import os
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer
from duckduckgo_search import DDGS

# 📌 Charger le tokenizer de ton modèle PsyBot
tokenizer = AutoTokenizer.from_pretrained("fatmata/psybot")

# 📌 Charger le token Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")

# 🔍 Vérifier que le token est bien chargé
if not HF_TOKEN:
    raise ValueError("🚨 Erreur : La variable d'environnement 'HF_TOKEN' est manquante.")
else:
    print(f"✅ Token HF chargé avec succès : {HF_TOKEN[:6]}... (masqué)")

# 📌 URL de l'API Hugging Face pour ton modèle
HF_MODEL_URL = "https://api-inference.huggingface.co/models/fatmata/psybot"

# 📌 Initialisation de FastAPI
app = FastAPI()

# 📌 Initialisation de l'analyseur VADER
analyzer = SentimentIntensityAnalyzer()

# 📌 Liste des mots-clés pour détecter une recherche
search_keywords = {
    "what", "who", "define", "explain", "is", "how", "causes", "symptoms",
    "treatment", "history", "types", "effects", "meaning", "scientific", "study", "research"
}

# 📌 Liste des mots-clés violents
violent_keywords = {"punch", "hit", "hurt", "kill", "destroy", "break", "explode", "attack"}

# 📌 Modèle pour recevoir l'entrée utilisateur
class UserInput(BaseModel):
    user_input: str

# 📌 Fonction pour tokenizer le texte
def tokenize_text(text):
    return set(tokenizer.tokenize(text.lower()))

# 📌 Fonction pour générer une réponse avec l'API Hugging Face
def generate_response(user_input):
    prompt = f"<|startoftext|><|user|> {user_input} <|bot|>"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
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
        print("🚀 Envoi de la requête à Hugging Face...")
        response = requests.post(HF_MODEL_URL, headers=headers, json=payload)
        print(f"📡 Statut HTTP: {response.status_code}")
        print(f"📡 Réponse brute de HF: {response.text}")

        if response.status_code == 401:
            return "🚨 Erreur 401 : Token Hugging Face invalide ou non autorisé."
        elif response.status_code == 400:
            return "🚨 Erreur 400 : Mauvaise requête. Vérifie le format des données envoyées."

        response.raise_for_status()
        response_json = response.json()

        if isinstance(response_json, dict) and "error" in response_json:
            return f"⚠️ Erreur API : {response_json['error']}"

        if isinstance(response_json, list) and len(response_json) > 0 and "generated_text" in response_json[0]:
            generated_text = response_json[0]["generated_text"]
            print(f"🤖 Réponse générée : {generated_text}")

            # Vérification pour éviter le copier-coller
            clean_response = generated_text.replace(prompt, "").strip()
            return clean_response if clean_response else "Désolé, je ne peux pas répondre pour le moment."

        return "Désolé, je ne peux pas répondre pour le moment."
    except requests.exceptions.RequestException as e:
        return f"🛑 Erreur de connexion à Hugging Face : {str(e)}"

# 📌 Fonction de recherche avec DuckDuckGo
def search_duckduckgo(query, max_results=3):
    try:
        search_results = list(DDGS().text(query, max_results=max_results))
        return [result["body"] for result in search_results if "body" in result] or ["Je n'ai pas trouvé d'informations sur ce sujet."]
    except Exception as e:
        return [f"Erreur de recherche : {str(e)}"]

# 📌 Fonction de classification et réponse
def classify_and_respond(text):
    print(f"🔍 Message reçu : {text}")
    try:
        tokens = tokenize_text(text)
        print(f"✅ Tokens : {tokens}")

        # 🔍 Vérifier si la question est une recherche
        if tokens.intersection(search_keywords) or text.endswith('?'):
            return search_duckduckgo(text)

        # 🔍 Vérification du sentiment avec VADER
        vader_score = analyzer.polarity_scores(text)["compound"]
        print(f"🧠 Score VADER : {vader_score}")

        # 🔴 Vérification des mots violents
        if any(word in text.lower().split() for word in violent_keywords):
            return ["🔴 Non Accepté: Essayez de vous calmer. La violence ne résout rien."]

        # 🚀 Génération de réponse via Hugging Face
        response = generate_response(text)
        print(f"🤖 Réponse GPT : {response}")
        return [f"🟢 Accepté: {response}"]

    except Exception as e:
        print(f"❌ Erreur classification : {e}")
        return ["⚠️ Une erreur est survenue dans la classification du message."]

# 📌 Endpoint API
@app.post("/chat/")
async def chat_with_bot(user_input: UserInput):
    return {"response": classify_and_respond(user_input.user_input)}

@app.get("/")
async def home():
    return {"message": "PsyBot API is running!"}
