from fastapi import FastAPI
import requests
import os
from pydantic import BaseModel
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from duckduckgo_search import DDGS
import nltk

# 📌 Télécharger les ressources nécessaires pour NLTK
nltk.download("punkt")

# 📌 Charger le token Hugging Face depuis la variable d’environnement
HF_TOKEN = os.getenv("HF_TOKEN")

# 📌 URL de l'API Hugging Face pour ton modèle PsyBot
HF_MODEL_URL = "https://api-inference.huggingface.co/models/fatmata/psybot"

# 📌 Initialisation de FastAPI
app = FastAPI()

# 📌 Analyse des émotions avec VADER
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
    payload = {"inputs": prompt, "parameters": {
        "max_new_tokens": 100,
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.2
    }}

    response = requests.post(HF_MODEL_URL, headers=headers, json=payload)
    response_json = response.json()

    # 📌 Extraire la réponse correctement
    if isinstance(response_json, list) and len(response_json) > 0:
        generated_text = response_json[0]['generated_text']
        return generated_text.split("<|bot|>")[-1].strip()
    else:
        return "Désolé, je ne peux pas répondre pour le moment."

# 📌 Fonction de recherche avec DuckDuckGo
def search_duckduckgo(query, max_results=3):
    """Recherche des informations sur DuckDuckGo et retourne une liste de résultats."""
    search_results = list(DDGS().text(query, max_results=max_results))
    if search_results:
        return [result["body"] for result in search_results]
    return ["Je n'ai pas trouvé d'informations sur ce sujet."]

# 📌 Fonction de classification et réponse
def classify_and_respond(text):
    tokens = set(word_tokenize(text.lower()))

    # 🔹 Vérifier si c'est une recherche
    if tokens.intersection(search_keywords) or text.endswith('?'):
        return search_duckduckgo(text)
    
    # 🔹 Analyse du sentiment avec VADER
    vader_score = analyzer.polarity_scores(text)["compound"]

    # 🔹 Bloquer les messages violents
    violent_keywords = {"punch", "hit", "hurt", "kill", "destroy", "break", "explode", "attack"}
    if any(word in text.lower() for word in violent_keywords):
        return ["🔴 Non Accepté: Essayez de vous calmer. La violence ne résout rien."]

    # 🔹 Si la requête est acceptable, utiliser GPT
    response = generate_response(text)
    return [f"🟢 Accepté: {response}"]

# 📌 Endpoint principal de l'API
@app.post("/chat/")
async def chat_with_bot(user_input: UserInput):
    response = classify_and_respond(user_input.user_input)
    return {"response": response}
