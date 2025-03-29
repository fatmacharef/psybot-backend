from fastapi import FastAPI
import requests
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer
from duckduckgo_search import DDGS
from fastapi.middleware.cors import CORSMiddleware  # ✅ Ajout CORS

# 📌 Charger le tokenizer de PsyBot
tokenizer = AutoTokenizer.from_pretrained("fatmata/psybot")

# 📌 Initialisation de FastAPI
app = FastAPI()

# ✅ Ajouter CORS pour autoriser les requêtes du frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🚨 Mettre "*" pour tester, mais mieux de restreindre plus tard
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# 📌 Fonction pour générer une réponse avec l'API Hugging Face Spaces
def generate_response(user_input):
    HF_SPACE_URL = "https://fatmata-psybot-api.hf.space/generate"  # Vérifie bien cette URL

    prompt = f"<|startoftext|><|user|> {user_input} <|bot|>"
    payload = {
        "prompt": prompt,
        "max_new_tokens": 100,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.2
    }

    headers = {"Content-Type": "application/json"}

    try:
        print(f"🚀 Envoi de la requête à {HF_SPACE_URL}...")

        response = requests.post(HF_SPACE_URL, json=payload, headers=headers, timeout=30)

        print(f"📡 Statut HTTP: {response.status_code}")
        print(f"📡 Réponse brute de HF: {response.text}")

        if response.status_code != 200:
            try:
                error_detail = response.json().get("detail", "Impossible d'obtenir une réponse.")
            except Exception:
                error_detail = "Impossible d'obtenir une réponse."
            return f"🚨 Erreur {response.status_code} : {error_detail}"

        response_json = response.json()

        if isinstance(response_json, dict) and "response" in response_json:
            return response_json["response"]

        return "Désolé, je ne peux pas répondre pour le moment."

    except requests.exceptions.Timeout:
        return "🛑 Erreur : Temps de réponse trop long. Réessaie plus tard."
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

        if tokens.intersection(search_keywords) or text.endswith('?'):
            return search_duckduckgo(text)

        if any(word in text.lower().split() for word in violent_keywords):
            return ["🔴 Non Accepté: Essayez de vous calmer. La violence ne résout rien."]

        vader_score = analyzer.polarity_scores(text)["compound"]
        print(f"🧠 Score VADER : {vader_score}")

        response = generate_response(text)
        print(f"🤖 Réponse GPT : {response}")
        return [f"🟢 Accepté: {response}"]

    except Exception as e:
        print(f"❌ Erreur classification : {e}")
        return ["⚠️ Une erreur est survenue dans la classification du message."]

# 📌 Endpoint API
@app.post("/chat/")
async def chat_with_bot(user_input: UserInput):
    print(f"📥 Requête reçue du frontend: {user_input.user_input}")
    return {"response": classify_and_respond(user_input.user_input)}

@app.get("/")
async def home():
    return {"message": "PsyBot API is running!"}
