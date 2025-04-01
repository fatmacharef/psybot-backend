from fastapi import FastAPI
import requests
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer
from duckduckgo_search import DDGS
from fastapi.middleware.cors import CORSMiddleware
from mtranslate import translate
from langdetect import detect  # 📌 Pour détecter la langue

# 📌 Charger le tokenizer de PsyBot
tokenizer = AutoTokenizer.from_pretrained("fatmata/psybot")

# 📌 Initialisation de FastAPI
app = FastAPI()

# ✅ Ajouter CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📌 Initialisation de l’analyseur VADER
analyzer = SentimentIntensityAnalyzer()

# 📌 Liste des mots-clés pour détecter une recherche
search_keywords = {
    "what", "who", "define", "explain", "is", "how", "causes", "symptoms",
    "treatment", "history", "types", "effects", "meaning", "scientific", "study", "research"
}

# 📌 Liste des mots-clés violents
violent_keywords = {"punch", "hit", "hurt", "kill", "destroy", "break", "explode", "attack"}

# 📌 Modèle pour recevoir l’entrée utilisateur
class UserInput(BaseModel):
    user_input: str

# 📌 Détecter la langue du message
def detect_language(text):
    try:
        detected_lang = detect(text)
        if detected_lang not in ["fr", "en", "ar"]:  # 📌 Si ce n'est pas une langue valide, on force l'anglais
            return "en"
        return detected_lang
    except:
        return "en"  # 📌 Si on ne détecte pas, on retourne l'anglais par défaut

# 📌 Fonction de recherche DuckDuckGo
def search_duckduckgo(query, max_results=3):
    try:
        search_results = list(DDGS().text(query, max_results=max_results))
        return [result["body"] for result in search_results if "body" in result] or ["Je n'ai pas trouvé d'informations sur ce sujet."]
    except Exception as e:
        return [f"Erreur de recherche : {str(e)}"]

# 📌 Fonction de génération avec GPT
def generate_response(user_input):
    HF_SPACE_URL = "https://fatmata-psybot-api.hf.space/generate"

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
        response = requests.post(HF_SPACE_URL, json=payload, headers=headers, timeout=30)

        if response.status_code != 200:
            return f"🚨 Erreur {response.status_code} : Impossible d'obtenir une réponse."

        response_json = response.json()
        return response_json.get("response", "Désolé, je ne peux pas répondre pour le moment.")

    except requests.exceptions.Timeout:
        return "🛑 Erreur : Temps de réponse trop long. Réessaie plus tard."
    except requests.exceptions.RequestException as e:
        return f"🛑 Erreur de connexion à Hugging Face : {str(e)}"

# 📌 Fonction de classification et réponse
def classify_and_respond(text, original_lang):
    print(f"🔍 Message reçu (traduit en anglais) : {text}")

    try:
        tokens = set(tokenizer.tokenize(text.lower()))
        print(f"✅ Tokens : {tokens}")

        if tokens.intersection(search_keywords) or text.endswith('?'):
            response = search_duckduckgo(text)
        elif any(word in text.lower().split() for word in violent_keywords):
            response = ["🔴 Non Accepté: Essayez de vous calmer. La violence ne résout rien."]
        else:
            vader_score = analyzer.polarity_scores(text)["compound"]
            print(f"🧠 Score VADER : {vader_score}")

            response = [generate_response(text)]
            print(f"🤖 Réponse GPT : {response}")

        # Traduire la réponse dans la langue originale détectée
        translated_response = [translate(r, original_lang) for r in response]
        return translated_response

    except Exception as e:
        print(f"❌ Erreur classification : {e}")
        return ["⚠️ Une erreur est survenue dans la classification du message."]

# 📌 Endpoint API
@app.post("/chat/")
async def chat_with_bot(user_input: UserInput):
    print(f"📥 Requête reçue du frontend: {user_input.user_input}")

    # 📌 Détection de la langue originale
    detected_lang = detect_language(user_input.user_input)
    print(f"🌍 Langue détectée : {detected_lang}")

    # 📌 Traduire l'entrée utilisateur en anglais avant tout traitement
    user_input_en = translate(user_input.user_input, "en")
    
    # 📌 Effectuer la classification et la génération de réponse
    response = classify_and_respond(user_input_en, detected_lang)

    return {"response": response}

@app.get("/")
async def home():
    return {"message": "PsyBot API is running!"}
