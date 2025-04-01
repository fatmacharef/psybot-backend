from fastapi import FastAPI, WebSocket
import requests
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer
from duckduckgo_search import DDGS
from fastapi.middleware.cors import CORSMiddleware
from mtranslate import translate
from langdetect import detect

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
search_keywords = {"what", "who", "define", "explain", "is", "how", "causes", "symptoms", "treatment"}

# 📌 Liste des mots-clés violents
violent_keywords = {"punch", "hit", "hurt", "kill", "destroy"}

# 📌 Modèle pour recevoir l’entrée utilisateur
class UserInput(BaseModel):
    user_input: str

# 📌 Détecter la langue du message
def detect_language(text):
    try:
        lang = detect(text)
        print(f"🌍 Langue détectée : {lang}")
        return lang
    except:
        return "en"  # 📌 Si erreur, on assume anglais

# 📌 Fonction de recherche DuckDuckGo
def search_duckduckgo(query, max_results=3):
    try:
        search_results = list(DDGS().text(query, max_results=max_results))
        return [result["body"] for result in search_results if "body" in result] or ["Je n'ai pas trouvé d'informations."]
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
        response_json = response.json()
        return response_json.get("response", "Désolé, je ne peux pas répondre pour le moment.")
    except:
        return "🛑 Erreur de connexion à Hugging Face."

# 📌 Fonction principale de traitement
def classify_and_respond(text, original_lang):
    tokens = set(tokenizer.tokenize(text.lower()))
    
    if tokens.intersection(search_keywords) or text.endswith('?'):
        response = search_duckduckgo(text)
    elif any(word in text.lower().split() for word in violent_keywords):
        response = ["🔴 Message inapproprié détecté."]
    else:
        response = [generate_response(text)]
    
    # 📌 Traduire la réponse vers la langue originale
    translated_response = [translate(r, original_lang) for r in response]
    return translated_response

# 📌 Endpoint API classique
@app.post("/chat/")
async def chat_with_bot(user_input: UserInput):
    detected_lang = detect_language(user_input.user_input)
    user_input_en = translate(user_input.user_input, "en")
    response = classify_and_respond(user_input_en, detected_lang)
    return {"response": response}

# 📌 WebSocket pour communication en temps réel
@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            user_message = await websocket.receive_text()
            print(f"📩 Message reçu via WebSocket : {user_message}")
            
            detected_lang = detect_language(user_message)
            user_message_en = translate(user_message, "en")
            response = classify_and_respond(user_message_en, detected_lang)
            
            await websocket.send_text(response[0])
    except Exception as e:
        print(f"❌ Erreur WebSocket : {str(e)}")
        await websocket.close()

@app.get("/")
async def home():
    return {"message": "PsyBot API with WebSocket is running!"}
