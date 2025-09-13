from fastapi import FastAPI
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from duckduckgo_search import DDGS
from fastapi.middleware.cors import CORSMiddleware
from mtranslate import translate
from langdetect import detect
import torch
import torch.nn.functional as F
import re

# 📌 Fonction de nettoyage de réponse GPT améliorée
import re

def clean_response(text):
    # 1. Supprime les balises HTML et XML
    text = re.sub(r'<[^>]+>', '', text)

    # 2. Coupe à la première fermeture suspecte
    text = re.split(r'</(Bot|name|opinion|User|[a-zA-Z]*)>', text)[0]

    # 3. Supprime les préfixes inutiles
    text = re.sub(r'^\s*[,.:;-]*', '', text)
    text = re.sub(r'^\s*(Psyche|Therapist|Bot|Assistant|Psyidically|AI):?\s*', '', text)

    # 4. Supprime les parenthèses et crochets
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    # 5. Supprime les smileys simples comme :) :( ;-) etc.
    text = re.sub(r'[:;=8][-~]?[)D(\\/*|]', '', text)
    # 6. Supprime les phrases toutes faites ou inacceptables
    forbidden_starts = [
        r"I'?m a great listener",
        r"What makes me happy is",
        r"I'm just an AI",
        r"As a language model",
        r"I'm sorry you feel that way",
        r"Thank you for your question",
    ]
    for phrase in forbidden_starts:
        text = re.sub(phrase, '', text, flags=re.IGNORECASE)

    # 6. Supprime les fins incohérentes (`, </`, `:`, etc.)
    text = re.sub(r'[,:;]?\s*(</|<[^>]*|[:\-]|\.\.\.)\s*$', '', text.strip())

    # 7. Réduction des espaces
    text = re.sub(r'\s{2,}', ' ', text).strip()

    # 8. Garde les 2 premières phrases seulement
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return " ".join(sentences[:2]).strip()


# 📌 Charger le modèle et tokenizer GPT localement
MODEL_PATH = "fatmata/gpt-psybot"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

# 📌 Charger le modèle BERT-GoEmotions
BERT_MODEL_NAME = "fatmata/bert_model"
bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_NAME)
#charger le modele BERT-première-classification
save_path = "fatmata/mini_bert"
model_c = AutoModelForSequenceClassification.from_pretrained(save_path)
tokenizer_c = AutoTokenizer.from_pretrained(save_path)

# 📌 Initialisation de FastAPI
app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # Autorise toutes les origines
    
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📌 Initialisation VADER
analyzer = SentimentIntensityAnalyzer()

# 📌 Mots-clés
search_keywords = {
    "what", "who", "define", "explain", "is", "how", "causes", "symptoms",
    "treatment", "history", "types", "effects", "meaning", "scientific", "study", "research"
}
violent_keywords = {"punch", "hit", "hurt", "kill", "destroy", "break", "explode", "attack"}
UNACCEPTABLE_EMOTIONS = {"anger"}
GOEMOTIONS_LABELS = [
    "admiration", "anger", "approval", "autre", "curiosity", "disapproval", "gratitude", "joy", "love", "neutral",
    "sadness",
]

# 📌 Détection de langue
def detect_language(text):
    try:
        detected_lang = detect(text)
        return detected_lang if detected_lang in ["fr", "en", "ar"] else "en"
    except:
        return "en"

# 📌 Recherche externe si question
def search_duckduckgo(query, max_results=3):
    try:
        search_results = list(DDGS().text(query, max_results=max_results))
        return [result["body"] for result in search_results if "body" in result] or ["Je n'ai pas trouvé d'informations sur ce sujet."]
    except Exception as e:
        return [f"Erreur de recherche : {str(e)}"]

# 📌 Génération avec prompt simplifié
def generate_response(user_input):
    prompt = f"User: {user_input}\nBot:"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=150,  # Réduire la taille pour accélérer la génération
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    if "Bot:" in generated_text:
        raw_response = generated_text.split("Bot:")[-1].strip()
    else:
        raw_response = generated_text

    return clean_response(raw_response)

# 📌 Classification avec BERT + VADER

def classify_emotion(text):
    # Analyse de sentiment avec VADER
    sentiment_scores = analyzer.polarity_scores(text)
    compound = sentiment_scores['compound'] * 100

    # Tokenisation BERT
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        logits = bert_model(**inputs).logits

    # Calcul des probabilités avec softmax
    probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()

    # Sélection de l'émotion avec la probabilité la plus haute
    top_emotion_index = probs.argmax()
    top_emotion = GOEMOTIONS_LABELS[top_emotion_index]

    # Vérifie si l’émotion détectée est dans la liste des émotions inacceptables
    is_unacceptable = top_emotion in UNACCEPTABLE_EMOTIONS

    return compound, is_unacceptable, top_emotion


# Fonction pour prédire la classe de la requête
def predict(text):
    # Tokeniser la requête
    inputs = tokenizer_c(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Faire une prédiction avec le modèle
    with torch.no_grad():
        outputs = model_c(**inputs)
    
    # Extraire les logits de la sortie
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    
    # Interpréter la prédiction
    if predictions.item() == 1:
        return "recherche"  # Si la prédiction est 1, il s'agit d'une question
    else:
        return "GPT"  # Sinon, ce n'est pas une question




# 📌 Classification initiale
def classify_and_respond(text, original_lang):
    steps = []
    steps.append(" Étape 1: Requête reçue")

    category = predict(text)
    steps.append(f" Étape 2: Classification initiale -> {category}")

    if category == "recherche":
        response = search_duckduckgo(text)
        steps.append(" Étape 3: Recherche externe via DuckDuckGo")
        translated_response = [translate(r, original_lang) for r in response]
        steps.append(" Étape 4: Traduction des résultats")
        return translated_response, "recherche", [], steps

    compound, is_unacceptable, emotions = classify_emotion(text)
    steps.append(f" Étape 3: Analyse des émotions -> {emotions}, score VADER: {compound:.2f}")

    if is_unacceptable and abs(compound) > 50:
        alert = " Je ressens beaucoup de tension dans votre message. Essayez de vous calmer un peu."
        translated_alert = translate(alert, original_lang)
        steps.append(" Étape 4: Emotion inacceptable détectée -> Alerte générée")
        return [translated_alert], "non acceptable", emotions, steps

    gpt_response = generate_response(text)
    steps.append(" Étape 4: Réponse générée avec GPT")

    translated_gpt = translate(gpt_response, original_lang)
    return [translated_gpt], "gpt", emotions, steps

# 📌 API models
class UserInput(BaseModel):
    user_input: str

# 📌 Route principale
@app.post("/chat/")
async def chat_with_bot(user_input: UserInput):
    detected_lang = detect_language(user_input.user_input)
    user_input_en = translate(user_input.user_input, "en")

    response, response_type, emotions, steps = classify_and_respond(user_input_en, detected_lang)

    return {
        "response": response,
        "response_type": response_type,
        "emotions": emotions,
        "steps": steps
    }


# 📌 Route test
@app.get("/")
async def home():
    return {"message": "PsyBot API is running locally!"}
