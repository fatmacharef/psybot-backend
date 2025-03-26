from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from duckduckgo_search import DDGS
import os
from huggingface_hub import login

# 📌 Charger le token Hugging Face depuis la variable d’environnement
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    login(HF_TOKEN)

# 📌 Charger le modèle GPT et le tokenizer
model_name = "fatmata/psybot"  # Remplace par ton modèle
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=HF_TOKEN)

# 📌 Charger le modèle BERT pour l’analyse des émotions
emotion_classifier = pipeline("text-classification", model="monologg/bert-base-cased-goemotions-original")
analyzer = SentimentIntensityAnalyzer()

# 📌 Définition de l’API FastAPI
app = FastAPI()

# 📌 Modèle pour recevoir l'entrée utilisateur
class UserInput(BaseModel):
    user_input: str

# 📌 Fonction pour générer une réponse avec GPT
def generate_response(user_input):
    prompt = f"<|startoftext|><|user|> {user_input} <|bot|>"
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=100,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.split("<|bot|>")[-1].strip()

# 📌 Endpoint principal de l'API
@app.post("/chat/")
async def chat_with_bot(user_input: UserInput):
    response = generate_response(user_input.user_input)
    return {"response": response}

