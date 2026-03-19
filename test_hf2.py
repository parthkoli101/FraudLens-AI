import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('HF_API_KEY')

models_to_test = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
]

prompt = "Generate a JSON object with keys 'investigation_outcome' and 'reasoning' mapping to strings. Return ONLY valid JSON, no markdown fences."

success_models = []

for model in models_to_test:
    try:
        res = requests.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "inputs": prompt,
                "parameters": {"max_new_tokens": 100, "return_full_text": False}
            },
            timeout=10
        )
        if res.status_code == 200:
            text = res.json()[0].get('generated_text', '').strip()
            if text.startswith('```'):
                text = text.split('```')[1]
                if text.startswith('json'):
                    text = text[4:]
            try:
                json.loads(text.strip())
                success_models.append((model, "Valid JSON"))
            except json.JSONDecodeError:
                success_models.append((model, "Invalid JSON (returned text)"))
        else:
            success_models.append((model, f"Failed: {res.status_code}"))
    except Exception as e:
        success_models.append((model, f"Error: {e}"))

print("RESULTS:")
for m, status in success_models:
    print(f"{m}: {status}")
