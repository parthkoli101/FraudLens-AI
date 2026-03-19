import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('HF_API_KEY')

models_to_test = [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "HuggingFaceH4/zephyr-7b-beta",
    "meta-llama/Meta-Llama-3-8B-Instruct"
]

prompt = "Generate a JSON object with keys a, b mapping to strings 1, 2. Return ONLY valid JSON, no markdown."

for model in models_to_test:
    print(f"\n--- Testing {model} ---")
    try:
        res = requests.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "inputs": prompt,
                "parameters": {"max_new_tokens": 50, "return_full_text": False}
            },
            timeout=10
        )
        print("Status:", res.status_code)
        try:
            print("Response:", res.json())
        except:
            print("Raw text:", res.text)
    except Exception as e:
        print("Error:", e)
