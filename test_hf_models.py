import requests, os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('HF_API_KEY')

models = [
    "HuggingFaceH4/zephyr-7b-beta",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "Qwen/Qwen2.5-72B-Instruct"
]

for model in models:
    try:
        res = requests.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"inputs": "Test"}
        )
        print(f"{model}: {res.status_code}")
        if res.status_code != 200:
            print(f"Error: {res.text}")
    except Exception as e:
        print(f"{model} Exception: {e}")
