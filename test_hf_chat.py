import requests, os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('HF_API_KEY')

models = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "HuggingFaceH4/zephyr-7b-beta"
]

prompt = "Generate a JSON with keys a and b. No text, just JSON."

for model in models:
    try:
        res = requests.post(
            f"https://api-inference.huggingface.co/models/{model}/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 50,
                "response_format": {"type": "json_object"}
            }
        )
        print(f"{model}: Status {res.status_code}")
        if res.status_code == 200:
            print("Content:", res.json()['choices'][0]['message']['content'])
        else:
            print("Error:", res.text)
    except Exception as e:
        print(f"{model} Exception: {e}")
