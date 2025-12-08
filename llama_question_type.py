import requests
import json

OLLAMA_URL = "http://127.0.0.1:9999/api/generate"
MODEL_NAME = "llama3.3:70b-instruct-q4_0"

def ask_llama(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    data = response.json()
    return data.get("response", "")


if __name__ == "__main__":
    print("Hello, does LLaMA work?")
    answer = ask_llama("Explain quantum mechanics in simple terms.")
    print("\n🧠 LLaMA Response:\n")
    print(answer)
