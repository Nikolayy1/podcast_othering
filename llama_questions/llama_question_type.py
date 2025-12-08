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
    return response.json().get("response", "")

def load_prompt_template(path="./prompts/prompt.txt"):
    with open(path, "r") as f:
        return f.read()

if __name__ == "__main__":
    template = load_prompt_template()

    question = "Why does education matter in society?"  # you can replace this later
    prompt = template.replace("{QUESTION}", question)

    print("Prompt sent to LLaMA:\n", prompt)

    label = ask_llama(prompt)
    print("\n🧠 LLaMA Classification:", label)
