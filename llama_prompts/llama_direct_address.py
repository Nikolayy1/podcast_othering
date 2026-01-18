import json
import requests
import os

OLLAMA_URL = "http://127.0.0.1:9999/api/generate"
MODEL_NAME = "llama3.3:70b-instruct-q4_0"

# Load your prompt template
def load_prompt_template(path="prompts/direct_address_prompt.txt"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, path)
    with open(prompt_path, "r") as f:
        return f.read()

# Ask local LLaMA
def ask_llama(prompt: str) -> str:
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json().get("response", "").strip()

if __name__ == "__main__":
    print("Loading test data...")
    with open("test_set_direct_address.json", "r") as f:
        items = json.load(f)

    template = load_prompt_template()

    results = []

    print(f"Annotating {len(items)} examples...")
    for i, item in enumerate(items):
        text = item["text"]

        # fill template
        prompt = template.replace("{QUESTION}", text)

        # query LLaMA
        label = ask_llama(prompt)

        # store result
        results.append({"text": text, "label": label})

        if i % 25 == 0:
            print(f"Processed {i}/{len(items)}")

    print("Saving annotations...")
    with open("llm_annotations_direct.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("DONE. Saved to llm_annotations_direct.json")