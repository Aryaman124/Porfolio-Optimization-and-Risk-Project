# src/app.py
import os
from dotenv import load_dotenv
from google import genai

def main():
    # load API key + model name
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")  # fallback if not in .env

    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY missing in .env")

    # create client
    client = genai.Client(api_key=api_key)

    # send a simple test prompt
    response = client.models.generate_content(
        model=model,
        contents="Hello Gemini, can you confirm you’re working?"
    )

    print(f"Using model: {model}")
    print("Gemini Response:", response.text.strip())

if __name__ == "__main__":
    main()


    # test run seeing if the gemini model is working now
    