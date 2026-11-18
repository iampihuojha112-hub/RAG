from google import genai

from src.config import GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)

for model in client.models.list():
    if "generateContent" in (model.supported_actions or []):
        print(model.name)