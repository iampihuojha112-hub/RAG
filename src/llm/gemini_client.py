# Gemini API client.
# Filled in: Module 7
from google import genai
from google.genai import errors, types

from src.config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE


class GeminiClient:
    """Sends prompts to Gemini and returns the text answer."""

    def __init__(
        self,
        api_key: str | None = GEMINI_API_KEY,
        model: str = GEMINI_MODEL,
        temperature: float = GEMINI_TEMPERATURE,
    ):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing. Put it in the .env file.")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=self.temperature,
                ),
            )
        except errors.APIError as error:
            raise RuntimeError(
                f"Gemini request failed ({error.code}): {error.message}"
            ) from error
        return response.text or ""