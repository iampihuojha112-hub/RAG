
from groq import Groq

from src.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_TEMPERATURE,
)


class GroqClient:
    """Sends prompts to Groq and returns the generated answer."""

    def __init__(
        self,
        api_key: str | None = GROQ_API_KEY,
        model: str = GROQ_MODEL,
        temperature: float = GROQ_TEMPERATURE,
    ):
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. Put it in the .env file."
            )

        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None
    ) -> str:

        messages = []

        if system_instruction:
            messages.append({
                "role": "system",
                "content": system_instruction,
            })

        messages.append({
            "role": "user",
            "content": prompt,
        })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )

        return response.choices[0].message.content or ""
