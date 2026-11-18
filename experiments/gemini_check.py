from src.llm.gemini_client import GeminiClient

client = GeminiClient()

answer = client.generate(
    "Explain what a version control system is in 2 simple sentences.",
    system_instruction="You are a friendly teacher. Use simple words.",
)
print(answer)