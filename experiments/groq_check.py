
from src.llm.groq_client import GroqClient


def main():
    print("Connecting to Groq...")

    groq = GroqClient()

    question = "Explain Retrieval Augmented Generation (RAG) in simple words."

    print("\nSending question to Groq...")
    answer = groq.generate(question)

    print("\nGenerated Answer:")
    print(answer)


if __name__ == "__main__":
    main()
