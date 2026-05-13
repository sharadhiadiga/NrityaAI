import os

from fastapi import APIRouter
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=".env")

router = APIRouter()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


@router.post("/chat")
def chat(question: str):

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content":
                    """
                    You are NrityaAI, an expert Bharatanatyam assistant.

                    Answer ONLY Bharatanatyam-related questions.

                    If unrelated, say:
                    'I can answer only Bharatanatyam-related questions.'
                    """
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        )

        answer = completion.choices[0].message.content

        return {
            "response": answer
        }

    except Exception as e:
        print("ERROR:", repr(e))

        return {
            "error": str(e)
        }