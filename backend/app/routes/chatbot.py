import os

from fastapi import APIRouter
from dotenv import load_dotenv
from openai import OpenAI
import logging

load_dotenv(dotenv_path=".env")

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize client lazily
client = None

def get_client():
    """Get or initialize OpenAI client"""
    global client
    if client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            logger.warning("OPENROUTER_API_KEY not set. Chatbot endpoint will not work.")
            return None
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    return client


@router.post("/chat")
def chat(question: str):

    try:
        openai_client = get_client()
        if openai_client is None:
            return {
                "response": "Chatbot is not configured. Please set OPENROUTER_API_KEY environment variable.",
                "error": "API key not configured"
            }
        
        completion = openai_client.chat.completions.create(
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