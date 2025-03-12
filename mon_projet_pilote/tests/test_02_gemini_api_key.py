import pytest
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import PermissionDenied, InvalidArgument

from dotenv import load_dotenv
import os

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
@pytest.fixture
def llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=API_KEY
    )

def test_gemini_api_key_validity(llm):
    try:
        response = llm.invoke("Test")
        assert response.content is not None
    except (PermissionDenied, InvalidArgument) as e:
        pytest.fail(f"API key invalide ou problème d'accès : {e}")
