import pytest
import requests

def test_network_connectivity():
    """
    Teste la connectivité réseau en vérifiant si l'endpoint du LLM est accessible.
    """
    # Remplacez par l'URL réelle de l'API de Gemini 2.0 Flash
    llm_endpoint ="https://google.com"
    
    try:
        # Effectuer une requête GET
        response = requests.get(llm_endpoint, timeout=5)
        # Vérifiez si le statut HTTP est 200
        assert response.status_code == 200, f"Connexion échouée : {response.status_code}"
    except requests.exceptions.RequestException as e:
        # Fait échouer le test en cas d'exception réseau
        pytest.fail(f"Erreur de connectivité réseau : {e}")
