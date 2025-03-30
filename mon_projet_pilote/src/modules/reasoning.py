"""
Fonctions de raisonnement pour l'agent (nœuds du graphe).
"""
from typing import Literal, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import time

from .state import AgentState
from .tools import recherche_météo, calculatrice
from .errors import (
    handle_state_errors, 
    logger, 
    LLMResponseError, 
    ToolExecutionError,
    safe_execute
)

def get_llm(retries=2, backoff=1.5):
    """Obtient une instance LLM avec gestion des erreurs et retry.
    
    Args:
        retries: Nombre de tentatives en cas d'erreur
        backoff: Facteur de multiplication pour le temps d'attente entre les tentatives
        
    Returns:
        Instance du modèle LLM
    """
    attempt = 0
    last_error = None
    
    while attempt <= retries:
        try:
            return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
        except Exception as e:
            last_error = e
            attempt += 1
            if attempt <= retries:
                logger.warning(f"Erreur lors de l'initialisation du LLM (tentative {attempt}): {str(e)}")
                time.sleep(backoff ** attempt)  # Attente exponentielle
            else:
                logger.error(f"Échec de l'initialisation du LLM après {retries} tentatives: {str(e)}")
                raise
    
    # Ne devrait jamais arriver, mais au cas où
    raise last_error

@handle_state_errors
def analyser(state: AgentState) -> Dict[str, Any]:
    """Analyse la question initiale et génère des réflexions."""
    logger.info(f"Analyse de la question: {state.get('question', '')}")
    
    if not state.get("question"):
        logger.warning("Tentative d'analyse sans question fournie")
        return {"thoughts": "Je n'ai pas reçu de question à analyser."}
    
    try:
        llm = get_llm()
        prompt = ChatPromptTemplate.from_template(
            "Question: {question}\nRéfléchissez au problème."
        )
        chain = prompt | llm
        thoughts = chain.invoke({"question": state["question"]})
        logger.info("Analyse réussie")
        return {"thoughts": thoughts.content}
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse: {str(e)}")
        return {
            "thoughts": "Je rencontre des difficultés à analyser cette question.",
            "error": True
        }

@handle_state_errors
def choisir_outil(state: AgentState) -> Dict[str, Any]:
    """Choisit l'outil approprié et prépare l'entrée pour cet outil."""
    logger.info("Choix de l'outil approprié")
    
    if not state.get("thoughts"):
        logger.warning("Tentative de choix d'outil sans pensées préalables")
        return {
            "tool_name": "réponse_directe",
            "tool_input": "",
            "error": True
        }
    
    try:
        # Liste des outils disponibles
        outils = ["recherche_météo", "calculatrice", "réponse_directe"]
        
        # Obtention du LLM avec retry
        llm = get_llm()
        
        # Choix de l'outil
        prompt = ChatPromptTemplate.from_template(
            "Question: {question}\nRéflexion: {thoughts}\n"
            "Choisissez l'outil le plus approprié parmi: {outils}. "
            "Répondez uniquement avec le nom de l'outil."
        )
        chain = prompt | llm
        
        # Appel sécurisé au LLM
        tool_name_response = safe_execute(
            chain.invoke,
            {"content": "réponse_directe"},  # Fallback en cas d'erreur
            {
                "question": state["question"], 
                "thoughts": state["thoughts"],
                "outils": ", ".join(outils)
            }
        )
        
        tool_name = tool_name_response.content
        
        # Validation du nom d'outil
        if tool_name not in outils:
            logger.warning(f"Nom d'outil invalide: {tool_name}")
            tool_name = "réponse_directe"
        
        logger.info(f"Outil choisi: {tool_name}")
        
        # Préparer l'entrée de l'outil
        tool_input = ""
        if tool_name == "recherche_météo":
            prompt_input = ChatPromptTemplate.from_template(
                "Extrayez le nom de la ville de la question: {question}"
            )
            tool_input_response = safe_execute(
                (prompt_input | llm).invoke,
                {"content": ""},
                {"question": state["question"]}
            )
            tool_input = tool_input_response.content
            
        elif tool_name == "calculatrice":
            prompt_input = ChatPromptTemplate.from_template(
                "Extrayez l'expression mathématique de la question: {question}. "
                "Ne retournez que l'expression mathématique, sans texte supplémentaire."
            )
            tool_input_response = safe_execute(
                (prompt_input | llm).invoke,
                {"content": ""},
                {"question": state["question"]}
            )
            tool_input = tool_input_response.content
        
        logger.info(f"Entrée de l'outil: {tool_input}")
        return {"tool_name": tool_name, "tool_input": tool_input}
        
    except Exception as e:
        logger.error(f"Erreur lors du choix d'outil: {str(e)}")
        # En cas d'erreur, on utilise la réponse directe comme fallback
        return {
            "tool_name": "réponse_directe", 
            "tool_input": "", 
            "error": True
        }

@handle_state_errors
def appeler_météo(state: AgentState) -> Dict[str, Any]:
    """Appelle l'outil de météo avec l'entrée préparée."""
    logger.info(f"Appel de l'outil météo avec: {state.get('tool_input', '')}")
    
    if not state.get("tool_input"):
        logger.warning("Tentative d'appel à l'outil météo sans entrée")
        return {
            "observation": "Je n'ai pas pu déterminer la ville pour laquelle vous souhaitez la météo.",
            "error": True
        }
        
    try:
        # Appeler la fonction recherche_météo directement
        observation = recherche_météo(state["tool_input"])
        logger.info(f"Résultat météo obtenu: {observation}")
        return {"observation": observation}
    except ToolExecutionError as e:
        logger.error(f"Erreur d'exécution de l'outil météo: {str(e)}")
        return {
            "observation": str(e),
            "error": True
        }
    except Exception as e:
        logger.error(f"Erreur inattendue lors de l'appel météo: {str(e)}")
        return {
            "observation": "Une erreur s'est produite lors de la recherche météo.",
            "error": True
        }

@handle_state_errors
def appeler_calculatrice(state: AgentState) -> Dict[str, Any]:
    """Appelle la calculatrice avec l'entrée préparée."""
    logger.info(f"Appel de l'outil calculatrice avec: {state.get('tool_input', '')}")
    
    if not state.get("tool_input"):
        logger.warning("Tentative d'appel à la calculatrice sans entrée")
        return {
            "observation": "Je n'ai pas pu déterminer l'expression mathématique à calculer.",
            "error": True
        }
        
    try:
        # Appeler la fonction calculatrice directement
        observation = calculatrice(state["tool_input"])
        logger.info(f"Résultat calculatrice obtenu: {observation}")
        return {"observation": observation}
    except ToolExecutionError as e:
        logger.error(f"Erreur d'exécution de la calculatrice: {str(e)}")
        return {
            "observation": str(e),
            "error": True
        }
    except Exception as e:
        logger.error(f"Erreur inattendue lors de l'appel calculatrice: {str(e)}")
        return {
            "observation": "Une erreur s'est produite lors du calcul.",
            "error": True
        }

@handle_state_errors
def réponse_directe(state: AgentState) -> Dict[str, Any]:
    """Génère une réponse directe sans utiliser d'outils."""
    logger.info("Génération d'une réponse directe")
    
    try:
        llm = get_llm()
        prompt = ChatPromptTemplate.from_template(
            "Question: {question}\nRéflexion: {thoughts}\n"
            "Donnez une réponse directe et utile."
        )
        
        response = safe_execute(
            (prompt | llm).invoke,
            {"content": "Je ne peux pas répondre à cette question pour le moment."},
            state
        )
        
        answer = response.content
        logger.info("Réponse directe générée avec succès")
        return {"observation": "Réponse directe", "answer": answer}
    except Exception as e:
        logger.error(f"Erreur lors de la génération de réponse directe: {str(e)}")
        return {
            "observation": "Réponse directe",
            "answer": "Je suis désolé, mais je ne peux pas générer une réponse à cette question pour le moment.",
            "error": True
        }

@handle_state_errors
def formuler_réponse(state: AgentState) -> Dict[str, Any]:
    """Formule une réponse finale basée sur l'observation de l'outil."""
    logger.info("Formulation de la réponse finale")
    
    if not state.get("observation"):
        logger.warning("Tentative de formulation de réponse sans observation")
        return {
            "answer": "Je n'ai pas pu générer une réponse car je n'ai pas d'observation à interpréter.",
            "error": True
        }
    
    try:
        llm = get_llm()
        prompt = ChatPromptTemplate.from_template(
            "Question: {question}\nRéflexion: {thoughts}\n"
            "Observation: {observation}\n"
            "Formulez une réponse complète, claire et utile."
        )
        
        response = safe_execute(
            (prompt | llm).invoke,
            {"content": state.get("observation", "Je ne peux pas formuler de réponse pour le moment.")},
            state
        )
        
        answer = response.content
        logger.info("Réponse finale formulée avec succès")
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Erreur lors de la formulation de la réponse: {str(e)}")
        return {
            "answer": f"Voici ce que j'ai trouvé: {state.get('observation', 'Aucune information disponible')}",
            "error": True
        }

def router(state: AgentState) -> Literal["appeler_météo", "appeler_calculatrice", "réponse_directe"]:
    """Détermine quel nœud appeler en fonction de l'outil choisi."""
    logger.info(f"Routage basé sur l'outil: {state.get('tool_name', 'non défini')}")
    
    # Vérifier s'il y a eu une erreur
    if state.get("error"):
        logger.warning("Routage vers réponse_directe en raison d'une erreur précédente")
        return "réponse_directe"
    
    if state["tool_name"] == "recherche_météo":
        return "appeler_météo"
    elif state["tool_name"] == "calculatrice":
        return "appeler_calculatrice"
    else:
        return "réponse_directe" 