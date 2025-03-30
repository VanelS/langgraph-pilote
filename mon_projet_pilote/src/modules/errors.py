"""
Module de gestion des erreurs pour l'agent LangGraph.
Définit les exceptions personnalisées et les gestionnaires d'erreurs.
"""
import logging
import traceback
from typing import Any, Dict, Callable, TypeVar, Optional

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agent_errors.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("agent")

# Type générique pour les fonctions décorées
F = TypeVar('F', bound=Callable[..., Any])

# Exceptions personnalisées
class AgentError(Exception):
    """Exception de base pour les erreurs de l'agent."""
    pass

class ToolExecutionError(AgentError):
    """Erreur lors de l'exécution d'un outil."""
    pass

class LLMResponseError(AgentError):
    """Erreur lors de la génération de réponse par le LLM."""
    pass

class GraphExecutionError(AgentError):
    """Erreur lors de l'exécution du graphe."""
    pass

class InputValidationError(AgentError):
    """Erreur de validation des entrées."""
    pass

# Décorateurs de gestion d'erreurs
def handle_tool_errors(fallback_response: str = "Une erreur s'est produite lors de l'exécution de l'outil."):
    """Décorateur pour gérer les erreurs dans les outils.
    
    Args:
        fallback_response: Réponse à renvoyer en cas d'erreur
        
    Returns:
        Fonction décorée avec gestion d'erreurs
    """
    def decorator(func: F) -> F:
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_id = logger.error(
                    f"Erreur dans l'outil {func.__name__}: {str(e)}\n{traceback.format_exc()}"
                )
                raise ToolExecutionError(
                    f"{fallback_response} (ID: {error_id})"
                ) from e
        return wrapper
    return decorator

def handle_state_errors(func: F) -> F:
    """Décorateur pour gérer les erreurs dans les fonctions d'état.
    
    Args:
        func: Fonction à décorer
        
    Returns:
        Fonction décorée avec gestion d'erreurs
    """
    def wrapper(state: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        try:
            return func(state, *args, **kwargs)
        except Exception as e:
            error_message = f"Erreur dans {func.__name__}: {str(e)}"
            logger.error(f"{error_message}\n{traceback.format_exc()}")
            
            # Mise à jour de l'état avec l'erreur
            return {
                "error": True,
                "error_message": error_message,
                "answer": f"Je suis désolé, j'ai rencontré une erreur: {str(e)}. Veuillez réessayer."
            }
    return wrapper

def validate_input(validation_func: Callable[[Any], bool], error_message: str = "Entrée invalide"):
    """Décorateur pour valider les entrées des fonctions.
    
    Args:
        validation_func: Fonction qui valide l'entrée
        error_message: Message d'erreur à afficher
        
    Returns:
        Fonction décorée avec validation d'entrée
    """
    def decorator(func: F) -> F:
        def wrapper(input_value: Any, *args, **kwargs) -> Any:
            if not validation_func(input_value):
                logger.warning(f"Validation d'entrée échouée pour {func.__name__}: {input_value}")
                raise InputValidationError(error_message)
            return func(input_value, *args, **kwargs)
        return wrapper
    return decorator

# Fonction de récupération pour la résilience
def safe_execute(func: Callable, fallback: Any, *args, **kwargs) -> Any:
    """Exécute une fonction de manière sécurisée avec fallback en cas d'erreur.
    
    Args:
        func: Fonction à exécuter
        fallback: Valeur à retourner en cas d'erreur
        args: Arguments positionnels pour la fonction
        kwargs: Arguments nommés pour la fonction
        
    Returns:
        Résultat de la fonction ou fallback en cas d'erreur
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Erreur dans safe_execute pour {func.__name__}: {str(e)}")
        return fallback 