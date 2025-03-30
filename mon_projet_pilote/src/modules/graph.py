"""
Construction et compilation du graphe d'agent.
"""
from langgraph.graph import StateGraph, END
from typing import Dict, Any, Optional, Callable, Annotated
import time

from .state import AgentState
from .reasoning import (
    analyser, 
    choisir_outil, 
    appeler_météo, 
    appeler_calculatrice, 
    réponse_directe, 
    formuler_réponse, 
    router
)
from .errors import logger, GraphExecutionError, safe_execute

def nœud_de_récupération(state: Dict[str, Any]) -> Dict[str, Any]:
    """Nœud de récupération en cas d'erreur dans le graphe.
    
    Args:
        state: État actuel avec erreur
        
    Returns:
        État mis à jour avec une réponse d'erreur
    """
    logger.warning(f"Activation du nœud de récupération. Erreur: {state.get('error_message', 'inconnue')}")
    # Ajout d'un champ error_handled pour éviter les conflits d'état
    return {
        "error_handled": True,
        "recovery_message": (
            "Je suis désolé, une erreur s'est produite lors du traitement de votre demande. "
            "Veuillez réessayer ou reformuler votre question."
        )
    }

def build_agent_graph(max_retries: int = 3) -> Any:
    """Construit et compile le graphe d'agent avec gestion des erreurs.
    
    Args:
        max_retries: Nombre maximum de tentatives de compilation
        
    Returns:
        Graphe compilé
        
    Raises:
        GraphExecutionError: Si la compilation échoue après le nombre maximum de tentatives
    """
    logger.info("Construction du graphe d'agent")
    
    # Tentatives de compilation avec backoff exponentiel
    for attempt in range(max_retries):
        try:
            # Utilisation d'Annotated pour les champs qui peuvent être mis à jour par plusieurs nœuds
            workflow = StateGraph(AgentState)
            
            # Ajout des nœuds principaux
            workflow.add_node("analyser", analyser)
            workflow.add_node("choisir_outil", choisir_outil)
            workflow.add_node("appeler_météo", appeler_météo)
            workflow.add_node("appeler_calculatrice", appeler_calculatrice)
            workflow.add_node("réponse_directe", réponse_directe)
            workflow.add_node("formuler_réponse", formuler_réponse)
            
            # Ajout du nœud de récupération
            workflow.add_node("récupération", nœud_de_récupération)
            
            # Définition des arêtes avec routage dynamique
            workflow.set_entry_point("analyser")
            
            # Arêtes standards
            workflow.add_edge("analyser", "choisir_outil")
            workflow.add_conditional_edges("choisir_outil", router)
            workflow.add_edge("appeler_météo", "formuler_réponse")
            workflow.add_edge("appeler_calculatrice", "formuler_réponse")
            workflow.add_edge("réponse_directe", END)
            workflow.add_edge("formuler_réponse", END)
            
            # Arêtes de récupération d'erreur
            def détecteur_erreur(state: Dict[str, Any]) -> str:
                """Détecte si une erreur s'est produite et route vers le nœud approprié."""
                if state.get("error", False):
                    logger.warning(f"Erreur détectée, routage vers récupération: {state.get('error_message', '')}")
                    return "récupération"
                return "normal"
            
            # Ajout de chemins de récupération simplifiés
            workflow.add_edge("récupération", END)
            
            # Compilation avec suivi des performances
            start_time = time.time()
            compiled_graph = workflow.compile()
            compilation_time = time.time() - start_time
            
            logger.info(f"Graphe compilé avec succès en {compilation_time:.2f} secondes")
            return compiled_graph
            
        except Exception as e:
            logger.error(f"Erreur lors de la compilation du graphe (tentative {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                # Backoff exponentiel
                wait_time = 2 ** attempt
                logger.info(f"Nouvelle tentative dans {wait_time} secondes...")
                time.sleep(wait_time)
            else:
                logger.critical(f"Échec de la compilation du graphe après {max_retries} tentatives")
                raise GraphExecutionError(f"Impossible de compiler le graphe: {str(e)}")
    
    # Cette ligne ne devrait jamais être atteinte
    raise GraphExecutionError("Erreur inattendue lors de la compilation du graphe") 