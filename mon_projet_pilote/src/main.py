"""
Point d'entrée principal pour l'agent LangGraph.
"""
import sys
import time
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from modules import (
    AgentState,
    build_agent_graph,
    print_graph_structure,
    visualize_graph
)
from modules.reasoning import (
    analyser,
    choisir_outil,
    appeler_météo,
    appeler_calculatrice,
    réponse_directe,
    formuler_réponse,
    router
)
from modules.errors import logger, GraphExecutionError, safe_execute

# Charger les variables d'environnement
load_dotenv()

def main():
    """Point d'entrée principal du programme."""
    try:
        logger.info("Démarrage de l'agent")
        
        # Construire un graph simple pour le test
        workflow = StateGraph(AgentState)
        workflow.add_node("analyser", analyser)
        workflow.add_node("choisir_outil", choisir_outil)
        workflow.add_node("appeler_météo", appeler_météo)
        workflow.add_node("appeler_calculatrice", appeler_calculatrice)
        workflow.add_node("réponse_directe", réponse_directe)
        workflow.add_node("formuler_réponse", formuler_réponse)
        
        workflow.set_entry_point("analyser")
        workflow.add_edge("analyser", "choisir_outil")
        workflow.add_conditional_edges("choisir_outil", router)
        workflow.add_edge("appeler_météo", "formuler_réponse")
        workflow.add_edge("appeler_calculatrice", "formuler_réponse")
        workflow.add_edge("réponse_directe", END)
        workflow.add_edge("formuler_réponse", END)
        
        app = workflow.compile()
        
        # Test simple
        question = "Quelle est la météo à Paris?"
        print(f"Test avec la question: {question}")
        
        start_time = time.time()
        result = app.invoke({"question": question})
        execution_time = time.time() - start_time
        
        print(f"Exécution en {execution_time:.2f} secondes")
        print(f"Réponse: {result.get('answer', 'Pas de réponse')}")
        
        return 0
    except Exception as e:
        logger.error(f"Erreur: {str(e)}")
        print(f"Une erreur s'est produite: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 