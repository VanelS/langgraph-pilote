"""
Point d'entrée principal pour l'agent LangGraph.
"""
import argparse
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

def run_agent_with_question(app, question):
    """Exécute l'agent avec une question donnée.
    
    Args:
        app: Le graphe d'agent compilé
        question: La question à poser
        
    Returns:
        La réponse générée
    """
    try:
        logger.info(f"Traitement de la question: {question}")
        start_time = time.time()
        result = app.invoke({"question": question})
        execution_time = time.time() - start_time
        
        logger.info(f"Question traitée en {execution_time:.2f} secondes")
        if "error" in result:
            logger.warning(f"La réponse contient une erreur: {result.get('error_message', 'Non spécifiée')}")
            
        return result.get("answer", "Je n'ai pas pu générer de réponse.")
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution de l'agent: {str(e)}")
        return f"Une erreur s'est produite lors du traitement de votre question: {str(e)}"

def create_visualization():
    """Crée une visualisation du graphe d'agent.
    
    Returns:
        Le workflow créé pour la visualisation
    """
    workflow = StateGraph(AgentState)
    workflow.add_node("analyser", analyser)
    workflow.add_node("choisir_outil", choisir_outil)
    workflow.add_node("appeler_météo", appeler_météo)
    workflow.add_node("appeler_calculatrice", appeler_calculatrice)
    workflow.add_node("réponse_directe", réponse_directe)
    workflow.add_node("formuler_réponse", formuler_réponse)
    workflow.add_node("récupération", lambda state: {"answer": "Erreur récupérée"})
    
    workflow.set_entry_point("analyser")
    workflow.add_edge("analyser", "choisir_outil")
    workflow.add_conditional_edges("choisir_outil", router)
    workflow.add_edge("appeler_météo", "formuler_réponse")
    workflow.add_edge("appeler_calculatrice", "formuler_réponse")
    workflow.add_edge("réponse_directe", END)
    workflow.add_edge("formuler_réponse", END)
    workflow.add_edge("récupération", END)
    
    return workflow

def main():
    """Point d'entrée principal du programme."""
    parser = argparse.ArgumentParser(description="Agent LangGraph avec gestion d'erreurs")
    parser.add_argument("--question", type=str, help="Question à poser à l'agent")
    parser.add_argument("--visualize", action="store_true", help="Générer une visualisation du graphe")
    parser.add_argument("--debug", action="store_true", help="Activer le mode debug")
    
    args = parser.parse_args()
    
    # Construction du graphe
    try:
        logger.info("Démarrage de l'agent")
        app = build_agent_graph()
        
        # Exécution interactive ou avec une question fournie
        if args.question:
            answer = run_agent_with_question(app, args.question)
            print(f"Réponse: {answer}")
        else:
            # Tests par défaut
            print("Test de l'agent avec différentes questions...\n")
            
            questions = [
                "Quelle est la météo à Paris?",
                "Combien font 15 * 32 + 48?",
                "Quelles sont les capitales des pays d'Europe?",
                "Quelle est la météo à XYZVille?",  # Ville qui n'existe pas
                "Calcule 10/0"  # Erreur de division par zéro
            ]
            
            for question in questions:
                print(f"\nQuestion: {question}")
                answer = run_agent_with_question(app, question)
                print(f"Réponse: {answer}")
                time.sleep(1)  # Pause entre les questions
    
    except GraphExecutionError as e:
        logger.critical(f"Erreur critique lors de la construction du graphe: {str(e)}")
        print(f"Erreur: {str(e)}")
        return 1
    except Exception as e:
        logger.critical(f"Erreur non gérée: {str(e)}")
        print(f"Une erreur inattendue s'est produite: {str(e)}")
        return 1
    
    # Génération de la visualisation si demandée
    if args.visualize:
        try:
            workflow = create_visualization()
            print_graph_structure(workflow)
            visualize_graph(workflow)
        except Exception as e:
            logger.error(f"Erreur lors de la génération de la visualisation: {str(e)}")
            print(f"Impossible de générer la visualisation: {str(e)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 