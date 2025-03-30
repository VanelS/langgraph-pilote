"""
Point d'entrée principal pour l'agent LangGraph.
"""
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

# Charger les variables d'environnement
load_dotenv()

def main():
    """Point d'entrée principal du programme."""
    # Construction du graphe
    app = build_agent_graph()
    
    # Test avec une question
    print("Test de l'agent avec une question sur la météo...")
    result = app.invoke({"question": "Quelle est la météo à Paris?"})
    print(f"Réponse: {result['answer']}")
    
    print("\nTest de l'agent avec une question mathématique...")
    result = app.invoke({"question": "Combien font 15 * 32 + 48?"})
    print(f"Réponse: {result['answer']}")
    
    # Créer le graphe pour visualisation
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
    
    # Afficher la représentation textuelle
    print_graph_structure(workflow)
    
    # Générer la visualisation graphique
    visualize_graph(workflow)

if __name__ == "__main__":
    main() 