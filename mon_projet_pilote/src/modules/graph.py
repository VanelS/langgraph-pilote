"""
Construction et compilation du graphe d'agent.
"""
from langgraph.graph import StateGraph, END
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

def build_agent_graph():
    """Construit et compile le graphe d'agent."""
    workflow = StateGraph(AgentState)
    
    # Ajout des nœuds
    workflow.add_node("analyser", analyser)
    workflow.add_node("choisir_outil", choisir_outil)
    workflow.add_node("appeler_météo", appeler_météo)
    workflow.add_node("appeler_calculatrice", appeler_calculatrice)
    workflow.add_node("réponse_directe", réponse_directe)
    workflow.add_node("formuler_réponse", formuler_réponse)
    
    # Définition des arêtes avec routage dynamique
    workflow.set_entry_point("analyser")
    workflow.add_edge("analyser", "choisir_outil")
    workflow.add_conditional_edges("choisir_outil", router)
    workflow.add_edge("appeler_météo", "formuler_réponse")
    workflow.add_edge("appeler_calculatrice", "formuler_réponse")
    workflow.add_edge("réponse_directe", END)
    workflow.add_edge("formuler_réponse", END)
    
    return workflow.compile() 