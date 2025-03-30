"""
Fonctions de visualisation du graphe d'agent.
"""
from graphviz import Digraph

def print_graph_structure(graph):
    """Affiche une représentation textuelle du graphe."""
    print("\n=== Structure du graphe d'agent ===")
    print("Nœuds:", ", ".join(graph.nodes))
    print("\nArêtes:")
    for edge in graph.edges:
        print(f"  {edge[0]} -> {edge[1]}")
    
    print("=== Fin de la structure ===")

def visualize_graph(graph):
    """Génère une visualisation graphique du graphe."""
    dot = Digraph(comment='Agent Workflow')
    
    # Définir les styles
    dot.attr('node', shape='box', style='filled', color='lightblue')
    dot.attr('edge', color='navy')
    
    # Ajouter les nœuds
    for node in graph.nodes:
        dot.node(node, node)
    
    # Ajouter le nœud END
    dot.node("__end__", "END", shape='doublecircle', color='darkgreen', fontcolor='white', style='filled')
    
    # Ajouter le nœud START
    dot.node("__start__", "START", shape='circle', color='darkgreen', fontcolor='white', style='filled')
    
    # Ajouter les arêtes
    for edge in graph.edges:
        source, target = edge
        # Ajouter un style spécial pour les arêtes conditionnelles
        if source == "choisir_outil":
            dot.edge(source, target, style='dashed', label='condition')
        else:
            dot.edge(source, target)
    
    # Générer et sauvegarder le graphe
    dot.render('agent_workflow', format='png', cleanup=True)
    print("Graphe généré: agent_workflow.png") 