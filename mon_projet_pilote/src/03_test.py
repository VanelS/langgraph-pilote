"""
Agent LangGraph modulaire pour répondre aux questions en utilisant différents outils.
"""

# Imports
import os
from typing import TypedDict, List, Union, Annotated, Literal
from dotenv import load_dotenv
from graphviz import Digraph
import requests
import json

from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# -----------------------------
# Définition de l'état de l'agent
# -----------------------------

class AgentState(TypedDict):
    """État typé pour l'agent."""
    question: str
    thoughts: str
    tool_name: str
    tool_input: str
    observation: str
    answer: str

# -----------------------------
# Outils disponibles
# -----------------------------

@tool
def recherche_météo(location: str) -> str:
    """Recherche la météo pour une ville donnée en utilisant l'API Open-Meteo (sans clé API nécessaire)"""
    try:
        # D'abord, on doit géocoder la ville pour obtenir ses coordonnées
        geocoding_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=fr&format=json"
        geocoding_response = requests.get(geocoding_url)
        geocoding_data = geocoding_response.json()
        
        if not geocoding_data.get("results"):
            return f"Ville non trouvée: {location}"
            
        # Extraction des coordonnées
        lat = geocoding_data["results"][0]["latitude"]
        lon = geocoding_data["results"][0]["longitude"]
        city_name = geocoding_data["results"][0]["name"]
        
        # Requête météo avec les coordonnées
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m&timezone=auto&language=fr"
        weather_response = requests.get(weather_url)
        weather_data = weather_response.json()
        
        # Interprétation du code météo
        weather_codes = {
            0: "ciel dégagé",
            1: "principalement dégagé",
            2: "partiellement nuageux",
            3: "couvert",
            45: "brouillard",
            48: "brouillard givrant",
            51: "bruine légère",
            53: "bruine modérée",
            55: "bruine dense",
            56: "bruine verglaçante légère",
            57: "bruine verglaçante dense",
            61: "pluie légère",
            63: "pluie modérée",
            65: "pluie forte",
            66: "pluie verglaçante légère",
            67: "pluie verglaçante forte",
            71: "chute de neige légère",
            73: "chute de neige modérée",
            75: "chute de neige forte",
            77: "grains de neige",
            80: "averses de pluie légères",
            81: "averses de pluie modérées",
            82: "averses de pluie violentes",
            85: "averses de neige légères",
            86: "averses de neige fortes",
            95: "orage",
            96: "orage avec grêle légère",
            99: "orage avec grêle forte"
        }
        
        # Extraction des données météo actuelles
        current = weather_data.get("current", {})
        temp = current.get("temperature_2m", "N/A")
        humidity = current.get("relative_humidity_2m", "N/A")
        weather_code = current.get("weather_code", 0)
        wind_speed = current.get("wind_speed_10m", "N/A")
        
        weather_desc = weather_codes.get(weather_code, "conditions inconnues")
        
        return f"À {city_name}, il fait {temp}°C avec {weather_desc}. Humidité: {humidity}%, Vent: {wind_speed} km/h"
    except Exception as e:
        return f"Erreur technique lors de la recherche météo: {str(e)}"

@tool
def calculatrice(expression: str) -> str:
    """Calcule une expression mathématique"""
    try:
        return f"Résultat: {eval(expression)}"
    except:
        return "Expression invalide"

# -----------------------------
# Fonctions de raisonnement (nœuds du graphe)
# -----------------------------

def analyser(state: AgentState) -> AgentState:
    """Analyse la question initiale et génère des réflexions."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    prompt = ChatPromptTemplate.from_template(
        "Question: {question}\nRéfléchissez au problème."
    )
    chain = prompt | llm
    thoughts = chain.invoke({"question": state["question"]})
    return {"thoughts": thoughts.content}

def choisir_outil(state: AgentState) -> AgentState:
    """Choisit l'outil approprié et prépare l'entrée pour cet outil."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    outils = ["recherche_météo", "calculatrice", "réponse_directe"]
    
    prompt = ChatPromptTemplate.from_template(
        "Question: {question}\nRéflexion: {thoughts}\n"
        "Choisissez l'outil le plus approprié parmi: {outils}. "
        "Répondez uniquement avec le nom de l'outil."
    )
    chain = prompt | llm
    tool_name = chain.invoke({"question": state["question"], 
                              "thoughts": state["thoughts"],
                              "outils": ", ".join(outils)})
    
    # Préparer l'entrée de l'outil
    if tool_name.content == "recherche_météo":
        prompt_input = ChatPromptTemplate.from_template(
            "Extrayez le nom de la ville de la question: {question}"
        )
        tool_input = (prompt_input | llm).invoke({"question": state["question"]})
    elif tool_name.content == "calculatrice":
        prompt_input = ChatPromptTemplate.from_template(
            "Extrayez l'expression mathématique de la question: {question}"
        )
        tool_input = (prompt_input | llm).invoke({"question": state["question"]})
    else:
        tool_input = ChatPromptTemplate.from_template("").invoke({})
    
    return {"tool_name": tool_name.content, "tool_input": tool_input.content}

def appeler_météo(state: AgentState) -> AgentState:
    """Appelle l'outil de météo avec l'entrée préparée."""
    observation = recherche_météo.invoke(state["tool_input"])
    return {"observation": observation}

def appeler_calculatrice(state: AgentState) -> AgentState:
    """Appelle la calculatrice avec l'entrée préparée."""
    observation = calculatrice.invoke(state["tool_input"])
    return {"observation": observation}

def réponse_directe(state: AgentState) -> AgentState:
    """Génère une réponse directe sans utiliser d'outils."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    prompt = ChatPromptTemplate.from_template(
        "Question: {question}\nRéflexion: {thoughts}\n"
        "Donnez une réponse directe sans utiliser d'outils."
    )
    response = (prompt | llm).invoke(state)
    return {"observation": "Réponse directe", "answer": response.content}

def formuler_réponse(state: AgentState) -> AgentState:
    """Formule une réponse finale basée sur l'observation de l'outil."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    prompt = ChatPromptTemplate.from_template(
        "Question: {question}\nRéflexion: {thoughts}\n"
        "Observation: {observation}\n"
        "Formulez une réponse complète et claire."
    )
    response = (prompt | llm).invoke(state)
    return {"answer": response.content}

# -----------------------------
# Routeur pour diriger le flux
# -----------------------------

def router(state: AgentState) -> Literal["appeler_météo", "appeler_calculatrice", "réponse_directe"]:
    """Détermine quel nœud appeler en fonction de l'outil choisi."""
    if state["tool_name"] == "recherche_météo":
        return "appeler_météo"
    elif state["tool_name"] == "calculatrice":
        return "appeler_calculatrice"
    else:
        return "réponse_directe"

# -----------------------------
# Fonctions de visualisation
# -----------------------------

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

# -----------------------------
# Construction et compilation du graphe
# -----------------------------

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

# -----------------------------
# Point d'entrée principal
# -----------------------------

if __name__ == "__main__":
    # Construction du graphe
    app = build_agent_graph()
    
    # Test avec une question
    result = app.invoke({"question": "Quelle est la météo à Paris?"})
    print(result["answer"])
    
    # Afficher la structure du graphe pour visualisation
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