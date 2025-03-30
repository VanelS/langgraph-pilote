"""
Fonctions de raisonnement pour l'agent (nœuds du graphe).
"""
from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from .state import AgentState
from .tools import recherche_météo, calculatrice

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

def router(state: AgentState) -> Literal["appeler_météo", "appeler_calculatrice", "réponse_directe"]:
    """Détermine quel nœud appeler en fonction de l'outil choisi."""
    if state["tool_name"] == "recherche_météo":
        return "appeler_météo"
    elif state["tool_name"] == "calculatrice":
        return "appeler_calculatrice"
    else:
        return "réponse_directe" 