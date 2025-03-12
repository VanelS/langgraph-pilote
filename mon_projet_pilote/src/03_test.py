from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Union, Annotated, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
import os

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Outils disponibles
@tool
def recherche_météo(location: str) -> str:
    """Recherche la météo pour une ville donnée"""
    # Simulons une API météo
    return f"Il fait 22°C et ensoleillé à {location}"

@tool
def calculatrice(expression: str) -> str:
    """Calcule une expression mathématique"""
    try:
        return f"Résultat: {eval(expression)}"
    except:
        return "Expression invalide"

# État typé
class AgentState(TypedDict):
    question: str
    thoughts: str
    tool_name: str
    tool_input: str
    observation: str
    answer: str

# Nœuds du graphe
def analyser(state: AgentState) -> AgentState:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    prompt = ChatPromptTemplate.from_template(
        "Question: {question}\nRéfléchissez au problème."
    )
    chain = prompt | llm
    thoughts = chain.invoke({"question": state["question"]})
    return {"thoughts": thoughts.content}

def choisir_outil(state: AgentState) -> AgentState:
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
    observation = recherche_météo.invoke(state["tool_input"])
    return {"observation": observation}

def appeler_calculatrice(state: AgentState) -> AgentState:
    observation = calculatrice.invoke(state["tool_input"])
    return {"observation": observation}

def réponse_directe(state: AgentState) -> AgentState:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    prompt = ChatPromptTemplate.from_template(
        "Question: {question}\nRéflexion: {thoughts}\n"
        "Donnez une réponse directe sans utiliser d'outils."
    )
    response = (prompt | llm).invoke(state)
    return {"observation": "Réponse directe", "answer": response.content}

def formuler_réponse(state: AgentState) -> AgentState:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    prompt = ChatPromptTemplate.from_template(
        "Question: {question}\nRéflexion: {thoughts}\n"
        "Observation: {observation}\n"
        "Formulez une réponse complète et claire."
    )
    response = (prompt | llm).invoke(state)
    return {"answer": response.content}

# Router pour diriger le flux
def router(state: AgentState) -> Literal["appeler_météo", "appeler_calculatrice", "réponse_directe"]:
    if state["tool_name"] == "recherche_météo":
        return "appeler_météo"
    elif state["tool_name"] == "calculatrice":
        return "appeler_calculatrice"
    else:
        return "réponse_directe"

# Construction du graphe
workflow = StateGraph(AgentState)
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

# Compilation
app = workflow.compile()

# Test
result = app.invoke({"question": "Quelle est la météo à Paris?"})
print(result["answer"])