from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from typing import TypedDict

from dotenv import load_dotenv
import os

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Structure d'état typée
class AgentState(TypedDict):
    question: str
    reasoning: str
    answer: str

# Définition des nœuds
def raisonner(state: AgentState) -> AgentState:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    prompt = ChatPromptTemplate.from_template(
        "Question: {question}\nRéfléchissez étape par étape."
    )
    chain = prompt | llm
    reasoning = chain.invoke({"question": state["question"]})
    return {"reasoning": reasoning.content}

def répondre(state: AgentState) -> AgentState:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    prompt = ChatPromptTemplate.from_template(
        "Question: {question}\nRaisonnement: {reasoning}\nDonnez une réponse concise."
    )
    chain = prompt | llm
    response = chain.invoke(state)
    return {"answer": response.content}

def recherche(state: AgentState) -> AgentState:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    prompt = ChatPromptTemplate.from_template(
        "Effectuez une recherche approfondie pour la question suivante : {question}."
    )
    chain = prompt | llm
    search_result = chain.invoke({"question": state["question"]})
    return {"reasoning": search_result.content}

# Condition d'activation du nœud de recherche
def condition_recherche(state: AgentState) -> str:
    if "recherche" in state["question"].lower():
        return "recherche"
    return "raisonner"

# Construction du graphe
workflow = StateGraph(AgentState)
workflow.add_node("raisonner", raisonner)
workflow.add_node("répondre", répondre)
workflow.add_node("recherche", recherche)

# Définition des arêtes (flux)
workflow.set_conditional_entry_point(condition_recherche)
workflow.add_edge("recherche", "répondre")
workflow.add_edge("raisonner", "répondre")
workflow.set_finish_point("répondre")

# Compilation du graphe
app = workflow.compile()

# Test du graphe
result = app.invoke({"question": "Effectuez une recherche sur la couleur du ciel."})
print(result["answer"])
