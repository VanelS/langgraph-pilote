"""
Définition de l'état de l'agent.
"""
from typing import TypedDict

class AgentState(TypedDict):
    """État typé pour l'agent."""
    question: str
    thoughts: str
    tool_name: str
    tool_input: str
    observation: str
    answer: str 