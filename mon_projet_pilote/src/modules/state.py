"""
Définition de l'état de l'agent.
"""
from typing import TypedDict, Optional, Any

class AgentState(TypedDict, total=False):
    """État typé pour l'agent.
    
    Les champs marqués comme optionnels peuvent ne pas être présents
    à certaines étapes du traitement ou en cas d'erreur.
    """
    # Champs principaux
    question: str
    thoughts: Optional[str]
    tool_name: Optional[str]
    tool_input: Optional[str]
    observation: Optional[str]
    answer: Optional[str]
    
    # Champs de gestion d'erreurs
    error: Optional[bool]
    error_message: Optional[str]
    error_type: Optional[str]
    retry_count: Optional[int]
    fallback_used: Optional[bool] 