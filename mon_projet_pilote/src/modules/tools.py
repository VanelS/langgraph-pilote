"""
Outils disponibles pour l'agent.
"""
from langchain_core.tools import tool
import requests
import re
from typing import Optional

from .errors import handle_tool_errors, validate_input, logger

def is_valid_location(location: str) -> bool:
    """Valide si une chaîne est un nom de ville potentiellement valide.
    
    Args:
        location: Nom de ville à valider
        
    Returns:
        bool: True si le nom semble valide, False sinon
    """
    # Vérifie que l'entrée n'est pas vide et contient des lettres
    return bool(location) and bool(re.match(r'^[a-zA-Z\s\-éèêëàâäôöùûüç\']+$', location))

def is_valid_expression(expression: str) -> bool:
    """Valide si une chaîne est une expression mathématique potentiellement valide.
    
    Args:
        expression: Expression mathématique à valider
        
    Returns:
        bool: True si l'expression semble valide, False sinon
    """
    # Vérification basique que l'expression ne contient que des caractères autorisés
    return bool(expression) and bool(re.match(r'^[\d\s\+\-\*\/\(\)\.\,\%]+$', expression))

@tool
@handle_tool_errors(fallback_response="Je n'ai pas pu obtenir les informations météo.")
@validate_input(is_valid_location, "Le nom de ville fourni n'est pas valide.")
def recherche_météo(location: str) -> str:
    """Recherche la météo pour une ville donnée en utilisant l'API Open-Meteo (sans clé API nécessaire)"""
    logger.info(f"Recherche météo pour: {location}")
    
    # Vérification supplémentaire de la longueur
    if len(location) < 2:
        raise ValueError("Le nom de ville est trop court.")
        
    # D'abord, on doit géocoder la ville pour obtenir ses coordonnées
    try:
        geocoding_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=fr&format=json"
        geocoding_response = requests.get(geocoding_url, timeout=10)
        geocoding_response.raise_for_status()  # Lève une exception en cas d'erreur HTTP
        geocoding_data = geocoding_response.json()
        
        if not geocoding_data.get("results"):
            logger.warning(f"Ville non trouvée: {location}")
            return f"Ville non trouvée: {location}"
            
        # Extraction des coordonnées
        lat = geocoding_data["results"][0]["latitude"]
        lon = geocoding_data["results"][0]["longitude"]
        city_name = geocoding_data["results"][0]["name"]
        
        # Requête météo avec les coordonnées
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m&timezone=auto&language=fr"
        weather_response = requests.get(weather_url, timeout=10)
        weather_response.raise_for_status()
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
        
        # Extraction des données météo actuelles avec vérification
        current = weather_data.get("current", {})
        temp = current.get("temperature_2m")
        humidity = current.get("relative_humidity_2m")
        weather_code = current.get("weather_code")
        wind_speed = current.get("wind_speed_10m")
        
        # Vérification des données manquantes
        if temp is None or humidity is None or weather_code is None or wind_speed is None:
            logger.warning(f"Données météo incomplètes pour {city_name}")
            return f"Les données météo pour {city_name} sont incomplètes."
        
        weather_desc = weather_codes.get(weather_code, "conditions inconnues")
        
        logger.info(f"Météo récupérée avec succès pour {city_name}")
        return f"À {city_name}, il fait {temp}°C avec {weather_desc}. Humidité: {humidity}%, Vent: {wind_speed} km/h"
    
    except requests.exceptions.Timeout:
        logger.error(f"Timeout lors de la connexion à l'API météo pour {location}")
        raise TimeoutError("Les serveurs météo mettent trop de temps à répondre, veuillez réessayer plus tard.")
    
    except requests.exceptions.HTTPError as e:
        logger.error(f"Erreur HTTP lors de la requête météo: {str(e)}")
        raise ValueError(f"Erreur lors de la connexion aux services météo: {e.response.status_code}")
    
    except requests.exceptions.ConnectionError:
        logger.error(f"Erreur de connexion aux serveurs météo pour {location}")
        raise ConnectionError("Impossible de se connecter aux serveurs météo, vérifiez votre connexion internet.")
    
    except Exception as e:
        logger.error(f"Erreur inattendue dans recherche_météo: {str(e)}")
        raise

@tool
@handle_tool_errors(fallback_response="Je n'ai pas pu calculer cette expression.")
@validate_input(is_valid_expression, "L'expression mathématique fournie n'est pas valide.")
def calculatrice(expression: str) -> str:
    """Calcule une expression mathématique"""
    logger.info(f"Calcul de l'expression: {expression}")
    
    # Nettoyage de l'expression
    expression = expression.replace(',', '.')  # Remplace les virgules par des points
    
    # Limite la longueur de l'expression pour éviter les attaques
    if len(expression) > 100:
        logger.warning(f"Expression trop longue: {expression[:20]}...")
        raise ValueError("L'expression est trop longue (max 100 caractères).")
    
    try:
        # Évaluation sécurisée avec vérification du résultat
        result = eval(expression)
        
        # Vérification du résultat
        if isinstance(result, complex):
            return f"Résultat: {result} (nombre complexe)"
        
        # Formater les nombres avec beaucoup de décimales
        if isinstance(result, float):
            # Limiter à 6 décimales pour la lisibilité
            result = round(result, 6)
        
        logger.info(f"Calcul réussi: {expression} = {result}")
        return f"Résultat: {result}"
    
    except ZeroDivisionError:
        logger.warning(f"Division par zéro dans l'expression: {expression}")
        return "Erreur: Division par zéro"
    
    except SyntaxError:
        logger.warning(f"Erreur de syntaxe dans l'expression: {expression}")
        return "Erreur: Syntaxe incorrecte dans l'expression mathématique"
    
    except (NameError, TypeError):
        logger.warning(f"Expression non évaluable: {expression}")
        return "Erreur: L'expression contient des éléments non évaluables"
    
    except Exception as e:
        logger.error(f"Erreur inattendue dans calculatrice: {str(e)}")
        raise 