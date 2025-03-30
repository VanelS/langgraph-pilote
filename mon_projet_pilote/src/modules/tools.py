"""
Outils disponibles pour l'agent.
"""
from langchain_core.tools import tool
import requests

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