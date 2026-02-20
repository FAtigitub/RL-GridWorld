"""
Tester l'extraction des Q-values
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from stable_baselines3 import DQN
from src.environment.gridworld import GridWorldEnv
from src.agents.q_value_extractor import QValueExtractor, format_decision_for_human
import json
import numpy as np

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def test_extraction():
    """Test l'extraction des informations de décision"""
    
    # Charger le modèle
    print("📦 Chargement du modèle...")
    model = DQN.load("models/best_model.zip")
    
    # Créer l'environnement (même config que l'entraînement!)
    env = GridWorldEnv(size=5, num_traps=1)
    obs, info = env.reset(seed=42)
    
    # Créer l'extracteur
    extractor = QValueExtractor(model, env)
    
    # Extraire le contexte
    print("\n🔍 Extraction du contexte de décision...\n")
    context = extractor.extract_decision_context(obs)
    
    # Afficher en format lisible
    print(format_decision_for_human(context))
    
    # Sauvegarder en JSON (convertir les types numpy d'abord)
    context_serializable = convert_numpy_types(context)
    with open("results/decision_context_example.json", "w", encoding="utf-8") as f:
        json.dump(context_serializable, f, indent=2, ensure_ascii=False)
    
    print("\n✅ Contexte sauvegardé dans 'results/decision_context_example.json'")
    
    env.close()

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    test_extraction()
