"""
Tester la détection des hallucinations
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from stable_baselines3 import DQN
from src.environment.gridworld import GridWorldEnv
from src.agents.q_value_extractor import QValueExtractor
from src.llm.explainer import RLExplainer
from src.llm.hallucination_detector import HallucinationDetector
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


def test_hallucination_detection():
    """Test de la détection d'hallucinations (RULE-BASED, sans API)"""
    
    print("="*70)
    print("⚠️  TEST DE DÉTECTION DES HALLUCINATIONS (RULE-BASED)")
    print("="*70)
    
    # Charger le modèle et environnement
    model = DQN.load("models/best_model.zip")
    env = GridWorldEnv(size=5, num_traps=3)
    
    # Créer les modules
    extractor = QValueExtractor(model, env)
    explainer = RLExplainer(temperature=0.3)
    detector = HallucinationDetector()
    
    # Collecter UNE SEULE explication (SANS API pour éviter quota)
    print("\n📊 Génération de 1 explication rule-based (AUCUN appel API)...\n")
    
    explanations_data = []
    
    obs, info = env.reset(seed=0)
    context = extractor.extract_decision_context(obs)
    
    print(f"État 1:")
    print(f"  Position: {context['state']['agent_position']}")
    print(f"  Action choisie: {context['best_action']['name']}")
    
    # Créer step_data pour explain_first_step_only
    step_data = {
        "step": 1,
        "position_before": context["state"]["agent_position"],
        "action": context["best_action"]["name"],
        "q_value": context["best_action"]["q_value"],
        "context": context,
    }
    
    # Générer l'explication SANS API (rule-based)
    result = explainer.explain_first_step_only(step_data)
    explanation = result["explanation"]
    
    print(f"  Explication: {explanation[:100]}...")
    
    # Valider
    validation = detector.validate_explanation(explanation, context)
    
    print(f"  ✅ Valide: {validation['is_valid']}")
    print(f"  📊 Score fiabilité: {validation['reliability_score']}/100")
    
    if validation['issues']:
        print(f"  ⚠️  Problèmes détectés:")
        for issue in validation['issues']:
            print(f"     - {issue}")
    
    print()
    
    explanations_data.append({
        "state": 0,
        "context": convert_numpy_types(context),
        "explanation": explanation,
        "validation": convert_numpy_types(validation)
    })
    
    # Statistiques globales
    print("="*70)
    print("📈 STATISTIQUES (1 explication rule-based testée)")
    print("="*70)
    
    batch_result = detector.batch_validate([
        (item["explanation"], item["context"]) 
        for item in explanations_data
    ])
    
    print(f"\nTotal d'explications: {batch_result['total_explanations']}")
    print(f"Explications valides: {batch_result['valid_explanations']}")
    print(f"Explications invalides: {batch_result['invalid_explanations']}")
    print(f"Taux de validité: {batch_result['validity_rate']:.1f}%")
    print(f"Score moyen de fiabilité: {batch_result['average_reliability_score']:.1f}/100")
    
    print(f"\n📊 Types de problèmes:")
    for issue_type, count in batch_result['issue_types'].items():
        if count > 0:
            print(f"   - {issue_type}: {count}")
    
    # Sauvegarder les résultats (avec conversion numpy)
    with open("results/hallucination_test_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "explanations": explanations_data,
            "batch_statistics": convert_numpy_types(batch_result)
        }, f, indent=2, ensure_ascii=False)
    
    print("\n✅ Résultats sauvegardés dans 'results/hallucination_test_results.json'")
    print("💡 Explication générée SANS appel API Gemini (économie de quota)")
    
    env.close()

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    test_hallucination_detection()