"""
Tester l'intégration du LLM — explication pour chaque mouvement de l'épisode
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from stable_baselines3 import DQN
from src.environment.gridworld import GridWorldEnv
from src.agents.q_value_extractor import QValueExtractor, format_decision_for_human
from src.llm.explainer import RLExplainer


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


def test_llm_explanation():
    """Test complet : Explication rule-based du premier mouvement (quota-free)"""

    print("=" * 70)
    print("🧪 TEST EXPLAINABILITY — PREMIER MOUVEMENT (RULE-BASED)")
    print("=" * 70)

    # 1. Charger le modèle RL
    print("\n📦 Chargement du modèle RL...")
    model = DQN.load("models/best_model.zip")

    # 2. Créer l'environnement
    env = GridWorldEnv(size=5, num_traps=1)
    obs, info = env.reset(seed=42)

    # 3. Créer l'extracteur et l'explainer
    extractor = QValueExtractor(model, env)
    explainer = RLExplainer(model_name="gemini-2.5-flash", temperature=0.3)

    print(f"\n🗺️  Grille {env.size}x{env.size}")
    print(f"   Départ     : {env.agent_pos.tolist()}")
    print(f"   Objectif   : {env.goal_pos.tolist()}")
    print(f"   Pièges     : {[t.tolist() for t in env.traps]}")

    # 4. Boucle sur l'épisode complet (SANS appel LLM pour économiser quota)
    all_steps = []
    step_num = 0
    terminated = False
    truncated = False
    max_steps = env.size * env.size * 2  # sécurité

    print("\n" + "=" * 70)
    print("🚀 DÉBUT DE L'ÉPISODE — COLLECTE DES DONNÉES")
    print("=" * 70)

    while not terminated and not truncated and step_num < max_steps:
        step_num += 1

        # --- Extraire le contexte AVANT le mouvement ---
        context = extractor.extract_decision_context(obs)
        best_action = context["best_action"]

        print(f"\n{'─'*70}")
        print(f"📍 ÉTAPE {step_num}  |  Position: {context['state']['agent_position']}"
              f"  →  Action: {best_action['name']} (Q={best_action['q_value']:.2f})")

        # --- Exécuter l'action dans l'environnement ---
        action_idx = best_action["index"]
        obs, reward, terminated, truncated, info = env.step(action_idx)

        print(f"   ✅ Nouvelle position: {info['agent_pos']} | Récompense: {reward:.2f}")

        if terminated:
            if reward > 0:
                print(f"\n   🎉 OBJECTIF ATTEINT en {step_num} étapes !")
            else:
                print(f"\n   💀 PIÈGE ! Épisode terminé à l'étape {step_num}.")

        # Sauvegarder les données de cette étape
        step_data = {
            "step": step_num,
            "position_before": context["state"]["agent_position"],
            "position_after": info["agent_pos"],
            "action": best_action["name"],
            "q_value": best_action["q_value"],
            "reward": float(reward),
            "terminated": terminated,
            "truncated": truncated,
            "context": convert_numpy_types(context),
        }
        all_steps.append(step_data)

    # 5. Explication SANS API pour le premier mouvement seulement (quota-free)
    print("\n" + "=" * 70)
    print("🤖 EXPLICATION DU PREMIER MOUVEMENT (rule-based, sans API)")
    print("=" * 70)
    
    first_step_explanation_result = explainer.explain_first_step_only(all_steps[0])
    
    print("\n" + "=" * 70)
    print("💬 EXPLICATION DU PREMIER PAS:")
    print("=" * 70)
    print(first_step_explanation_result["explanation"])
    print("=" * 70)
    
    episode_explanation = first_step_explanation_result["explanation"]
    total_tokens = first_step_explanation_result.get("total_tokens", 0)

    # 6. Résumé de l'épisode
    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ DE L'ÉPISODE")
    print("=" * 70)
    print(f"   Nombre de mouvements : {step_num}")
    total_reward = sum(s["reward"] for s in all_steps)
    print(f"   Récompense totale    : {total_reward:.2f}")
    print(f"   Tokens utilisés      : {total_tokens} (rule-based, pas d'API)")
    success = any(s["terminated"] and s["reward"] > 0 for s in all_steps)
    print(f"   Résultat             : {'✅ Succès' if success else '❌ Échec'}")

    # 7. Sauvegarder tous les résultats
    full_result = {
        "episode_summary": {
            "total_steps": step_num,
            "total_reward": total_reward,
            "success": success,
            "total_tokens_used": total_tokens,
            "model_rl": "DQN best_model",
            "model_llm": "rule-based (no API)",
            "first_step_explanation": episode_explanation,
        },
        "steps": all_steps,
    }

    os.makedirs("results", exist_ok=True)
    output_path = "results/llm_explanation_test.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_result, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Résultats complets sauvegardés dans '{output_path}'")
    env.close()


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    
    # Note: Using rule-based explanation for first step only (no API required)
    # This avoids Gemini quota issues
    
    test_llm_explanation()