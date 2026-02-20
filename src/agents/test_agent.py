"""
Tester l'agent entraîné avec analyses avancées et visualisations
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
import time

from stable_baselines3 import DQN
from src.environment.gridworld import GridWorldEnv
from src.agents.q_value_extractor import QValueExtractor


def plot_trajectory(trajectory, env_info, episode_num, save_path=None):
    """Visualiser la trajectoire de l'agent sur une grille"""
    size = env_info['size']
    goal = env_info['goal_pos']
    traps = env_info['traps']
    start = trajectory[0]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Dessiner la grille
    for i in range(size + 1):
        ax.axhline(i, color='gray', linewidth=0.5)
        ax.axvline(i, color='gray', linewidth=0.5)
    
    # Marquer le départ (cercle bleu)
    ax.add_patch(patches.Circle((start[0] + 0.5, start[1] + 0.5), 
                                0.3, color='cyan', zorder=3))
    ax.text(start[0] + 0.5, start[1] + 0.5, 'S', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Marquer l'objectif (étoile jaune)
    ax.add_patch(patches.RegularPolygon((goal[0] + 0.5, goal[1] + 0.5), 
                                        5, 0.4, color='gold', zorder=3))
    ax.text(goal[0] + 0.5, goal[1] + 0.2, 'G', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Marquer les pièges (X rouge)
    for trap in traps:
        ax.add_patch(patches.Rectangle((trap[0], trap[1]), 1, 1, 
                                       color='red', alpha=0.3, zorder=2))
        ax.text(trap[0] + 0.5, trap[1] + 0.5, 'X', 
                ha='center', va='center', fontsize=20, fontweight='bold', color='red')
    
    # Dessiner la trajectoire (ligne bleue avec points)
    x_coords = [pos[0] + 0.5 for pos in trajectory]
    y_coords = [pos[1] + 0.5 for pos in trajectory]
    ax.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.6, zorder=1)
    ax.plot(x_coords, y_coords, 'bo', markersize=8, alpha=0.8, zorder=2)
    
    # Numéroter les steps
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        if i > 0 and i < len(trajectory) - 1:  # Skip start and end
            ax.text(x + 0.15, y + 0.15, str(i), fontsize=8, color='blue')
    
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect('equal')
    ax.set_title(f'Trajectoire - Épisode {episode_num}\n{len(trajectory)} steps', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📊 Trajectoire sauvegardée: {save_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_q_values(q_values_history, action_names):
    """Analyser l'évolution des Q-values"""
    if not q_values_history:
        return None
    
    num_steps = len(q_values_history)
    num_actions = len(q_values_history[0])
    
    # Créer graphique
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Graphique 1: Évolution des Q-values
    for action_idx in range(num_actions):
        q_vals = [q_vals[action_idx] for q_vals in q_values_history]
        ax1.plot(range(num_steps), q_vals, 
                marker='o', label=action_names[action_idx], linewidth=2)
    
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Q-Value', fontsize=12)
    ax1.set_title('Évolution des Q-Values par Action', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Graphique 2: Max Q-value par step
    max_q_vals = [max(q_vals) for q_vals in q_values_history]
    avg_q_vals = [np.mean(q_vals) for q_vals in q_values_history]
    
    ax2.plot(range(num_steps), max_q_vals, 'g-', marker='o', 
            label='Max Q-Value', linewidth=2)
    ax2.plot(range(num_steps), avg_q_vals, 'b--', marker='s', 
            label='Avg Q-Value', linewidth=2)
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Q-Value', fontsize=12)
    ax2.set_title('Q-Values Max et Moyenne', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def test_agent(model_path="models/dqn_gridworld_final.zip", 
               episodes=5, 
               render=True, 
               speed=0.3,
               save_plots=False,
               save_results=False,
               analyze_qvalues=False,
               verbose=True):
    """
    Tester l'agent avec analyses avancées
    
    Args:
        model_path: Chemin vers le modèle
        episodes: Nombre d'épisodes à tester
        render: Afficher le rendu Pygame
        speed: Vitesse d'animation (secondes entre steps, 0 = rapide)
        save_plots: Sauvegarder les graphiques
        save_results: Sauvegarder les résultats JSON
        analyze_qvalues: Analyser les Q-values
        verbose: Mode verbeux
    """
    
    print("="*70)
    print("🧪 TEST AVANCÉ DE L'AGENT ENTRAÎNÉ")
    print("="*70)
    
    # Charger le modèle
    print(f"\n📦 Chargement du modèle: {model_path}")
    try:
        model = DQN.load(model_path)
    except FileNotFoundError:
        print(f"❌ Modèle non trouvé: {model_path}")
        print("💡 Modèles disponibles:")
        import os
        if os.path.exists("models"):
            for f in os.listdir("models"):
                if f.endswith('.zip'):
                    print(f"   - models/{f}")
        return
    
    # Créer l'environnement
    render_mode = "human" if render else None
    num_traps = 3  # Peut être modifié (1-5 pièges supportés)
    env = GridWorldEnv(size=5, num_traps=num_traps, render_mode=render_mode)
    extractor = QValueExtractor(model, env)
    
    print(f"\n🎮 Configuration:")
    print(f"   Épisodes: {episodes}")
    print(f"   Grille: 5x5")
    print(f"   Pièges: {num_traps}")
    print(f"   Rendu: {'Activé' if render else 'Désactivé'}")
    print(f"   Vitesse: {speed}s/step" if speed > 0 else "   Vitesse: Maximum")
    print(f"   Analyse Q-values: {'Oui' if analyze_qvalues else 'Non'}")
    
    # Statistiques globales
    all_results = []
    total_rewards = []
    total_steps = []
    successes = 0
    action_counts = Counter()
    
    # Créer dossier pour les résultats
    if save_plots or save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(f"test_results_{timestamp}")
        results_dir.mkdir(exist_ok=True)
        print(f"\n📁 Dossier de résultats: {results_dir}")
    
    print(f"\n🚀 Lancement des tests...\n")
    
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        step = 0
        
        # Données de l'épisode
        trajectory = [info['agent_pos'].copy()]
        actions_taken = []
        rewards_history = []
        q_values_history = []
        
        if verbose:
            print(f"{'─'*70}")
            print(f"📍 Épisode {episode + 1}/{episodes}")
            print(f"   Départ: {info['agent_pos']} → Objectif: {info['goal_pos']}")
            print(f"   Pièges: {info['traps']}")
        
        while not done:
            # Extraire Q-values si analyse demandée
            if analyze_qvalues:
                context = extractor.extract_decision_context(obs)
                q_vals = [context['q_values'][name] for name in env.action_to_name.values()]
                q_values_history.append(q_vals)
            
            # Prédire l'action (mode déterministe)
            action, _states = model.predict(obs, deterministic=True)
            action = int(action)
            
            # Exécuter l'action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            step += 1
            
            # Enregistrer
            trajectory.append(info['agent_pos'].copy())
            actions_taken.append(info['action_name'])
            rewards_history.append(reward)
            action_counts[info['action_name']] += 1
            
            # Afficher
            if render:
                env.render()
            
            if speed > 0:
                time.sleep(speed)
            
            if verbose:
                direction = ""
                if step > 1:
                    prev = trajectory[-2]
                    curr = trajectory[-1]
                    if curr[1] < prev[1]: direction = "↑"
                    elif curr[1] > prev[1]: direction = "↓"
                    elif curr[0] < prev[0]: direction = "←"
                    elif curr[0] > prev[0]: direction = "→"
                
                print(f"   Step {step:2d}: {info['action_name']:7s} {direction} → "
                      f"{info['agent_pos']} | R={reward:6.2f} | Total={episode_reward:6.2f}")
        
        total_rewards.append(episode_reward)
        total_steps.append(step)
        
        # Déterminer succès
        success = episode_reward > 50
        if success:
            successes += 1
        
        if verbose:
            status = "✅ SUCCÈS" if success else "❌ ÉCHEC"
            print(f"\n   {status} - Récompense finale: {episode_reward:.2f} ({step} steps)")
        
        # Sauvegarder trajectoire
        if save_plots:
            plot_path = results_dir / f"trajectory_ep{episode+1}.png"
            plot_trajectory(trajectory, 
                          {'size': 5, 'goal_pos': info['goal_pos'], 
                           'traps': info['traps']},
                          episode + 1, 
                          plot_path)
        
        # Analyser Q-values
        if analyze_qvalues and q_values_history:
            fig = analyze_q_values(q_values_history, env.action_to_name)
            if save_plots and fig:
                fig.savefig(results_dir / f"qvalues_ep{episode+1}.png", 
                           dpi=150, bbox_inches='tight')
                print(f"  📊 Q-values sauvegardés: qvalues_ep{episode+1}.png")
                plt.close(fig)
            elif fig:
                plt.show()
                plt.close(fig)
        
        # Enregistrer résultats épisode
        all_results.append({
            "episode": episode + 1,
            "success": success,
            "reward": float(episode_reward),
            "steps": step,
            "start_pos": trajectory[0],
            "goal_pos": info['goal_pos'],  # Already a list from env
            "trajectory": trajectory,
            "actions": actions_taken,
            "rewards": rewards_history
        })
    
    env.close()
    
    # Statistiques finales détaillées
    print(f"\n{'='*70}")
    print("📊 STATISTIQUES DÉTAILLÉES")
    print(f"{'='*70}")
    
    success_rate = successes / episodes * 100
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)
    std_reward = np.std(total_rewards)
    std_steps = np.std(total_steps)
    
    print(f"\n🎯 Performance Globale:")
    print(f"   Taux de succès:    {successes}/{episodes} ({success_rate:.1f}%)")
    print(f"   Récompense moy.:   {avg_reward:.2f} (±{std_reward:.2f})")
    print(f"   Steps moyens:      {avg_steps:.1f} (±{std_steps:.1f})")
    print(f"   Meilleure récomp.: {max(total_rewards):.2f}")
    print(f"   Pire récompense:   {min(total_rewards):.2f}")
    print(f"   Steps min/max:     {min(total_steps)}/{max(total_steps)}")
    
    print(f"\n🎲 Distribution des Actions:")
    total_actions = sum(action_counts.values())
    for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_actions * 100
        bar = "█" * int(percentage / 2)
        print(f"   {action:7s}: {count:4d} ({percentage:5.1f}%) {bar}")
    
    # Graphique de synthèse
    if save_plots or episodes > 1:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Graphique 1: Récompenses par épisode
        colors = ['green' if r > 50 else 'red' for r in total_rewards]
        ax1.bar(range(1, episodes + 1), total_rewards, color=colors, alpha=0.7)
        ax1.axhline(y=50, color='orange', linestyle='--', label='Seuil succès')
        ax1.set_xlabel('Épisode')
        ax1.set_ylabel('Récompense')
        ax1.set_title('Récompenses par Épisode')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Graphique 2: Steps par épisode
        ax2.plot(range(1, episodes + 1), total_steps, 'b-o', linewidth=2)
        ax2.set_xlabel('Épisode')
        ax2.set_ylabel('Nombre de Steps')
        ax2.set_title('Longueur des Épisodes')
        ax2.grid(True, alpha=0.3)
        
        # Graphique 3: Distribution des actions
        actions_list = list(action_counts.keys())
        counts_list = [action_counts[a] for a in actions_list]
        ax3.pie(counts_list, labels=actions_list, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Distribution des Actions')
        
        # Graphique 4: Histogramme des récompenses
        ax4.hist(total_rewards, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
        ax4.axvline(x=avg_reward, color='red', linestyle='--', linewidth=2, label=f'Moy: {avg_reward:.2f}')
        ax4.set_xlabel('Récompense')
        ax4.set_ylabel('Fréquence')
        ax4.set_title('Distribution des Récompenses')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            summary_path = results_dir / "summary_statistics.png"
            plt.savefig(summary_path, dpi=150, bbox_inches='tight')
            print(f"\n📊 Statistiques graphiques: {summary_path}")
        else:
            plt.show()
        plt.close()
    
    # Sauvegarder résultats JSON
    if save_results:
        results_json = {
            "test_date": datetime.now().isoformat(),
            "model_path": model_path,
            "config": {
                "episodes": episodes,
                "grid_size": 5,
                "num_traps": num_traps
            },
            "summary": {
                "success_rate": float(success_rate),
                "avg_reward": float(avg_reward),
                "std_reward": float(std_reward),
                "avg_steps": float(avg_steps),
                "std_steps": float(std_steps),
                "min_steps": int(min(total_steps)),
                "max_steps": int(max(total_steps)),
                "best_reward": float(max(total_rewards)),
                "worst_reward": float(min(total_rewards))
            },
            "action_distribution": dict(action_counts),
            "episodes": all_results
        }
        
        json_path = results_dir / "test_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, indent=2, ensure_ascii=False)
        print(f"💾 Résultats JSON: {json_path}")
    
    print(f"\n{'='*70}")
    print("✅ TESTS TERMINÉS")
    print(f"{'='*70}\n")
    
    return {
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "results": all_results
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tester l'agent DQN entraîné")
    parser.add_argument("--model", type=str, default="models/dqn_gridworld_final.zip",
                       help="Chemin vers le modèle")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Nombre d'épisodes à tester")
    parser.add_argument("--no-render", action="store_true",
                       help="Désactiver le rendu Pygame")
    parser.add_argument("--speed", type=float, default=0.3,
                       help="Vitesse d'animation (secondes, 0=max)")
    parser.add_argument("--save-plots", action="store_true",
                       help="Sauvegarder les graphiques")
    parser.add_argument("--save-results", action="store_true",
                       help="Sauvegarder les résultats JSON")
    parser.add_argument("--analyze-qvalues", action="store_true",
                       help="Analyser les Q-values")
    parser.add_argument("--quiet", action="store_true",
                       help="Mode silencieux")
    
    args = parser.parse_args()
    
    test_agent(
        model_path=args.model,
        episodes=args.episodes,
        render=not args.no_render,
        speed=args.speed,
        save_plots=args.save_plots,
        save_results=args.save_results,
        analyze_qvalues=args.analyze_qvalues,
        verbose=not args.quiet
    )