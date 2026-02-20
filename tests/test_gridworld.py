"""
Tests pour l'environnement Gridworld
"""

import sys
from pathlib import Path

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.environment.gridworld import GridWorldEnv

def test_environment_creation():
    """Test la création de l'environnement"""
    env = GridWorldEnv(size=5, num_traps=3)
    obs, info = env.reset()
    
    print("✅ Test 1: Création de l'environnement")
    print(f"   Observation initiale: {obs}")
    print(f"   Position agent: {info['agent_pos']}")
    print(f"   Position objectif: {info['goal_pos']}")
    print(f"   Nombre de pièges: {len(info['traps'])}")
    print(f"   Pièges: {info['traps']}\n")
    
    assert obs.shape == (2,), "L'observation doit être de dimension 2"
    assert len(info['traps']) == 3, "Doit avoir 3 pièges"
    env.close()

def test_movements():
    """Test les mouvements de l'agent"""
    env = GridWorldEnv(size=5, num_traps=0, render_mode=None)
    obs, info = env.reset()
    
    print("✅ Test 2: Mouvements de l'agent")
    
    # Test mouvement droite
    obs, reward, terminated, truncated, info = env.step(3)  # Droite
    print(f"   Après Droite: position={info['agent_pos']}, reward={reward:.2f}")
    assert info['agent_pos'][0] == 1, "Devrait être en x=1"
    
    # Test mouvement bas
    obs, reward, terminated, truncated, info = env.step(1)  # Bas
    print(f"   Après Bas: position={info['agent_pos']}, reward={reward:.2f}")
    assert info['agent_pos'][1] == 1, "Devrait être en y=1"
    
    # Test mouvement haut (retour)
    obs, reward, terminated, truncated, info = env.step(0)  # Haut
    print(f"   Après Haut: position={info['agent_pos']}, reward={reward:.2f}")
    assert info['agent_pos'][1] == 0, "Devrait être en y=0\n"
    
    env.close()

def test_goal_reaching():
    """Test l'atteinte de l'objectif"""
    env = GridWorldEnv(size=3, num_traps=0)
    obs, info = env.reset()
    
    print("✅ Test 3: Atteinte de l'objectif")
    print(f"   Position initiale: {info['agent_pos']}")
    print(f"   Objectif: {info['goal_pos']}")
    
    # Aller directement à l'objectif (coins opposés dans grille 3x3)
    # Droite, Droite, Bas, Bas
    for action in [3, 3, 1, 1]:
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   Action: {info['action_name']}, Position: {info['agent_pos']}, Reward: {reward:.2f}")
        if terminated:
            print(f"   🎉 OBJECTIF ATTEINT ! Récompense finale: {reward}")
            break
    
    assert terminated, "L'épisode devrait être terminé"
    assert reward == 100.0, "La récompense devrait être de 100\n"
    env.close()

def test_trap_collision():
    """Test la collision avec un piège"""
    env = GridWorldEnv(size=5, num_traps=1)
    env.reset()
    
    # Forcer un piège à une position connue
    env.traps = [np.array([1, 0])]
    
    print("✅ Test 4: Collision avec piège")
    print(f"   Piège à la position: {env.traps[0]}")
    
    # Se déplacer vers le piège
    obs, reward, terminated, truncated, info = env.step(3)  # Droite
    
    print(f"   Position après mouvement: {info['agent_pos']}")
    print(f"   Récompense: {reward}")
    print(f"   Terminé: {terminated}\n")
    
    assert terminated, "L'épisode devrait être terminé"
    assert reward == -10.0, "La récompense devrait être de -10"
    env.close()

if __name__ == "__main__":
    print("="*60)
    print("🧪 TESTS DE L'ENVIRONNEMENT GRIDWORLD")
    print("="*60 + "\n")
    
    test_environment_creation()
    test_movements()
    test_goal_reaching()
    test_trap_collision()
    
    print("="*60)
    print("✅ TOUS LES TESTS SONT PASSÉS !")
    print("="*60)