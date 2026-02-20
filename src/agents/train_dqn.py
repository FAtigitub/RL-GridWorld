"""
Script d'entraînement de l'agent DQN sur Gridworld
"""

import os
import sys
from pathlib import Path

# Ajouter le dossier parent au path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

from src.environment.gridworld import GridWorldEnv
from src.environment.random_traps_wrapper import RandomTrapsWrapper
from src.utils.callbacks import CustomCallback

def create_env(size=5, num_traps=3, random_traps=False):
    """
    Créer et wrapper l'environnement
    
    Args:
        size: Taille de la grille
        num_traps: Nombre de pièges (ignoré si random_traps=True)
        random_traps: Si True, randomise le nombre de pièges (1-5) à chaque épisode
    """
    env = GridWorldEnv(size=size, num_traps=num_traps)
    
    # Wrapper pour randomiser les pièges si demandé
    if random_traps:
        env = RandomTrapsWrapper(env, min_traps=1, max_traps=5)
    
    env = Monitor(env)
    return env

def train_agent(
    total_timesteps=50000,
    size=5,
    num_traps=3,
    random_traps=False,
    learning_rate=1e-3,
    buffer_size=10000,
    batch_size=64,
    gamma=0.99
):
    """
    Entraîner l'agent DQN
    
    Args:
        total_timesteps: Nombre total de steps d'entraînement
        size: Taille de la grille
        num_traps: Nombre de pièges
        learning_rate: Taux d'apprentissage
        buffer_size: Taille du buffer de replay
        batch_size: Taille des batchs
        gamma: Facteur de discount
    """
    
    print("="*70)
    print("🎮 ENTRAÎNEMENT DE L'AGENT DQN SUR GRIDWORLD")
    print("="*70)
    
    # Créer les dossiers nécessaires
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("logs/tensorboard", exist_ok=True)
    
    # Créer l'environnement d'entraînement
    print("\n📦 Création de l'environnement...")
    env = create_env(size=size, num_traps=num_traps, random_traps=random_traps)
    
    # Créer l'environnement d'évaluation (aussi avec traps aléatoires si activé)
    eval_env = create_env(size=size, num_traps=num_traps, random_traps=random_traps)
    
    print(f"   ✅ Grille: {size}x{size}")
    print(f"   ✅ Agent: [0, 0] (fixe)")
    print(f"   ✅ Objectif: [{size-1}, {size-1}] (fixe)")
    if random_traps:
        print(f"   ✅ Pièges: 1-5 (randomisés à chaque épisode)")
    else:
        print(f"   ✅ Pièges: {num_traps} (fixe)")
    print(f"   ✅ Distance optimale: {(size-1)*2} steps")
    print(f"   ✅ Espace d'action: {env.action_space}")
    print(f"   ✅ Espace d'observation: {env.observation_space}")
    
    # Configuration de l'agent DQN
    print("\n🤖 Configuration de l'agent DQN...")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Buffer size: {buffer_size}")
    print(f"   Batch size: {batch_size}")
    print(f"   Gamma: {gamma}")
    
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        exploration_fraction=0.8,  # 80% du temps en exploration (AUGMENTÉ 0.7→0.8)
        exploration_initial_eps=1.0,  # Commence à 100% exploration
        exploration_final_eps=0.15,  # Epsilon final à 15% (AUGMENTÉ 0.05→0.15 pour contre-mémorisation)
        target_update_interval=250,  # Update target plus souvent (500→250)
        train_freq=4,
        gradient_steps=1,
        verbose=1,
        tensorboard_log="logs/tensorboard"
    )
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/",
        log_path="logs/",
        eval_freq=1000,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path="models/checkpoints/",
        name_prefix="dqn_gridworld"
    )
    
    # Entraînement
    print(f"\n🏋️ Début de l'entraînement ({total_timesteps} steps)...")
    print("   Appuyez sur Ctrl+C pour arrêter proprement\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            log_interval=100,
            progress_bar=True
        )
        
        # Sauvegarder le modèle final
        model.save("models/dqn_gridworld_final")
        print("\n✅ Modèle sauvegardé dans 'models/dqn_gridworld_final.zip'")
        
    except KeyboardInterrupt:
        print("\n⚠️  Entraînement interrompu par l'utilisateur")
        model.save("models/dqn_gridworld_interrupted")
        print("   Modèle sauvegardé dans 'models/dqn_gridworld_interrupted.zip'")
    
    # Fermer les environnements
    env.close()
    eval_env.close()
    
    print("\n" + "="*70)
    print("✅ ENTRAÎNEMENT TERMINÉ !")
    print("="*70)
    
    return model

if __name__ == "__main__":
    # Configuration AMÉLIORÉE v3: Objectif >90% succès
    # Résultats v2: 85% succès (17/20) avec 800k timesteps
    # Solution: Augmenter à 1.2M timesteps pour meilleure généralisation
    config = {
        "total_timesteps": 1200000,  # AUGMENTÉ: 800k→1.2M pour >90% succès
        "size": 5,
        "num_traps": 2,  # Randomise 1-3 pièges
        "random_traps": True,
        "learning_rate": 7e-4,
        "buffer_size": 100000,
        "batch_size": 128,
        "gamma": 0.97
    }
    
    # Entraîner
    model = train_agent(**config)
    
    print("\n📊 Pour visualiser l'entraînement avec TensorBoard:")
    print("   tensorboard --logdir=logs/tensorboard")