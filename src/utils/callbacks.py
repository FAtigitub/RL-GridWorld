"""
Callbacks personnalisés pour l'entraînement
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class CustomCallback(BaseCallback):
    """
    Callback pour logger des informations personnalisées
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Logger les infos de l'épisode terminé
        if self.locals.get("dones")[0]:
            info = self.locals["infos"][0]
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                
                # Calculer les statistiques
                mean_reward = np.mean(self.episode_rewards[-100:])
                mean_length = np.mean(self.episode_lengths[-100:])
                
                # Logger dans TensorBoard
                self.logger.record("custom/mean_reward_100", mean_reward)
                self.logger.record("custom/mean_length_100", mean_length)
                
        return True