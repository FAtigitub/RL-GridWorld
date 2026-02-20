"""
Wrapper pour randomiser le nombre de pièges à chaque épisode
Permet à l'agent d'apprendre à gérer différents nombres de pièges (1-5)
"""

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Optional

class RandomTrapsWrapper(gym.Wrapper):
    """
    Wrapper qui randomise le nombre de pièges à chaque reset
    L'agent apprendra ainsi à gérer 1 à 5 pièges
    """
    
    def __init__(self, env, min_traps: int = 1, max_traps: int = 3):
        """
        Args:
            env: Environnement GridWorld de base
            min_traps: Nombre minimum de pièges (défaut: 1)
            max_traps: Nombre maximum de pièges (défaut: 3 - RÉDUIT de 5 pour meilleur apprentissage)
        """
        super().__init__(env)
        self.min_traps = min_traps
        self.max_traps = max_traps
        
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Reset avec nombre aléatoire de pièges
        """
        # Randomiser le nombre de pièges
        self.env.num_traps = self.env.np_random.integers(
            self.min_traps, 
            self.max_traps + 1
        )
        
        # Reset normal
        return self.env.reset(**kwargs)
