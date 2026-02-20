"""
Environnement Gridworld personnalisé compatible avec Gymnasium
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
from typing import Tuple, Optional, Dict

class GridWorldEnv(gym.Env):
    """
    Environnement Gridworld simple :
    - Grille NxN
    - Agent qui doit atteindre un objectif
    - Pièges à éviter
    - 4 actions possibles : Haut, Bas, Gauche, Droite
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, 
                 size: int = 5,
                 num_traps: int = 3,
                 render_mode: Optional[str] = None):
        """
        Initialisation de l'environnement
        
        Args:
            size: Taille de la grille (size x size)
            num_traps: Nombre de pièges
            render_mode: Mode de rendu ('human' ou 'rgb_array')
        """
        super().__init__()
        
        self.size = size
        self.num_traps = num_traps
        self.render_mode = render_mode
        
        # Définir les espaces d'observation et d'action
        # Observation = [agent_x, agent_y, goal_x, goal_y, trap1_x, trap1_y, ..., trap5_x, trap5_y]
        # Supporte jusqu'à 5 pièges (padding à -1.0 si moins)
        # Cela permet à l'agent de voir TOUS les pièges et de planifier!
        self.observation_space = spaces.Box(
            low=-1.0,  # -1 pour padding des pièges absents
            high=size-1, 
            shape=(14,),  # 2 (agent) + 2 (goal) + 5*2 (5 pièges max)
            dtype=np.float32
        )
        
        # Actions : 0=Haut, 1=Bas, 2=Gauche, 3=Droite
        self.action_space = spaces.Discrete(4)
        
        # Dictionnaire des actions pour lisibilité
        self.action_to_name = {
            0: "Haut",
            1: "Bas", 
            2: "Gauche",
            3: "Droite"
        }
        
        # Initialisation du rendu Pygame
        self.window = None
        self.clock = None
        self.cell_size = 100  # Pixels par cellule
        
        # Positions des éléments (seront initialisées dans reset)
        self.agent_pos = None
        self.goal_pos = None
        self.traps = []
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Réinitialiser l'environnement
        
        Returns:
            observation: Position initiale de l'agent
            info: Informations additionnelles
        """
        super().reset(seed=seed)
        
        # POSITIONS FIXES pour apprentissage cohérent
        # Agent commence toujours en bas à gauche [0, 0]
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        
        # Objectif toujours en haut à droite [size-1, size-1]
        # Pour grille 5x5 → [4, 4]
        self.goal_pos = np.array([self.size - 1, self.size - 1], dtype=np.int32)
        
        # Générer des pièges aléatoirement
        self.traps = []
        while len(self.traps) < self.num_traps:
            trap_pos = self.np_random.integers(0, self.size, size=2, dtype=np.int32)
            
            # Ne pas placer de piège sur l'agent [0,0] ou l'objectif [size-1,size-1]
            if (not np.array_equal(trap_pos, self.agent_pos) and 
                not np.array_equal(trap_pos, self.goal_pos) and
                not any(np.array_equal(trap_pos, t) for t in self.traps)):
                self.traps.append(trap_pos)
        
        # Initialiser le compteur de steps
        self.current_step = 0
        # Distance optimale de [0,0] à [size-1,size-1] = (size-1)*2 = 8 pour 5x5
        # Max_steps = optimal + marge pour éviter pièges
        self.max_steps = (self.size - 1) * 2 + 10  # 8 + 10 = 18 steps max pour 5x5
        
        # Historique des positions pour détecter les boucles
        self.position_history = [self.agent_pos.copy()]
        self.previous_distance = float(np.linalg.norm(self.agent_pos - self.goal_pos))
        
        # Compteur de visites par position (pour détecter cycles longs)
        self.position_visit_count = {}
        pos_tuple = tuple(self.agent_pos)
        self.position_visit_count[pos_tuple] = 1
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Exécuter une action
        
        Args:
            action: Action à exécuter (0-3)
            
        Returns:
            observation: Nouvelle position
            reward: Récompense obtenue
            terminated: Episode terminé (objectif atteint ou piège)
            truncated: Episode tronqué (max steps atteint)
            info: Informations additionnelles
        """
        # Sauvegarder l'ancienne position
        old_pos = self.agent_pos.copy()
        old_distance = float(np.linalg.norm(self.agent_pos - self.goal_pos))
        
        # Effectuer le mouvement
        if action == 0:  # Haut
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1:  # Bas
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)
        elif action == 2:  # Gauche
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3:  # Droite
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)
        
        # Ajouter position à l'historique
        self.position_history.append(self.agent_pos.copy())
        
        # Compter les visites de cette position
        pos_tuple = tuple(self.agent_pos)
        if pos_tuple not in self.position_visit_count:
            self.position_visit_count[pos_tuple] = 0
        self.position_visit_count[pos_tuple] += 1
        
        # Incrémenter le compteur
        self.current_step += 1
        
        # Calculer la récompense avec détection de boucles
        new_distance = float(np.linalg.norm(self.agent_pos - self.goal_pos))
        reward, terminated = self._calculate_reward(old_pos, old_distance, new_distance)
        
        # Vérifier si on dépasse le max de steps
        truncated = self.current_step >= self.max_steps
        
        observation = self._get_obs()
        info = self._get_info()
        info['old_pos'] = old_pos
        info['action_name'] = self.action_to_name[action]
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self, old_pos: np.ndarray, old_distance: float, new_distance: float) -> Tuple[float, bool]:
        """
        Calculer la récompense et si l'épisode est terminé
        Avec détection de boucles et shaping avancé
        
        Args:
            old_pos: Position précédente
            old_distance: Distance précédente à l'objectif
            new_distance: Nouvelle distance à l'objectif
        
        Returns:
            reward: Récompense
            terminated: Si l'épisode est terminé
        """
        # Vérifier si on a atteint l'objectif
        if np.array_equal(self.agent_pos, self.goal_pos):
            return 100.0, True  # Grande récompense !
        
        # Vérifier si on est tombé dans un piège
        for trap in self.traps:
            if np.array_equal(self.agent_pos, trap):
                return -10.0, True  # Pénalité et fin
        
        # 🔥 PÉNALITÉ DE PROXIMITÉ AUX PIÈGES: Force l'agent à considérer leur position
        trap_proximity_penalty = 0.0
        for trap in self.traps:
            distance_to_trap = np.linalg.norm(self.agent_pos - trap)
            if distance_to_trap <= 1.0:  # Adjacent au piège (distance Manhattan = 1)
                trap_proximity_penalty -= 2.0  # FORTE pénalité pour être adjacent
            elif distance_to_trap <= 1.5:  # Proche du piège (diagonale)
                trap_proximity_penalty -= 0.8  # Pénalité moyenne
        
        # DÉTECTION DE BOUCLES AVANCÉE: Pénaliser fortement les cycles
        loop_penalty = 0.0
        
        # 1. Détection de boucles courtes (oscillations immédiates)
        if len(self.position_history) >= 3:
            # Vérifier si position actuelle = position il y a 2 steps (boucle de 2)
            if np.array_equal(self.agent_pos, self.position_history[-3]):
                loop_penalty -= 2.0  # FORTE pénalité pour oscillation
        
        # 2. Détection de cycles plus longs (4-6 steps)
        if len(self.position_history) >= 5:
            if np.array_equal(self.agent_pos, self.position_history[-5]):
                loop_penalty -= 1.5  # Pénalité pour cycle de 4
        
        if len(self.position_history) >= 7:
            if np.array_equal(self.agent_pos, self.position_history[-7]):
                loop_penalty -= 1.0  # Pénalité pour cycle de 6
        
        # 3. Pénaliser visites multiples de la même position
        pos_tuple = tuple(self.agent_pos)
        visit_count = self.position_visit_count.get(pos_tuple, 0)
        if visit_count > 1:
            # Pénalité exponentielle: plus on visite, plus c'est pénalisé
            loop_penalty -= (visit_count - 1) * 0.5
        
        # REWARD SHAPING: Récompenser mouvements vers l'objectif
        distance_improvement = old_distance - new_distance
        
        if distance_improvement > 0:
            # Se rapproche de l'objectif: Récompense positive
            progress_reward = 0.5 * distance_improvement
        elif distance_improvement < 0:
            # S'éloigne de l'objectif: DOUBLE PÉNALITÉ
            progress_reward = 2.5 * distance_improvement  # Augmenté de 2.0 → 2.5
        else:
            # Même distance (mouvement inutile ou mur)
            progress_reward = -0.3  # Augmenté de -0.2 → -0.3
        
        # Pénalité de step (encourage chemins courts) - AUGMENTÉE
        step_penalty = -0.15  # Augmenté de -0.1 → -0.15
        
        # Récompense totale (INCLUT MAINTENANT LA PÉNALITÉ DE PROXIMITÉ AUX PIÈGES)
        reward = progress_reward + step_penalty + loop_penalty + trap_proximity_penalty
        
        return reward, False
    
    def _get_obs(self) -> np.ndarray:
        """
        Obtenir l'observation actuelle
        Retourne: [agent_x, agent_y, goal_x, goal_y, trap1_x, trap1_y, ..., trap5_x, trap5_y]
        L'agent voit TOUS les pièges (jusqu'à 5 max) pour mieux planifier!
        Si moins de 5 pièges, les positions manquantes sont remplies avec -1.0
        """
        # Position de l'agent
        obs = self.agent_pos.astype(np.float32).copy()
        
        # Position de l'objectif
        goal_pos = self.goal_pos.astype(np.float32)
        
        # Positions de TOUS les pièges (max 5)
        # Padding avec -1.0 pour les emplacements vides
        max_traps = 5
        traps_obs = []
        for i in range(max_traps):
            if i < len(self.traps):
                traps_obs.extend(self.traps[i].astype(np.float32).tolist())
            else:
                traps_obs.extend([-1.0, -1.0])  # Padding pour pièges absents
        
        traps_array = np.array(traps_obs, dtype=np.float32)
        
        # Concaténer: [agent_x, agent_y, goal_x, goal_y, trap1_x, trap1_y, ..., trap5_x, trap5_y]
        full_obs = np.concatenate([obs, goal_pos, traps_array])
        
        return full_obs
    
    def _get_info(self) -> Dict:
        """Obtenir les informations additionnelles"""
        return {
            "agent_pos": self.agent_pos.tolist(),
            "goal_pos": self.goal_pos.tolist(),
            "traps": [trap.tolist() for trap in self.traps],
            "distance_to_goal": float(np.linalg.norm(self.agent_pos - self.goal_pos)),
            "step": self.current_step
        }
    
    def render(self):
        """Afficher l'environnement"""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            return self._render_frame()
    
    def _render_frame(self):
        """Créer le frame de rendu avec Pygame"""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.size * self.cell_size, self.size * self.cell_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.size * self.cell_size, self.size * self.cell_size))
        canvas.fill((255, 255, 255))  # Fond blanc
        
        # Dessiner la grille
        for i in range(self.size + 1):
            # Lignes verticales
            pygame.draw.line(
                canvas,
                (200, 200, 200),
                (i * self.cell_size, 0),
                (i * self.cell_size, self.size * self.cell_size),
                2
            )
            # Lignes horizontales
            pygame.draw.line(
                canvas,
                (200, 200, 200),
                (0, i * self.cell_size),
                (self.size * self.cell_size, i * self.cell_size),
                2
            )
        
        # Dessiner les pièges (rouge)
        for trap in self.traps:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    trap[0] * self.cell_size + 5,
                    trap[1] * self.cell_size + 5,
                    self.cell_size - 10,
                    self.cell_size - 10
                )
            )
        
        # Dessiner l'objectif (vert)
        pygame.draw.circle(
            canvas,
            (0, 255, 0),
            (
                self.goal_pos[0] * self.cell_size + self.cell_size // 2,
                self.goal_pos[1] * self.cell_size + self.cell_size // 2
            ),
            self.cell_size // 3
        )
        
        # Dessiner l'agent (bleu)
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (
                self.agent_pos[0] * self.cell_size + self.cell_size // 2,
                self.agent_pos[1] * self.cell_size + self.cell_size // 2
            ),
            self.cell_size // 3
        )
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def close(self):
        """Fermer l'environnement"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()