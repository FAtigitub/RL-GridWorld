"""
Extraction des Q-values et informations de décision de l'agent
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
from stable_baselines3 import DQN

class QValueExtractor:
    """
    Classe pour extraire les informations de décision de l'agent DQN
    """
    
    def __init__(self, model: DQN, env):
        """
        Args:
            model: Modèle DQN entraîné
            env: Environnement Gridworld
        """
        self.model = model
        self.env = env
        self.action_names = env.action_to_name
        
    def extract_decision_context(self, obs: np.ndarray) -> Dict:
        """
        Extraire tout le contexte de décision pour un état donné
        
        Args:
            obs: Observation actuelle (position de l'agent)
            
        Returns:
            Dictionnaire contenant:
                - state: État actuel en format lisible
                - q_values: Q-values pour chaque action
                - best_action: Meilleure action selon l'agent
                - action_ranking: Actions classées par Q-value
                - state_analysis: Analyse de l'état (obstacles, distance, etc.)
        """
        # Obtenir les Q-values
        q_values = self._get_q_values(obs)
        
        # Identifier la meilleure action
        best_action_idx = int(np.argmax(q_values))
        best_action_name = self.action_names[best_action_idx]
        
        # Classer les actions par Q-value
        action_ranking = self._rank_actions(q_values)
        
        # Analyser l'état
        state_analysis = self._analyze_state(obs)
        
        # Extraire goal et traps de l'observation pour le retour
        goal_pos_from_obs = obs[2:4].tolist()
        traps_from_obs = []
        for i in range(5):
            trap_x = obs[4 + i*2]
            trap_y = obs[4 + i*2 + 1]
            if trap_x >= 0:  # Valide (pas de padding)
                traps_from_obs.append([float(trap_x), float(trap_y)])
        
        return {
            "state": {
                "agent_position": obs[0:2].tolist(),
                "goal_position": goal_pos_from_obs,
                "traps": traps_from_obs
            },
            "q_values": {
                self.action_names[i]: float(q_values[i]) 
                for i in range(len(q_values))
            },
            "best_action": {
                "index": best_action_idx,
                "name": best_action_name,
                "q_value": float(q_values[best_action_idx])
            },
            "action_ranking": action_ranking,
            "state_analysis": state_analysis
        }
    
    def _get_q_values(self, obs: np.ndarray) -> np.ndarray:
        """
        Obtenir les Q-values pour toutes les actions
        
        Args:
            obs: Observation
            
        Returns:
            Array des Q-values [q0, q1, q2, q3]
        """
        # Convertir l'observation en tensor
        obs_tensor = torch.as_tensor(obs).float().unsqueeze(0)
        
        # Obtenir les Q-values du réseau
        with torch.no_grad():
            q_values = self.model.q_net(obs_tensor)
        
        # Convertir en numpy
        return q_values.cpu().numpy()[0]
    
    def _rank_actions(self, q_values: np.ndarray) -> List[Dict]:
        """
        Classer les actions par Q-value décroissante
        
        Args:
            q_values: Q-values de chaque action
            
        Returns:
            Liste des actions classées avec leurs infos
        """
        ranking = []
        indices = np.argsort(q_values)[::-1]  # Tri décroissant
        
        for rank, idx in enumerate(indices):
            ranking.append({
                "rank": rank + 1,
                "action": self.action_names[idx],
                "q_value": float(q_values[idx])
            })
        
        return ranking
    
    def _analyze_state(self, obs: np.ndarray) -> Dict:
        """
        Analyser l'état actuel (obstacles, distances, etc.)
        
        Args:
            obs: Observation (position agent)
            
        Returns:
            Dictionnaire d'analyse
        """
        agent_pos = obs[0:2].astype(int)
        goal_pos = obs[2:4].astype(int)  # Extraire goal depuis observation au lieu de self.env
        
        # Extraire les pièges de l'observation (obs[4:14] contient 5 pièges max)
        traps = []
        for i in range(5):
            trap_x = obs[4 + i*2]
            trap_y = obs[4 + i*2 + 1]
            if trap_x >= 0:  # Pas de padding (valeurs >= 0 sont valides)
                traps.append(np.array([trap_x, trap_y], dtype=int))
        
        # Vérifier les obstacles autour
        obstacles = self._check_surroundings(agent_pos, goal_pos, traps)
        
        # Calculer la distance à l'objectif
        distance_to_goal = np.linalg.norm(agent_pos - goal_pos)
        
        # Direction vers l'objectif
        direction = self._get_direction_to_goal(agent_pos, goal_pos)
        
        # Vérifier si des pièges sont proches
        nearest_trap_distance = self._get_nearest_trap_distance(agent_pos, traps)
        
        return {
            "obstacles": obstacles,
            "distance_to_goal": float(distance_to_goal),
            "direction_to_goal": direction,
            "nearest_trap_distance": nearest_trap_distance,
            "is_near_edge": self._is_near_edge(agent_pos)
        }
    
    def _check_surroundings(self, pos: np.ndarray, goal_pos: np.ndarray, traps: list) -> Dict:
        """Vérifier les cases autour de l'agent"""
        size = self.env.size
        
        surroundings = {
            "Haut": "libre",
            "Bas": "libre",
            "Gauche": "libre",
            "Droite": "libre"
        }
        
        # Vérifier les murs
        if pos[1] == 0:
            surroundings["Haut"] = "mur"
        if pos[1] == size - 1:
            surroundings["Bas"] = "mur"
        if pos[0] == 0:
            surroundings["Gauche"] = "mur"
        if pos[0] == size - 1:
            surroundings["Droite"] = "mur"
        
        # Vérifier les pièges
        for trap in traps:
            if np.array_equal(trap, pos + [0, -1]):  # Haut
                surroundings["Haut"] = "piège"
            elif np.array_equal(trap, pos + [0, 1]):  # Bas
                surroundings["Bas"] = "piège"
            elif np.array_equal(trap, pos + [-1, 0]):  # Gauche
                surroundings["Gauche"] = "piège"
            elif np.array_equal(trap, pos + [1, 0]):  # Droite
                surroundings["Droite"] = "piège"
        
        # Vérifier l'objectif
        if np.array_equal(goal_pos, pos + [0, -1]):
            surroundings["Haut"] = "objectif"
        elif np.array_equal(goal_pos, pos + [0, 1]):
            surroundings["Bas"] = "objectif"
        elif np.array_equal(goal_pos, pos + [-1, 0]):
            surroundings["Gauche"] = "objectif"
        elif np.array_equal(goal_pos, pos + [1, 0]):
            surroundings["Droite"] = "objectif"
        
        return surroundings
    
    def _get_direction_to_goal(self, agent_pos: np.ndarray, goal_pos: np.ndarray) -> str:
        """Déterminer la direction générale vers l'objectif"""
        diff = goal_pos - agent_pos
        
        directions = []
        if diff[0] > 0:
            directions.append("droite")
        elif diff[0] < 0:
            directions.append("gauche")
        
        if diff[1] > 0:
            directions.append("bas")
        elif diff[1] < 0:
            directions.append("haut")
        
        return " et ".join(directions) if directions else "sur l'objectif"
    
    def _get_nearest_trap_distance(self, pos: np.ndarray, traps: list) -> float:
        """Calculer la distance au piège le plus proche"""
        if len(traps) == 0:
            return float('inf')
        
        distances = [np.linalg.norm(pos - trap) for trap in traps]
        return float(min(distances))
    
    def _is_near_edge(self, pos: np.ndarray) -> Dict:
        """Vérifier si l'agent est près d'un bord"""
        size = self.env.size
        return {
            "top": pos[1] == 0,
            "bottom": pos[1] == size - 1,
            "left": pos[0] == 0,
            "right": pos[0] == size - 1
        }

def format_decision_for_human(context: Dict) -> str:
    """
    Formater le contexte de décision en texte lisible
    
    Args:
        context: Contexte retourné par extract_decision_context
        
    Returns:
        String formatée pour affichage
    """
    state = context["state"]
    q_vals = context["q_values"]
    best = context["best_action"]
    analysis = context["state_analysis"]
    
    output = []
    output.append("="*60)
    output.append("🧠 CONTEXTE DE DÉCISION DE L'AGENT")
    output.append("="*60)
    
    output.append("\n📍 ÉTAT ACTUEL:")
    output.append(f"   Position agent: {state['agent_position']}")
    output.append(f"   Position objectif: {state['goal_position']}")
    output.append(f"   Distance à l'objectif: {analysis['distance_to_goal']:.2f}")
    output.append(f"   Direction vers objectif: {analysis['direction_to_goal']}")
    
    output.append("\n🚧 ENVIRONNEMENT:")
    for direction, status in analysis['obstacles'].items():
        emoji = "🚫" if status == "mur" else "🔥" if status == "piège" else "🌟" if status == "objectif" else "✅"
        output.append(f"   {direction}: {emoji} {status}")
    
    output.append("\n📊 Q-VALUES (Espérance de récompense):")
    for action, q_val in sorted(q_vals.items(), key=lambda x: x[1], reverse=True):
        is_best = "⭐" if action == best["name"] else "  "
        bar = "█" * int(max(0, min(20, (q_val + 10) / 1)))
        output.append(f"   {is_best} {action:10s}: {q_val:7.2f} {bar}")
    
    output.append(f"\n✨ ACTION CHOISIE: {best['name']} (Q-value: {best['q_value']:.2f})")
    
    output.append("\n📈 CLASSEMENT DES ACTIONS:")
    for item in context['action_ranking']:
        output.append(f"   {item['rank']}. {item['action']:10s} → {item['q_value']:7.2f}")
    
    output.append("="*60)
    
    return "\n".join(output)