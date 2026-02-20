"""
Détection des hallucinations dans les explications LLM
"""

import re
from typing import Dict, List
import numpy as np

class HallucinationDetector:
    """
    Classe pour détecter si le LLM invente des informations
    """
    
    def __init__(self):
        self.hallucination_patterns = [
            # Mots/phrases qui suggèrent une invention
            r"trésor",
            r"récompense cachée",
            r"bonus",
            r"secret",
            r"ennemi",
            r"monstre",
            r"pouvoir",
            r"item",
        ]
    
    def validate_explanation(self, explanation: str, context: Dict) -> Dict:
        """
        Valider qu'une explication ne contient pas d'hallucinations
        
        Args:
            explanation: Explication générée par le LLM
            context: Contexte réel de décision
            
        Returns:
            Dict avec les résultats de validation
        """
        issues = []
        
        # 1. Vérifier les inventions de concepts
        concept_issues = self._check_concept_hallucinations(explanation, context)
        issues.extend(concept_issues)
        
        # 2. Vérifier la cohérence des Q-values mentionnées
        qvalue_issues = self._check_qvalue_consistency(explanation, context)
        issues.extend(qvalue_issues)
        
        # 3. Vérifier les directions/positions mentionnées
        position_issues = self._check_position_consistency(explanation, context)
        issues.extend(position_issues)
        
        # 4. Vérifier l'action choisie
        action_issues = self._check_action_consistency(explanation, context)
        issues.extend(action_issues)
        
        # Score de fiabilité (0-100)
        reliability_score = max(0, 100 - len(issues) * 20)
        
        return {
            "is_valid": len(issues) == 0,
            "reliability_score": reliability_score,
            "issues": issues,
            "total_issues": len(issues)
        }
    
    def _check_concept_hallucinations(self, explanation: str, context: Dict) -> List[str]:
        """Vérifier si le LLM invente des concepts inexistants"""
        issues = []
        explanation_lower = explanation.lower()
        
        for pattern in self.hallucination_patterns:
            if re.search(pattern, explanation_lower):
                issues.append(f"Mention d'un concept non existant: '{pattern}'")
        
        return issues
    
    def _check_qvalue_consistency(self, explanation: str, context: Dict) -> List[str]:
        """Vérifier que les Q-values mentionnées correspondent"""
        issues = []
        q_values = context["q_values"]
        best_action = context["best_action"]["name"]
        
        # Vérifier que la meilleure action est correctement identifiée
        best_q = max(q_values.values())
        mentioned_best = None
        
        for action, qval in q_values.items():
            if qval == best_q and action.lower() in explanation.lower():
                mentioned_best = action
                break
        
        if mentioned_best and mentioned_best != best_action:
            issues.append(f"L'action identifiée comme meilleure ({mentioned_best}) "
                        f"ne correspond pas à la vraie meilleure action ({best_action})")
        
        # Vérifier les valeurs numériques mentionnées
        numbers_in_text = re.findall(r'[-+]?\d+\.?\d*', explanation)
        for num_str in numbers_in_text:
            num = float(num_str)
            # Vérifier si ce nombre est proche d'une Q-value réelle
            min_diff = min(abs(num - qval) for qval in q_values.values())
            if min_diff > 1.0:  # Tolérance de 1.0
                # Nombre mentionné ne correspond à aucune Q-value
                if abs(num) > 10:  # Seulement pour les grandes valeurs
                    issues.append(f"Q-value mentionnée ({num}) ne correspond "
                                f"à aucune Q-value réelle")
        
        return issues
    
    def _check_position_consistency(self, explanation: str, context: Dict) -> List[str]:
        """Vérifier la cohérence des positions/directions"""
        issues = []
        analysis = context["state_analysis"]
        explanation_lower = explanation.lower()
        
        # Vérifier les obstacles mentionnés
        obstacles = analysis["obstacles"]
        
        for direction, status in obstacles.items():
            direction_lower = direction.lower()
            
            if status == "mur" and direction_lower in explanation_lower:
                # OK, le mur est mentionné
                pass
            elif status == "piège" and direction_lower in explanation_lower:
                # Vérifier que "piège" est aussi mentionné
                if "piège" not in explanation_lower and "trap" not in explanation_lower:
                    issues.append(f"Direction '{direction}' mentionnée mais pas le piège")
            elif status == "libre" and direction_lower in explanation_lower:
                # Vérifier qu'aucun obstacle n'est inventé
                if "mur" in explanation_lower or "piège" in explanation_lower:
                    nearby = explanation_lower[max(0, explanation_lower.index(direction_lower)-50):
                                             min(len(explanation_lower), 
                                                 explanation_lower.index(direction_lower)+50)]
                    if "mur" in nearby or "piège" in nearby:
                        issues.append(f"Obstacle inventé dans la direction '{direction}'")
        
        return issues
    
    def _check_action_consistency(self, explanation: str, context: Dict) -> List[str]:
        """Vérifier que l'action mentionnée est correcte"""
        issues = []
        best_action = context["best_action"]["name"].lower()
        explanation_lower = explanation.lower()
        
        # L'action choisie doit être mentionnée
        if best_action not in explanation_lower:
            issues.append(f"L'action choisie '{best_action}' n'est pas mentionnée "
                        f"dans l'explication")
        
        return issues
    
    def batch_validate(self, explanations: List[tuple]) -> Dict:
        """
        Valider un ensemble d'explications
        
        Args:
            explanations: Liste de tuples (explanation, context)
            
        Returns:
            Statistiques globales de validation
        """
        results = []
        
        for explanation, context in explanations:
            validation = self.validate_explanation(explanation, context)
            results.append(validation)
        
        # Calculer les statistiques
        total = len(results)
        valid = sum(1 for r in results if r["is_valid"])
        avg_score = np.mean([r["reliability_score"] for r in results])
        all_issues = [issue for r in results for issue in r["issues"]]
        
        return {
            "total_explanations": total,
            "valid_explanations": valid,
            "invalid_explanations": total - valid,
            "validity_rate": (valid / total * 100) if total > 0 else 0,
            "average_reliability_score": avg_score,
            "all_issues": all_issues,
            "issue_types": self._categorize_issues(all_issues)
        }
    
    def _categorize_issues(self, issues: List[str]) -> Dict:
        """Catégoriser les types de problèmes détectés"""
        categories = {
            "concept_hallucinations": 0,
            "qvalue_inconsistencies": 0,
            "position_errors": 0,
            "action_errors": 0
        }
        
        for issue in issues:
            if "concept non existant" in issue:
                categories["concept_hallucinations"] += 1
            elif "Q-value" in issue:
                categories["qvalue_inconsistencies"] += 1
            elif "Direction" in issue or "Obstacle" in issue:
                categories["position_errors"] += 1
            elif "action" in issue:
                categories["action_errors"] += 1
        
        return categories