"""
Module pour générer des explications avec Llama en local via Ollama
AVANTAGES:
- ✅ Pas de quota / rate limiting
- ✅ Gratuit et illimité
- ✅ Privé (données ne quittent pas votre machine)
- ✅ Pas besoin de clé API

INSTALLATION:
1. Télécharger Ollama: https://ollama.com/download
2. Installer un modèle: ollama pull llama3.2:3b
3. pip install ollama
"""

import ollama
from typing import Dict

class RLExplainerOllama:
    """Explainer utilisant Llama via Ollama (local)"""
    
    def __init__(self, model_name='llama3.2:3b', temperature=0.3, verbose=False):
        """
        Args:
            model_name: Modèle Ollama (llama3.2:3b recommandé - rapide et performant)
            temperature: Créativité (0-1)
            verbose: Mode détaillé avec analyse de chaque action
        """
        self.model_name = model_name
        self.temperature = temperature
        self.verbose = verbose
        
        # Vérifier qu'Ollama est installé
        try:
            ollama.list()
        except Exception as e:
            raise RuntimeError(
                "❌ Ollama n'est pas installé ou ne fonctionne pas.\n"
                "📥 Téléchargez: https://ollama.com/download\n"
                "📦 Puis: ollama pull llama3.2:3b"
            ) from e
    
    def explain_decision(self, context: Dict, max_tokens=None) -> Dict:
        """Génère une explication (courte ou détaillée selon self.verbose)"""
        if max_tokens is None:
            max_tokens = 1500 if self.verbose else 800
        
        system_prompt = self._get_system_prompt_verbose() if self.verbose else self._get_system_prompt()
        user_prompt = self._create_prompt_verbose(context) if self.verbose else self._create_prompt(context)
        
        return self._call_ollama(system_prompt, user_prompt, max_tokens, result_key='explanation')
    
    def explain_with_comparison(self, context: Dict) -> Dict:
        """Génère une explication comparative entre toutes les actions"""
        ranking = context['action_ranking']
        best, worst = ranking[0], ranking[-1]
        
        system_prompt = self._get_system_prompt()
        user_prompt = (
            f"Classement des actions par Q-value:\n"
            f"  #1 (meilleure): {best['action']} → {best['q_value']:.2f}\n"
            f"  #4 (pire):      {worst['action']} → {worst['q_value']:.2f}\n\n"
            f"En 3 phrases complètes et concises:\n"
            f"1. Pourquoi '{best['action']}' est la meilleure action.\n"
            f"2. Pourquoi '{worst['action']}' est la pire action.\n"
            f"3. La différence de stratégie entre ces deux actions.\n"
            f"Termine avec un point final. Pas de listes ni de titres."
        )
        
        return self._call_ollama(system_prompt, user_prompt, max_tokens=512, result_key='comparison')
    
    def _call_ollama(self, system_prompt: str, user_prompt: str, max_tokens: int, result_key: str) -> Dict:
        """Appel à Ollama local"""
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                options={
                    'temperature': self.temperature,
                    'num_predict': max_tokens,  # Équivalent de max_tokens
                }
            )
            
            text = response['message']['content'].strip()
            
            # Calcul approximatif des tokens
            prompt_tokens = len(system_prompt.split()) + len(user_prompt.split())
            completion_tokens = len(text.split())
            
            return {
                result_key: text,
                'model': self.model_name,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens,
            }
            
        except Exception as e:
            return {
                result_key: f"❌ Erreur Ollama: {str(e)}",
                'error': True
            }
    
    def _get_system_prompt(self) -> str:
        """Prompt pour mode CONCIS (épisodes complets)"""
        return (
            "Tu es un expert en Reinforcement Learning (RL). "
            "Tu expliques les décisions d'un agent RL de façon COURTE, CLAIRE et COMPLÈTE.\n\n"
            "RÈGLES ABSOLUES:\n"
            "1. Réponds en 3-4 phrases complètes (50-80 mots).\n"
            "2. Base-toi sur les Q-values, positions et obstacles fournis.\n"
            "3. Mentionne la Q-value de l'action choisie ET pourquoi elle est supérieure.\n"
            "4. Explique brièvement pourquoi les autres actions sont moins bonnes.\n"
            "5. N'utilise PAS de titres, listes ni formatage markdown.\n"
            "6. Termine avec un point final."
        )
    
    def _get_system_prompt_verbose(self) -> str:
        """Prompt pour mode DÉTAILLÉ (analyse ponctuelle)"""
        return (
            "Tu es un expert en Reinforcement Learning (RL). "
            "Tu analyses les décisions d'un agent RL de manière DÉTAILLÉE et STRUCTURÉE.\n\n"
            "RÈGLES:\n"
            "1. Génère une explication structurée avec sections claires.\n"
            "2. Analyse CHAQUE action individuellement avec sa Q-value.\n"
            "3. Explique le contexte spatial (positions, obstacles, direction optimale).\n"
            "4. Base-toi UNIQUEMENT sur les données fournies.\n"
            "5. Utilise un format avec tirets (-) pour chaque action.\n"
            "6. Sois précis avec les valeurs numériques."
        )
    
    def _create_prompt(self, context: Dict) -> str:
        """Prompt concis"""
        state = context['state']
        q_values = context['q_values']
        best_action = context['best_action']
        analysis = context['state_analysis']
        
        q_lines = []
        for action, q_val in sorted(q_values.items(), key=lambda x: x[1], reverse=True):
            marker = '[CHOISI]' if action == best_action['name'] else '       '
            q_lines.append(f"  {marker} {action}: {q_val:.2f}")
        
        obs_lines = [f"  - {d}: {s}" for d, s in analysis['obstacles'].items()]
        
        return (
            f"Situation de l'agent RL (Gridworld 5x5):\n"
            f"- Position agent: {state['agent_position']}, Objectif: {state['goal_position']}\n"
            f"- Distance: {analysis['distance_to_goal']:.2f} cases, Direction: {analysis['direction_to_goal']}\n"
            f"- Obstacles:\n" + '\n'.join(obs_lines) + "\n"
            f"- Q-values:\n" + '\n'.join(q_lines) + "\n"
            f"- Action choisie: {best_action['name']} (Q-value: {best_action['q_value']:.2f})\n\n"
            f"En 3-4 phrases complètes (50-80 mots), explique POURQUOI l'agent a choisi "
            f"{best_action['name']} plutôt que les autres actions. "
            f"Mentionne les positions, la direction vers objectif, et pourquoi les autres actions "
            f"sont moins bonnes (obstacles, éloignement). Termine avec un point final."
        )
    
    def _create_prompt_verbose(self, context: Dict) -> str:
        """Prompt détaillé avec analyse de chaque action"""
        state = context['state']
        q_values = context['q_values']
        best_action = context['best_action']
        analysis = context['state_analysis']
        ranking = context['action_ranking']
        
        lines = [
            "Analyse la décision de l'agent dans cette situation GridWorld:",
            "",
            "CONTEXTE SPATIAL:",
            f"- Position agent: {state['agent_position']}",
            f"- Position objectif: {state['goal_position']}",
            f"- Distance: {analysis['distance_to_goal']:.2f} cases",
            f"- Direction optimale: {analysis['direction_to_goal']}",
            "",
            "OBSTACLES IMMÉDIATS:",
        ]
        
        for direction, status in analysis['obstacles'].items():
            lines.append(f"- {direction}: {status}")
        
        lines.append("")
        lines.append("Q-VALUES:")
        for action, q_val in sorted(q_values.items(), key=lambda x: x[1], reverse=True):
            marker = 'CHOISI' if action == best_action['name'] else '      '
            lines.append(f"  [{marker}] {action}: {q_val:.2f}")
        
        lines.append("")
        lines.append("GÉNÈRE une explication structurée avec ce FORMAT:")
        lines.append("")
        lines.append("CONTEXTE: [1 phrase de situation]")
        lines.append("")
        lines.append("ANALYSE DES ACTIONS:")
        for i, r in enumerate(ranking):
            marker = 'CHOISI' if i == 0 else '      '
            lines.append(f"- {r['action']} (Q={r['q_value']:.2f}) [{marker}]: [1-2 phrases expliquant cette Q-value]")
        lines.append("")
        lines.append("CONCLUSION: [1 phrase résumant le choix optimal]")
        
        return '\n'.join(lines)
