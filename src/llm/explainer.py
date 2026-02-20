import os
import re
import time
from typing import Dict
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


class RLExplainer:
    """Explainer using Google Gemini - compact prompts to minimize quota usage."""

    def __init__(self, model_name="gemini-2.5-flash", temperature=0.3):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = model_name
        self.temperature = temperature

    def explain_decision(self, context, max_tokens=1024):
        """Short explanation for one agent step. Uses a compact prompt to save quota."""
        prompt = self._build_prompt(context)
        return self._call_gemini(prompt, max_tokens, result_key="explanation")

    def explain_with_comparison(self, context):
        """Two-sentence comparative summary of best vs worst action."""
        ranking = context["action_ranking"]
        best = ranking[0]
        worst = ranking[-1]
        prompt = (
            "Agent RL Gridworld. Q-values: "
            + best["action"] + "=" + str(round(best["q_value"], 2)) + " (meilleure), "
            + worst["action"] + "=" + str(round(worst["q_value"], 2)) + " (pire). "
            "En 2 phrases courtes: pourquoi " + best["action"] + " est le meilleur choix "
            "et pourquoi " + worst["action"] + " est le pire. Point final obligatoire."
        )
        return self._call_gemini(prompt, max_tokens=300, result_key="comparison")

    def explain_full_episode(self, episode_steps, max_tokens=800):
        """Analyze entire episode trajectory in ONE call to save quota.
        
        Args:
            episode_steps: List of step dictionaries with context and action taken
            max_tokens: Max tokens for response (default 800 for comprehensive analysis)
        """
        # Build ultra-compact trajectory summary
        trajectory = []
        for step in episode_steps:
            ctx = step["context"]
            pos = ctx["state"]["agent_position"]
            action = step["action"]
            q_val = round(step["q_value"], 1)
            trajectory.append(f"[{pos[0]},{pos[1]}]→{action}(Q={q_val})")
        
        traj_str = " → ".join(trajectory)
        
        start_pos = episode_steps[0]["context"]["state"]["agent_position"]
        goal_pos = episode_steps[0]["context"]["state"]["goal_position"]
        total_steps = len(episode_steps)
        success = episode_steps[-1].get("terminated", False) and episode_steps[-1].get("reward", 0) > 0
        
        prompt = (
            f"GridWorld RL: Agent {start_pos}→{goal_pos}, {total_steps} étapes, "
            f"{'SUCCÈS' if success else 'ÉCHEC'}. "
            f"Trajectoire: {traj_str}. "
            f"Explique en 1 paragraphe (5-7 phrases max): stratégie globale de l'agent, "
            f"pourquoi ces actions, qualité du chemin. Point final obligatoire."
        )
        
        return self._call_gemini(prompt, max_tokens, result_key="episode_explanation")

    def explain_first_step_only(self, step_data):
        """Generate rule-based explanation for first step WITHOUT API call (quota-free).
        
        Args:
            step_data: Dictionary with step context, action, q_value, etc.
        """
        ctx = step_data["context"]
        state = ctx["state"]
        best = ctx["best_action"]
        analysis = ctx["state_analysis"]
        ranking = ctx["action_ranking"]
        
        pos = state["agent_position"]
        goal = state["goal_position"]
        action = best["name"]
        q_val = round(best["q_value"], 2)
        dist = round(analysis["distance_to_goal"], 2)
        direction = analysis["direction_to_goal"]
        
        # Find second best action for comparison
        second_best = ranking[1] if len(ranking) > 1 else None
        worst = ranking[-1]
        
        # Build explanation from rules
        explanation = (
            f"L'agent situé en {pos} a choisi l'action {action} (Q-value: {q_val}) "
            f"pour se rapprocher de l'objectif {goal} situé à {dist} unités de distance. "
            f"La direction optimale est '{direction}', et {action} s'aligne avec cette stratégie. "
        )
        
        # Add obstacle context if relevant
        obstacles = analysis["obstacles"]
        blocked = [d for d, s in obstacles.items() if s == "mur"]
        if blocked:
            explanation += f"Les directions {', '.join(blocked)} sont bloquées par des murs. "
        
        # Add trap information (critical for safety)
        traps = state.get("traps", [])
        if traps:
            trap_dist = round(analysis["nearest_trap_distance"], 2)
            trap_positions = ", ".join([f"[{t[0]},{t[1]}]" for t in traps])
            explanation += (
                f"Attention: piège(s) présent(s) en {trap_positions} "
                f"(distance minimale: {trap_dist}). "
            )
        
        # Add comparison with other actions
        if second_best:
            diff = round(q_val - second_best["q_value"], 2)
            explanation += (
                f"{action} surpasse {second_best['action']} (Q={round(second_best['q_value'], 2)}) "
                f"avec un avantage de {diff} points. "
            )
        
        # Worst action context
        explanation += (
            f"L'action {worst['action']} (Q={round(worst['q_value'], 2)}) "
            f"est la moins favorable car elle éloigne de l'objectif."
        )
        
        return {
            "explanation": explanation,
            "model": "rule-based (no API)",
            "prompt_tokens": 0,
            "completion_tokens": len(explanation.split()),
            "total_tokens": len(explanation.split()),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, context):
        """Build a compact single-line prompt to minimise token usage."""
        state    = context["state"]
        q_values = context["q_values"]
        best     = context["best_action"]
        analysis = context["state_analysis"]

        obs_str = ", ".join(
            d + "=" + s for d, s in analysis["obstacles"].items()
        )
        q_str = ", ".join(
            a + ":" + str(round(v, 2))
            for a, v in sorted(q_values.items(), key=lambda x: x[1], reverse=True)
        )

        return (
            "Agent RL Gridworld: "
            "pos=" + str(state["agent_position"])
            + " objectif=" + str(state["goal_position"])
            + " dist=" + str(round(analysis["distance_to_goal"], 2))
            + " dir=" + analysis["direction_to_goal"]
            + " obstacles=[" + obs_str + "]"
            + " Q-values=[" + q_str + "]"
            + " CHOIX=" + best["name"] + "(" + str(round(best["q_value"], 2)) + ")."
            + " Explique en 2-3 phrases POURQUOI ce choix est optimal."
            + " Mentionne la Q-value et les obstacles. Termine par un point."
        )

    def _call_gemini(self, prompt, max_tokens, result_key, max_retries=3):
        """Call Gemini API with automatic retry on rate-limit (429)."""
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=max_tokens,
                    )
                )

                if not response or not response.text:
                    return {result_key: "Pas de reponse generee", "error": True}

                text = response.text.strip()

                finish_reason = None
                try:
                    finish_reason = response.candidates[0].finish_reason.name
                except Exception:
                    pass

                if finish_reason == "MAX_TOKENS" and not text.endswith((".", "!", "?", '"', "'")):
                    text += " [...]"
                    print(f"DEBUG: finish_reason={finish_reason}, tokens={ct}/{max_tokens}")
                elif finish_reason == "MAX_TOKENS":
                    print(f"DEBUG: finish_reason={finish_reason} BUT text seems complete. tokens={ct}/{max_tokens}")

                pt = len(prompt.split())
                ct = len(text.split())
                return {
                    result_key: text,
                    "model": self.model_name,
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "total_tokens": pt + ct,
                    "finish_reason": finish_reason,
                }

            except Exception as e:
                err = str(e)
                if ("429" in err or "RESOURCE_EXHAUSTED" in err) and attempt < max_retries - 1:
                    wait = 65
                    m = re.search(r"retry in ([0-9.]+)s", err)
                    if m:
                        wait = float(m.group(1)) + 2
                    print("Rate limit. Attente " + str(int(wait)) + "s (tentative "
                          + str(attempt + 1) + "/" + str(max_retries) + ")...")
                    time.sleep(wait)
                    continue
                return {result_key: "Erreur: " + err, "error": True}

        return {result_key: "Echec apres plusieurs tentatives (rate limit)", "error": True}
