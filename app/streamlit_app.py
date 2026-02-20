"""
Application Streamlit pour visualiser les explications RL+LLM
Animation Vidéo des Épisodes Complets
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from src.environment.gridworld import GridWorldEnv
from src.agents.q_value_extractor import QValueExtractor
from src.agents.q_value_extractor import QValueExtractor
import src.llm.explainer as explainer_module
import importlib
importlib.reload(explainer_module)
from src.llm.explainer import RLExplainer
from src.llm.hallucination_detector import HallucinationDetector
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go
import plotly.express as px
import time
import json
import os
from google import genai
from google.genai import types

# Configuration de la page
st.set_page_config(
    page_title="GridWorld RL Explainer - Episode Viewer",
    page_icon="📊",
    layout="wide"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .explanation-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .step-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Charger le modèle DQN (cache)"""
    return DQN.load("models/dqn_gridworld_final.zip")

def create_environment(size, num_traps):
    """
    Créer l'environnement (pas de cache pour permettre changements dynamiques)
    Le nouveau modèle a été entraîné avec 1-5 pièges aléatoires
    """
    return GridWorldEnv(size=size, num_traps=num_traps)

def run_episode(model, env, max_steps=50):
    """Exécuter un épisode complet et collecter toutes les données"""
    obs, info = env.reset()
    
    episode_data = []
    done = False
    step = 0
    total_reward = 0
    
    while not done and step < max_steps:
        # Prédire l'action
        action, _states = model.predict(obs, deterministic=True)
        action = int(action)
        
        # Sauvegarder état avant l'action
        position_before = env.agent_pos.copy()
        
        # Exécuter l'action
        obs_next, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # Collecter les données du step
        action_names = ["Haut", "Bas", "Gauche", "Droite"]
        step_data = {
            "step": step + 1,
            "position_before": position_before.tolist(),
            "position_after": env.agent_pos.copy().tolist(),
            "action": action_names[action],
            "action_idx": action,
            "reward": float(reward),
            "total_reward": float(total_reward),
            "terminated": terminated,
            "truncated": truncated,
        }
        
        episode_data.append(step_data)
        obs = obs_next
        step += 1
    
    return episode_data

def render_gridworld(env_state, agent_pos, figsize=(6, 6)):
    """Créer une visualisation matplotlib du gridworld"""
    fig, ax = plt.subplots(figsize=figsize)
    
    size = env_state['size']
    goal_pos = env_state['goal_pos']
    traps = env_state['traps']
    
    # Dessiner la grille
    for i in range(size + 1):
        ax.plot([0, size], [i, i], 'k-', linewidth=0.5)
        ax.plot([i, i], [0, size], 'k-', linewidth=0.5)
    
    # Dessiner les pièges
    for trap in traps:
        rect = patches.Rectangle(
            (trap[0], size - trap[1] - 1), 1, 1,
            linewidth=0, facecolor='red', alpha=0.6
        )
        ax.add_patch(rect)
        ax.text(trap[0] + 0.5, size - trap[1] - 0.5, '🔥',
                ha='center', va='center', fontsize=24)
    
    # Dessiner l'objectif
    ax.plot(goal_pos[0] + 0.5, size - goal_pos[1] - 0.5, 
            'g*', markersize=30, markeredgecolor='darkgreen', 
            markeredgewidth=2)
    
    # Dessiner l'agent
    ax.plot(agent_pos[0] + 0.5, size - agent_pos[1] - 0.5, 
            'bo', markersize=25, markeredgecolor='darkblue',
            markeredgewidth=2)
    
    # Si contexte fourni, montrer la direction choisie
    if context:
        best_action = context['best_action']['name']
        dx, dy = 0, 0
        if best_action == "Droite":
            dx = 0.3
        elif best_action == "Gauche":
            dx = -0.3
        elif best_action == "Bas":
            dy = -0.3
        elif best_action == "Haut":
            dy = 0.3
        
        if dx != 0 or dy != 0:
            ax.arrow(agent_pos[0] + 0.5, size - agent_pos[1] - 0.5,
                    dx, dy, head_width=0.2, head_length=0.15, 
                    fc='blue', ec='darkblue', linewidth=2)
    
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Gridworld Environment', fontsize=14, fontweight='bold')
    
    return fig

def visualize_trajectory(episode_data, env_state, current_step=None):
    """Visualiser la trajectoire complète avec Plotly"""
    size = env_state['size']
    goal_pos = env_state['goal_pos']
    traps = env_state['traps']
    
    # Extraire les positions
    positions = [step['position_before'] for step in episode_data]
    if episode_data:
        positions.append(episode_data[-1]['position_after'])
    
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]
    
    fig = go.Figure()
    
    # Grille de fond
    for i in range(size + 1):
        fig.add_shape(type="line", x0=0, y0=i, x1=size, y1=i,
                     line=dict(color="lightgray", width=1))
        fig.add_shape(type="line", x0=i, y0=0, x1=i, y1=size,
                     line=dict(color="lightgray", width=1))
    
    # Objectif
    fig.add_trace(go.Scatter(
        x=[goal_pos[0] + 0.5], y=[goal_pos[1] + 0.5],
        mode='markers',
        marker=dict(symbol='star', size=25, color='gold', 
                   line=dict(color='orange', width=2)),
        name='Objectif',
        showlegend=True
    ))
    
    # Pièges
    if traps:
        trap_x = [trap[0] + 0.5 for trap in traps]
        trap_y = [trap[1] + 0.5 for trap in traps]
        fig.add_trace(go.Scatter(
            x=trap_x, y=trap_y,
            mode='markers',
            marker=dict(symbol='x', size=20, color='red',
                       line=dict(color='darkred', width=2)),
            name='Pièges',
            showlegend=True
        ))
    
    # Trajectoire
    fig.add_trace(go.Scatter(
        x=[xc + 0.5 for xc in x_coords],
        y=[yc + 0.5 for yc in y_coords],
        mode='lines+markers',
        line=dict(color='#667eea', width=3),
        marker=dict(size=12, color='#764ba2', line=dict(color='white', width=2)),
        name='Trajectoire',
        showlegend=True,
        text=[f"Step {i+1}" for i in range(len(x_coords))],
        hovertemplate='<b>%{text}</b><br>Position: (%{x}, %{y})<extra></extra>'
    ))
    
    # Position actuelle (si spécifiée)
    if current_step is not None and current_step <= len(positions):
        curr_pos = positions[current_step]
        fig.add_trace(go.Scatter(
            x=[curr_pos[0] + 0.5], y=[curr_pos[1] + 0.5],
            mode='markers',
            marker=dict(symbol='circle', size=18, color='cyan',
                       line=dict(color='blue', width=3)),
            name='Position Actuelle',
            showlegend=True
        ))
    
    fig.update_layout(
        title='Trajectoire de l\'Épisode',
        xaxis=dict(range=[0, size], dtick=1, showgrid=False),
        yaxis=dict(range=[0, size], dtick=1, showgrid=False),
        height=500,
        showlegend=True,
        hovermode='closest'
    )
    
    return fig

def plot_rewards_evolution(episode_data):
    """Graphique de l'évolution des récompenses"""
    steps = [s['step'] for s in episode_data]
    rewards = [s['reward'] for s in episode_data]
    cumulative = [s['total_reward'] for s in episode_data]
    
    fig = go.Figure()
    
    # Récompenses par step
    fig.add_trace(go.Bar(
        x=steps,
        y=rewards,
        name='Récompense par Step',
        marker=dict(color='lightblue'),
        yaxis='y'
    ))
    
    # Récompense cumulative
    fig.add_trace(go.Scatter(
        x=steps,
        y=cumulative,
        name='Récompense Cumulative',
        mode='lines+markers',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Évolution des Récompenses',
        xaxis=dict(title='Step'),
        yaxis=dict(title='Récompense par Step'),
        yaxis2=dict(title='Récompense Cumulative', overlaying='y', side='right'),
        height=400,
        hovermode='x unified'
    )
    
    return fig

def plot_action_distribution(episode_data):
    """Graphique de la distribution des actions"""
    actions = [s['action'] for s in episode_data]
    action_counts = {a: actions.count(a) for a in set(actions)}
    
    fig = go.Figure(data=[go.Pie(
        labels=list(action_counts.keys()),
        values=list(action_counts.values()),
        hole=0.3,
        marker=dict(colors=['#667eea', '#764ba2', '#f59e0b', '#10b981'])
    )])
    
    fig.update_layout(
        title='Distribution des Actions',
        height=400
    )
    
    return fig

def plot_qvalues(q_values, best_action_name):
    """Créer un graphique des Q-values"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    actions = list(q_values.keys())
    values = list(q_values.values())
    colors = ['#2196F3' if action == best_action_name else '#90CAF9' 
              for action in actions]
    
    bars = ax.barh(actions, values, color=colors, edgecolor='darkblue', linewidth=1.5)
    
    # Ajouter les valeurs sur les barres
    for i, (action, value) in enumerate(zip(actions, values)):
        ax.text(value, i, f' {value:.2f}', 
                va='center', fontweight='bold')
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Q-Value (Espérance de récompense)', fontsize=12, fontweight='bold')
    ax.set_title('Q-Values pour chaque action', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">GridWorld RL Explainer - Episode Viewer</h1>', 
                unsafe_allow_html=True)
    st.markdown("**Visualisation Complète d'Épisodes avec Explications et Validation**")
    
    # Sidebar - Configuration
    st.sidebar.header("Configuration")
    
    # Bouton pour recharger le modèle
    st.sidebar.markdown("### 🤖 Modèle RL")
    if st.sidebar.button("🔄 Recharger Modèle", help="Recharge le modèle depuis models/dqn_gridworld_final.zip"):
        st.cache_resource.clear()
        st.rerun()
    st.sidebar.caption("Modèle: dqn_gridworld_final.zip")
    st.sidebar.caption("Positions fixes: Agent [0,0] → Objectif [4,4]")
    st.sidebar.caption("Pièges: 1-5 aléatoires")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎮 Environnement")
    st.sidebar.info("🎯 **Positions:**\n- Agent: [0, 0] (fixe)\n- Objectif: [4, 4] (fixe)\n- Distance optimale: 8 steps")
    
    grid_size = st.sidebar.slider("Taille de la grille", 3, 10, 5)
    num_traps = st.sidebar.slider("Nombre de pièges", 1, 5, 1)
    max_steps = st.sidebar.slider("Steps maximum", 10, 100, 50)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### LLM Configuration")
    llm_model = st.sidebar.selectbox(
        "Modèle LLM",
        ["gemini-2.5-flash"]
    )
    temperature = st.sidebar.slider("Température LLM", 0.0, 1.0, 0.3, 0.1)
    generate_explanations = st.sidebar.checkbox("Générer explications LLM", value=False)
    
    if st.sidebar.button("Test Connexion Gemini"):
        try:
            genai.Client(api_key=os.getenv("GEMINI_API_KEY")).models.generate_content(
                model=llm_model, contents="Test", config=types.GenerateContentConfig(max_output_tokens=5)
            )
            st.sidebar.success("✅ Connexion OK")
        except Exception as e:
            st.sidebar.error(f"❌ Erreur: {str(e)}")
    
    # Charger les modèles
    with st.spinner("Chargement du modèle RL..."):
        model = load_model()
        env = create_environment(grid_size, num_traps)
        extractor = QValueExtractor(model, env)
    
    # Info sur le modèle
    st.info(f"""
    🤖 **Modèle chargé:** dqn_gridworld_final.zip  
    🎯 **Configuration:** Grille {grid_size}x{grid_size}, {num_traps} piège(s) | Agent [0,0] → Objectif [{grid_size-1},{grid_size-1}]  
    📏 **Distance optimale:** {(grid_size-1)*2} steps
    """)
    
    if generate_explanations:
        explainer = RLExplainer(model_name=llm_model, temperature=temperature)
        detector = HallucinationDetector()
    
    # ==================== SECTION 1: LANCER ÉPISODE ====================
    st.markdown("## Contrôle d'Épisode")
    
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button("Lancer Nouvel Épisode", type="primary", use_container_width=True):
            with st.spinner("⏳ Exécution de l'épisode..."):
                # Exécuter l'épisode complet
                episode_data = run_episode(model, env, max_steps)
                st.session_state.episode_data = episode_data
                st.session_state.current_step = 0
                st.session_state.env_state = {
                    'size': env.size,
                    'goal_pos': env.goal_pos.tolist(),
                    'traps': [t.tolist() for t in env.traps]
                }
                st.success(f"Épisode terminé: {len(episode_data)} steps exécutés")
                st.rerun()
    
    with col_btn2:
        if st.button("Réinitialiser", use_container_width=True):
            st.session_state.episode_data = None
            st.session_state.current_step = None
            st.rerun()
    
    # ==================== SECTION 2: VISUALISATION ÉPISODE ====================
    if 'episode_data' in st.session_state and st.session_state.episode_data:
        episode_data = st.session_state.episode_data
        env_state = st.session_state.env_state
        
        st.markdown("---")
        st.markdown("## Statistiques de l'Épisode")
        
        # Métriques
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        total_steps = len(episode_data)
        total_reward = episode_data[-1]['total_reward']
        avg_reward = total_reward / total_steps
        success = episode_data[-1]['terminated'] and total_reward > 50
        
        with col_m1:
            st.metric("Total Steps", total_steps)
        
        with col_m2:
            st.metric("Récompense Totale", f"{total_reward:.2f}")
        
        with col_m3:
            st.metric("Récompense Moyenne", f"{avg_reward:.2f}")
        
        with col_m4:
            st.metric("Statut", "Succès" if success else "Échec")
        
        # Graphiques de statistiques
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            fig_rewards = plot_rewards_evolution(episode_data)
            st.plotly_chart(fig_rewards, use_container_width=True)
        
        with col_g2:
            fig_actions = plot_action_distribution(episode_data)
            st.plotly_chart(fig_actions, use_container_width=True)
        
        # ==================== SECTION 3: ANIMATION VIDÉO DES STEPS ====================
        st.markdown("---")
        st.markdown("## Animation des Mouvements")
        
        # Contrôles de navigation
        col_nav1, col_nav2, col_nav3 = st.columns([1, 3, 1])
        
        current_step = st.session_state.get('current_step', 0)
        
        with col_nav1:
            if st.button("<< Début", use_container_width=True):
                st.session_state.current_step = 0
                st.rerun()
        
        with col_nav2:
            # Slider pour navigation
            new_step = st.slider(
                "Step", 
                0, 
                total_steps - 1, 
                current_step,
                key="step_slider"
            )
            if new_step != current_step:
                st.session_state.current_step = new_step
                st.rerun()
        
        with col_nav3:
            if st.button("Fin >>", use_container_width=True):
                st.session_state.current_step = total_steps - 1
                st.rerun()
        
        # Navigation précédent/suivant
        col_p1, col_p2, col_p3 = st.columns([1, 2, 1])
        
        with col_p1:
            if st.button("< Précédent", disabled=(current_step == 0), use_container_width=True):
                st.session_state.current_step = max(0, current_step - 1)
                st.rerun()
        
        with col_p2:
            st.markdown(f"<h3 style='text-align:center;'>Step {current_step + 1} / {total_steps}</h3>", 
                       unsafe_allow_html=True)
        
        with col_p3:
            if st.button("Suivant >", disabled=(current_step >= total_steps - 1), use_container_width=True):
                st.session_state.current_step = min(total_steps - 1, current_step + 1)
                st.rerun()
        
        # Afficher le step actuel avec plus de détails
        step_data = episode_data[current_step]
        
        # Reconstruire l'environnement pour obtenir plus de contexte
        temp_env = GridWorldEnv(size=env_state['size'], num_traps=len(env_state['traps']))
        temp_env.goal_pos = np.array(env_state['goal_pos'])
        temp_env.traps = [np.array(t) for t in env_state['traps']]
        temp_env.agent_pos = np.array(step_data['position_before'])
        obs = temp_env._get_obs()
        context = extractor.extract_decision_context(obs)
        
        # Calculer distances et directions
        pos = np.array(step_data['position_before'])
        goal = np.array(env_state['goal_pos'])
        dist_to_goal = np.linalg.norm(pos - goal)
        direction_to_goal = goal - pos
        
        # Déterminer la direction du mouvement
        move_vector = np.array(step_data['position_after']) - np.array(step_data['position_before'])
        move_dir = ""
        if move_vector[0] == 1: move_dir = "↓ Bas"
        elif move_vector[0] == -1: move_dir = "↑ Haut"
        elif move_vector[1] == 1: move_dir = "→ Droite"
        elif move_vector[1] == -1: move_dir = "← Gauche"
        
        # Affichage principal du step
        col_info1, col_info2 = st.columns([2, 1])
        
        with col_info1:
            st.markdown(f"""
            <div class='step-card' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;'>
                <h2 style='color: white; margin: 0;'>📍 Step {step_data['step']}</h2>
                <h3 style='color: #f0f0f0; margin-top: 0.5rem;'>{step_data['action']} {move_dir}</h3>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;'>
                    <div>
                        <p style='margin: 0.3rem 0;'><b>Position:</b> {step_data['position_before']} → {step_data['position_after']}</p>
                        <p style='margin: 0.3rem 0;'><b>Distance but:</b> {dist_to_goal:.2f} cellules</p>
                    </div>
                    <div>
                        <p style='margin: 0.3rem 0;'><b>Récompense:</b> {step_data['reward']:.2f}</p>
                        <p style='margin: 0.3rem 0;'><b>Total cumulé:</b> {step_data['total_reward']:.2f}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_info2:
            # Statistiques du mouvement
            reward_type = "🎯 Objectif!" if step_data['reward'] > 50 else ("💀 Piège" if step_data['reward'] < -50 else "➡️ Mouvement")
            st.metric("Type", reward_type)
            st.metric("Q-Value", f"{context['best_action']['q_value']:.3f}")
        
        # Section détails Q-values avec expander interactif
        with st.expander("📊 Détails des Q-Values et Analyse", expanded=False):
            st.markdown("### Q-Values de toutes les actions")
            
            q_data = []
            for action_name, q_val in context['q_values'].items():
                is_chosen = (action_name == step_data['action'])
                q_data.append({
                    "Action": action_name,
                    "Q-Value": f"{q_val:.4f}",
                    "Choisi": "✅" if is_chosen else ""
                })
            
            df_q = pd.DataFrame(q_data)
            st.dataframe(df_q, use_container_width=True, hide_index=True)
            
            # Graphique des Q-values
            fig_q = go.Figure(data=[
                go.Bar(
                    x=list(context['q_values'].keys()),
                    y=list(context['q_values'].values()),
                    marker_color=['#667eea' if k == step_data['action'] else '#cccccc' 
                                  for k in context['q_values'].keys()],
                    text=[f"{v:.3f}" for v in context['q_values'].values()],
                    textposition='outside'
                )
            ])
            fig_q.update_layout(
                title="Comparaison des Q-Values",
                xaxis_title="Action",
                yaxis_title="Q-Value",
                height=300
            )
            st.plotly_chart(fig_q, use_container_width=True)
            
            # Analyse de l'état
            st.markdown("### Analyse de l'environnement")
            analysis = context['state_analysis']
            
            col_a1, col_a2, col_a3 = st.columns(3)
            with col_a1:
                st.metric("Distance Objectif", f"{analysis['distance_to_goal']:.2f}")
            with col_a2:
                st.metric("Distance Piège", f"{analysis['nearest_trap_distance']:.2f}" if analysis['nearest_trap_distance'] < 100 else "Aucun")
            with col_a3:
                obstacles = [d for d, s in analysis['obstacles'].items() if s == "mur"]
                st.metric("Murs Bloqués", len(obstacles))
            
            if obstacles:
                st.info(f"🚫 Directions bloquées: {', '.join(obstacles)}")
        
        # Visualisation de la trajectoire
        fig_traj = visualize_trajectory(episode_data, env_state, current_step)
        st.plotly_chart(fig_traj, use_container_width=True)
        
        # ==================== SECTION 4: EXPLICATION ET VALIDATION ====================
        # UNIQUEMENT POUR STEP 1 (comme test_llm_integration.py)
        if generate_explanations and current_step == 0:
            st.markdown("---")
            st.markdown("## 🤖 Explication LLM - Step 1 Uniquement")
            st.info("💡 Les explications LLM sont générées uniquement pour le premier step pour économiser le quota API")
            
            gen_gemini = st.button(f"🤖 Gemini 2.5 Flash — Step 1", type="primary")

            if gen_gemini:
                with st.spinner("⏳ Génération de l'explication pour Step 1..."):
                    # Le contexte a déjà été extrait plus haut
                    # Utiliser step_data[0] pour Step 1
                    step_1_data = episode_data[0]

                    explanation_result = None
                    used_fallback = False

                    try:
                        # Utiliser explain_first_step_only comme dans test_llm_integration.py
                        step_info = {
                            "step": 1,
                            "position_before": step_1_data["position_before"],
                            "action": step_1_data["action"],
                            "q_value": context["best_action"]["q_value"],
                            "context": context
                        }
                        explanation_result = explainer.explain_first_step_only(step_info)
                        if explanation_result.get("error"):
                            raise RuntimeError(explanation_result["explanation"])
                    except Exception as e:
                        err_msg = str(e)
                        if "getaddrinfo" in err_msg or "11001" in err_msg or "network" in err_msg.lower():
                            st.warning(
                                "⚠️ Pas de connexion réseau vers l'API Gemini.\n"
                                "Basculement automatique vers l'explication par règles."
                            )
                        else:
                            st.warning(f"⚠️ Erreur Gemini: {err_msg}\nBasculement vers les règles.")
                        step_info = {
                            "step": 1,
                            "position_before": step_1_data["position_before"],
                            "action": step_1_data["action"],
                            "q_value": context["best_action"]["q_value"],
                            "context": context
                        }
                        explanation_result = explainer.explain_first_step_only(step_info)
                        used_fallback = True

                    if explanation_result:
                        expl_text = explanation_result.get("explanation", "")
                        model_label = "Règles (hors-ligne)" if used_fallback else "Gemini 2.5 Flash"
                        tokens_label = explanation_result.get("total_tokens", 0)

                        st.markdown(f"""
                        <div class='explanation-box'>
                            <h4>💬 Explication du Premier Mouvement (Step 1):</h4>
                            <p style="font-size:1.1rem; line-height:1.8; color: #2c3e50;">{expl_text}</p>
                            <hr style="margin:1rem 0; border: none; border-top: 2px solid #667eea;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <small style="color: #666;">
                                    <b>🤖 Modèle:</b> {model_label}
                                </small>
                                <small style="color: #666;">
                                    <b>📊 Tokens:</b> {tokens_label}
                                </small>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Validation hallucinations (comme test_hallucination_detection.py)
                        st.markdown("### 🔍 Validation Anti-Hallucination")
                        
                        validation = detector.validate_explanation(expl_text, context)

                        col_v1, col_v2, col_v3 = st.columns(3)
                        with col_v1:
                            status_icon = "✅" if validation['is_valid'] else "❌"
                            st.metric("Statut", f"{status_icon} {'Valide' if validation['is_valid'] else 'Invalide'}")
                        with col_v2:
                            score = validation['reliability_score']
                            score_color = "🟢" if score >= 80 else ("🟡" if score >= 60 else "🔴")
                            st.metric("Fiabilité", f"{score_color} {score}/100")
                        with col_v3:
                            issues_count = validation['total_issues']
                            st.metric("Problèmes", f"{'⚠️ ' if issues_count > 0 else '✅ '}{issues_count}")

                        if validation['issues']:
                            with st.expander("⚠️ Détails des problèmes détectés", expanded=True):
                                for i, issue in enumerate(validation['issues'], 1):
                                    st.warning(f"**{i}.** {issue}")
                        else:
                            st.success("✅ Aucune hallucination détectée! L'explication est fiable.")
        
        # ==================== SECTION 5: VUE D'ENSEMBLE ====================
        st.markdown("---")
        st.markdown("## 📋 Vue d'Ensemble de l'Épisode")
        
        with st.expander("📊 Détails Complets de Tous les Steps", expanded=False):
            # Créer un tableau structuré
            overview_data = []
            for step in episode_data:
                move_vector = np.array(step['position_after']) - np.array(step['position_before'])
                direction = ""
                if move_vector[0] == 1: direction = "↓"
                elif move_vector[0] == -1: direction = "↑"
                elif move_vector[1] == 1: direction = "→"
                elif move_vector[1] == -1: direction = "←"
                
                overview_data.append({
                    "Step": step['step'],
                    "Action": f"{step['action']} {direction}",
                    "De": str(step['position_before']),
                    "Vers": str(step['position_after']),
                    "Récompense": f"{step['reward']:.2f}",
                    "Total": f"{step['total_reward']:.2f}"
                })
            
            df_overview = pd.DataFrame(overview_data)
            st.dataframe(df_overview, use_container_width=True, hide_index=True)
    
    else:
        st.info("Cliquez sur 'Lancer Nouvel Épisode' pour commencer")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#666;">
        <p><b>RL Episode Viewer avec Explainability</b></p>
        <p>Visualisation vidéo des décisions de l'agent RL</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()