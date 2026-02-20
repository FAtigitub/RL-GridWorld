# GridWorld RL - Explainable AI with DQN

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Accuracy](https://img.shields.io/badge/Accuracy-85%25-green)
![Framework](https://img.shields.io/badge/Framework-Stable--Baselines3-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Description du Projet

Système d'apprentissage par renforcement (RL) avec agent DQN naviguant dans un environnement GridWorld 5×5. L'agent apprend à éviter des pièges positionnés aléatoirement pour atteindre un objectif fixe. Le projet inclut des explications générées par LLM (Gemini 2.5 Flash) et une interface interactive Streamlit.

### Caractéristiques Principales

- ✅ Agent DQN entraîné avec reward shaping avancé
- ✅ Pénalités de proximité aux obstacles pour évitement anticipé
- ✅ Interface web interactive Streamlit
- ✅ Visualisations Plotly (trajectoires, Q-values, récompenses)
- ✅ Explications LLM avec détection automatique d'hallucinations
- ✅ Support dynamique de 1 à 5 pièges

## 📊 Performances du Modèle

### Métriques Globales

- **Accuracy finale:** 85.0% (17/20 succès avec 3 pièges)
- **Optimalité:** 100% des succès utilisent le chemin optimal (8 pas)
- **Entraînement:** 1.2M timesteps en 18-20 minutes sur CPU
- **Vitesse:** ~3000 frames/seconde

### Performance par Difficulté

| Nombre de Pièges | Taux de Succès |
|------------------|----------------|
| 1 piège          | ~95%           |
| 2 pièges         | ~90%           |
| 3 pièges         | 85%            |
| 4 pièges         | ~75%           |
| 5 pièges         | ~65%           |

### Métriques d'Entraînement (TensorBoard)

- Récompense moyenne: -30 (début) → +60-70 (convergence)
- Longueur épisodes: 11 pas (début) → 7-8 pas (optimal)
- Loss réseau: 2.5-5.0 (stable, pas de divergence)

## 🛠️ Stack Technique

### Apprentissage par Renforcement
- **Algorithme:** Deep Q-Network (DQN)
- **Framework RL:** Stable-Baselines3 2.0.0
- **Environnement:** Gymnasium 0.28.1
- **Deep Learning:** PyTorch 2.0.1

### Interface et Visualisation
- **Interface Web:** Streamlit 1.28.0
- **Graphiques Interactifs:** Plotly 5.17.0
- **Monitoring:** TensorBoard

### Explications IA
- **LLM:** Google Gemini 2.5 Flash
- **Validation:** Système de détection d'hallucinations custom

### Utilitaires
- **Calculs:** NumPy 1.24.3
- **Langage:** Python 3.10

## 📦 Installation

### Prérequis

- Python 3.10 ou supérieur
- pip (gestionnaire de paquets Python)
- Compte Google Cloud (pour API Gemini)

### Installation Rapide

```bash
# 1. Cloner le repository
git clone https://github.com/VOTRE_USERNAME/gridworld-rl-explainable.git
cd gridworld-rl-explainable

# 2. Créer environnement virtuel
python -m venv venv

# 3. Activer l'environnement
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 4. Installer les dépendances
pip install -r requirements.txt

# 5. Configurer l'API Gemini
# Créer fichier .env à la racine
echo "GOOGLE_API_KEY=votre_cle_api_ici" > .env
```

### Obtenir une clé API Gemini

1. Aller sur [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Créer une nouvelle clé API
3. Copier la clé dans le fichier `.env`

## 🎮 Utilisation

### 1. Entraîner le Modèle

```bash
python src/agents/train_dqn.py
```

**Paramètres modifiables dans le fichier:**
- `total_timesteps`: 1 200 000 (défaut)
- `learning_rate`: 0.0007
- `num_traps`: 1-3 pièges randomisés

**Durée:** ~18-20 minutes sur CPU moderne

### 2. Tester le Modèle

```bash
# Test standard (20 épisodes, 3 pièges)
python src/agents/test_agent.py --episodes 20

# Test personnalisé
python src/agents/test_agent.py --episodes 30 --render
```

**Options disponibles:**
- `--episodes`: Nombre d'épisodes à tester
- `--render`: Activer le rendu visuel
- `--analyze`: Afficher analyse détaillée des Q-values

### 3. Lancer l'Application Streamlit

```bash
streamlit run app/streamlit_app.py
```

**Accès:** http://localhost:8501

**Fonctionnalités:**
- Slider pour choisir nombre de pièges (1-5)
- Génération d'épisodes aléatoires
- Navigation dans historique d'épisodes
- 3 visualisations interactives Plotly
- Explications LLM pour premier pas

### 4. Visualiser l'Entraînement (TensorBoard)

```bash
tensorboard --logdir=logs/tensorboard
```

**Accès:** http://localhost:6006

**Métriques disponibles:**
- Récompense moyenne par épisode
- Longueur moyenne des épisodes
- Loss du réseau Q
- Taux d'exploration (epsilon)

## 📁 Structure du Projet

```
gridworld_llm_project/
├── src/
│   ├── environment/
│   │   ├── gridworld.py              # Environnement GridWorld custom
│   │   └── random_traps_wrapper.py   # Wrapper randomisation pièges
│   └── agents/
│       ├── train_dqn.py              # Script entraînement DQN
│       ├── test_agent.py             # Script test et évaluation
│       └── q_value_extractor.py      # Extraction Q-values du réseau
├── app/
│   └── streamlit_app.py              # Application web interactive
├── models/
│   └── dqn_gridworld_final.zip       # Modèle entraîné (85% accuracy)
├── logs/
│   └── tensorboard/                  # Logs d'entraînement
├── RAPPORT_TECHNIQUE.md              # Documentation technique complète
├── GUIDE_GITHUB.md                   # Guide push vers GitHub
├── requirements.txt                  # Dépendances Python
├── .gitignore                        # Fichiers exclus de Git
└── README.md                         # Ce fichier
```

## 🧠 Détails Techniques

### Environnement GridWorld

- **Taille:** Grille 5×5
- **Départ:** Toujours (0,0)
- **Objectif:** Toujours (4,4)
- **Pièges:** 1-5 placés aléatoirement
- **Actions:** 4 directions (Haut, Bas, Gauche, Droite)
- **Observation:** Vecteur 14D [agent_pos, goal_pos, 5×traps_pos]

### Système de Récompenses

- Atteindre objectif: **+100**
- Tomber dans piège: **-10**
- Rapprochement objectif: **+0.5 × distance**
- Éloignement objectif: **-2.5 × distance**
- Proximité piège (≤1.0): **-2.0**
- Proximité piège (1.0-1.5): **-0.8**
- Pénalité par pas: **-0.15**

### Configuration DQN

- **Architecture:** MLP (2 couches × 64 neurones, ReLU)
- **Learning rate:** 0.0007
- **Replay buffer:** 100 000 transitions
- **Batch size:** 128
- **Gamma:** 0.97
- **Exploration:** 80% du training (epsilon 1.0 → 0.15)

## 📈 Résultats Détaillés

### Statistiques sur 20 Épisodes (3 pièges)

- **Succès:** 17/20 (85%)
- **Récompense moyenne:** 81.36 ± 39.30
- **Pas moyen:** 7.3 ± 1.8
- **Meilleure récompense:** 101.28
- **Distribution actions:** 51.4% Droite, 48.6% Bas

### Analyse Qualitative

**Comportements observés:**
- 100% des succès atteignent l'optimal (8 pas)
- Trajectoires variées selon configuration pièges
- Évitement anticipé visible (pénalités proximité)
- Adaptation dynamique aux obstacles

**Types d'échecs:**
- Immédiats (1 pas): 2 directions optimales bloquées
- Précoces (4-5 pas): Engagement couloir bloqué
- Tardifs (6-7 pas): Piège sur case quasi-obligatoire

## 📸 Captures d'Écran

### Application Streamlit
![Streamlit App](docs/images/streamlit_app.png)

### Visualisation Trajectoire
![Trajectoire](docs/images/trajectory.png)

### Q-values Analysis
![Q-values](docs/images/qvalues.png)

### TensorBoard Training
![TensorBoard](docs/images/tensorboard.png)

*Note: Créer dossier `docs/images/` et ajouter captures*

## 🔍 Documentation Complète

Voir [RAPPORT_TECHNIQUE.md](RAPPORT_TECHNIQUE.md) pour:
- Architecture détaillée du système
- Métriques d'entraînement TensorBoard
- Analyse approfondie des performances
- Fonctionnalités complètes application Streamlit
- Système d'explications LLM
- Limites et perspectives d'amélioration

## 🚀 Améliorations Futures

### Court Terme
- [ ] Augmenter architecture réseau (plus de couches)
- [ ] Tester algorithmes alternatifs (PPO, SAC)
- [ ] Ajouter curriculum learning (1→5 pièges progressif)
- [ ] Export trajectoires en vidéo

### Moyen Terme
- [ ] Grilles plus grandes (10×10, 20×20)
- [ ] Obstacles dynamiques (pièges mobiles)
- [ ] Multi-agents collaboratifs
- [ ] Mécanismes d'attention

### Long Terme
- [ ] Extension 3D
- [ ] Deployment API REST
- [ ] Mobile app (React Native + API)
- [ ] Benchmarking autres algorithmes RL

## 📝 License

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour détails.

## 👤 Auteur

**Votre Nom**
- GitHub: [@VotreUsername](https://github.com/VotreUsername)
- LinkedIn: [Votre Profil](https://linkedin.com/in/votre-profil)
- Email: votre.email@example.com

## 🙏 Remerciements

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - Framework RL
- [OpenAI Gymnasium](https://gymnasium.farama.org/) - Interface environnements
- [Streamlit](https://streamlit.io/) - Framework web apps
- [Plotly](https://plotly.com/) - Visualisations interactives
- [Google Gemini](https://ai.google.dev/) - API LLM

## 📊 Citations

Si vous utilisez ce projet dans vos recherches, veuillez citer:

```bibtex
@software{gridworld_rl_explainable_2026,
  author = {Votre Nom},
  title = {GridWorld RL - Explainable AI with DQN},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/VOTRE_USERNAME/gridworld-rl-explainable}
}
```

## 🐛 Signaler un Bug

Problème trouvé? [Créer une issue](https://github.com/VOTRE_USERNAME/gridworld-rl-explainable/issues)

## 🤝 Contribuer

Les contributions sont les bienvenues! Voir [CONTRIBUTING.md](CONTRIBUTING.md) pour guidelines.

1. Fork le projet
2. Créer branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers branche (`git push origin feature/AmazingFeature`)
5. Ouvrir Pull Request

---

**⭐ Si ce projet vous aide, n'hésitez pas à lui donner une étoile!**
