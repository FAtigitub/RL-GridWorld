# GUIDE - PUSH CODE VERS GITHUB

## Étape 1 : Créer un dépôt sur GitHub

1. Aller sur https://github.com
2. Cliquer sur "New repository" (bouton vert)
3. Remplir les informations :
   - Repository name : `gridworld-rl-explainable`
   - Description : `GridWorld RL with DQN and LLM explanations (85% accuracy)`
   - Visibilité : Public ou Private
   - **NE PAS** cocher "Initialize with README" (déjà existant)
4. Cliquer "Create repository"

## Étape 2 : Configurer Git localement (première fois uniquement)

```bash
# Configurer nom et email
git config --global user.name "Votre Nom"
git config --global user.email "votre.email@example.com"
```

## Étape 3 : Initialiser le dépôt local

```bash
# Se positionner dans le dossier du projet
cd "c:\Users\HP\Desktop\ML-DL projects\RL\gridworld_llm_project"

# Initialiser Git
git init
```

## Étape 4 : Créer fichier .gitignore

Créer un fichier `.gitignore` avec le contenu suivant pour éviter de pousser les fichiers inutiles :

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Modèles entraînés (fichiers volumineux)
models/*.zip
models/*.pt
models/*.pth

# Logs TensorBoard (volumineux)
logs/tensorboard/*/events.*

# Environnement virtuel
venv/
.env

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Données temporaires
*.log
*.tmp
```

## Étape 5 : Ajouter les fichiers au staging

```bash
# Ajouter tous les fichiers (respecte .gitignore)
git add .

# Vérifier les fichiers ajoutés
git status
```

## Étape 6 : Créer le premier commit

```bash
git commit -m "Initial commit: GridWorld RL with DQN (85% accuracy) + Streamlit app + LLM explanations"
```

## Étape 7 : Lier au dépôt GitHub

```bash
# Remplacer VOTRE_USERNAME par votre nom d'utilisateur GitHub
git remote add origin https://github.com/VOTRE_USERNAME/gridworld-rl-explainable.git

# Vérifier le lien
git remote -v
```

## Étape 8 : Pousser le code

```bash
# Pousser vers la branche main
git push -u origin main

# Si erreur "branch main doesn't exist", utiliser master
git branch -M main
git push -u origin main
```

## Étape 9 : Authentification GitHub

Lors du push, GitHub demandera l'authentification :

### Option A : Personal Access Token (recommandé)
1. Aller sur GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Cliquer "Generate new token (classic)"
3. Cocher : `repo` (Full control of private repositories)
4. Générer et copier le token
5. Utiliser ce token comme mot de passe lors du push

### Option B : GitHub CLI
```bash
# Installer GitHub CLI
winget install GitHub.cli

# Authentifier
gh auth login

# Puis faire le push normalement
```

## Étape 10 : Vérifier sur GitHub

1. Aller sur https://github.com/VOTRE_USERNAME/gridworld-rl-explainable
2. Vérifier que les fichiers sont présents
3. Vérifier que le README.md s'affiche correctement

## COMMANDES FUTURES (après initial push)

### Ajouter des modifications

```bash
# Voir les changements
git status

# Ajouter fichiers modifiés
git add .

# Commit avec message descriptif
git commit -m "Description des changements"

# Pousser vers GitHub
git push
```

### Créer une branche pour nouvelle fonctionnalité

```bash
# Créer et basculer sur nouvelle branche
git checkout -b feature/nouvelle-fonctionnalite

# Développer et commit
git add .
git commit -m "Ajout nouvelle fonctionnalité"

# Pousser la branche
git push -u origin feature/nouvelle-fonctionnalite
```

### Récupérer les changements distants

```bash
# Télécharger et fusionner
git pull origin main
```

## STRUCTURE RECOMMANDÉE DU README.md

Créer un `README.md` attractif pour GitHub :

```markdown
# GridWorld RL - Explainable AI with DQN

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Accuracy](https://img.shields.io/badge/Accuracy-85%25-green)
![Framework](https://img.shields.io/badge/Framework-Stable--Baselines3-orange)

## 🎯 Projet

Système d'apprentissage par renforcement (RL) avec agent DQN navigant dans un environnement GridWorld 5x5, incluant explications LLM via Gemini 2.5 Flash.

## 📊 Performances

- **Accuracy:** 85% (17/20 succès avec 3 pièges)
- **Optimalité:** 100% des succès en chemin optimal (8 pas)
- **Entraînement:** 1.2M timesteps en 18-20 minutes
- **Généralisation:** 95% (1 piège) → 65% (5 pièges)

## 🚀 Fonctionnalités

- ✅ Agent DQN avec reward shaping avancé
- ✅ Pénalités de proximité aux obstacles
- ✅ Interface Streamlit interactive
- ✅ Visualisations Plotly (trajectoires, Q-values, récompenses)
- ✅ Explications LLM avec détection d'hallucinations
- ✅ Support 1-5 pièges dynamiques

## 🛠️ Stack Technique

- **RL:** Stable-Baselines3 (DQN)
- **Framework:** PyTorch 2.0.1
- **Environnement:** Gymnasium 0.28.1
- **Interface:** Streamlit 1.28.0
- **Visualisation:** Plotly 5.17.0
- **LLM:** Google Gemini 2.5 Flash
- **Monitoring:** TensorBoard

## 📦 Installation

```bash
# Cloner le repository
git clone https://github.com/VOTRE_USERNAME/gridworld-rl-explainable.git
cd gridworld-rl-explainable

# Créer environnement virtuel
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Installer dépendances
pip install -r requirements.txt

# Configurer API Gemini (créer fichier .env)
GOOGLE_API_KEY=votre_cle_api
```

## 🎮 Utilisation

### Entraîner le modèle

```bash
python src/agents/train_dqn.py
```

### Tester le modèle

```bash
python src/agents/test_agent.py --episodes 20
```

### Lancer l'application Streamlit

```bash
streamlit run app/streamlit_app.py
```

### Visualiser TensorBoard

```bash
tensorboard --logdir=logs/tensorboard
```

## 📁 Structure

```
gridworld_llm_project/
├── src/
│   ├── environment/
│   │   ├── gridworld.py
│   │   └── random_traps_wrapper.py
│   └── agents/
│       ├── train_dqn.py
│       ├── test_agent.py
│       └── q_value_extractor.py
├── app/
│   └── streamlit_app.py
├── models/
│   └── dqn_gridworld_final.zip
├── logs/
│   └── tensorboard/
├── RAPPORT_TECHNIQUE.md
└── requirements.txt
```

## 📈 Résultats

Voir [RAPPORT_TECHNIQUE.md](RAPPORT_TECHNIQUE.md) pour analyse détaillée.

## 📝 License

MIT License - Voir [LICENSE](LICENSE)

## 👤 Auteur

Votre Nom - [@VotreGitHub](https://github.com/VotreUsername)

## 🙏 Remerciements

- Stable-Baselines3
- OpenAI Gymnasium
- Google Gemini API
```

## NOTES IMPORTANTES

### Fichiers à ne PAS pousser
- Modèles entraînés (trop volumineux) → Utiliser Git LFS ou héberger ailleurs
- Logs TensorBoard complets → Pousser uniquement captures d'écran
- Clés API → Utiliser variables d'environnement (.env)
- Environnement virtuel (venv/)

### Alternatives pour fichiers volumineux

#### Git LFS (Large File Storage)
```bash
# Installer Git LFS
git lfs install

# Tracker fichiers volumineux
git lfs track "models/*.zip"
git lfs track "models/*.pt"

# Ajouter .gitattributes
git add .gitattributes

# Commit et push normalement
git add models/
git commit -m "Add trained models"
git push
```

#### GitHub Releases
1. Aller sur GitHub → Releases → Create new release
2. Upload les fichiers .zip de modèles
3. Ajouter lien dans README.md

### Sécurité

```bash
# Ne JAMAIS pusher les clés API
# Créer fichier .env
GOOGLE_API_KEY=your_key_here

# Ajouter .env dans .gitignore
echo ".env" >> .gitignore

# Utiliser dans le code
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
```

## RÉSUMÉ COMMANDES RAPIDES

```bash
# Setup initial
cd "c:\Users\HP\Desktop\ML-DL projects\RL\gridworld_llm_project"
git init
git add .
git commit -m "Initial commit: GridWorld RL (85% accuracy)"
git remote add origin https://github.com/VOTRE_USERNAME/gridworld-rl-explainable.git
git branch -M main
git push -u origin main

# Modifications futures
git add .
git commit -m "Description changement"
git push
```

---

**Prêt à pousser vers GitHub!** Suivez les étapes dans l'ordre pour un setup complet.
