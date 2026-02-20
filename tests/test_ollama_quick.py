"""
Script de test rapide pour vérifier qu'Ollama fonctionne
Exécuter: python test_ollama_quick.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import ollama
    print("✅ Package ollama installé")
except ImportError:
    print("❌ Package ollama non installé")
    print("📦 Installer avec: pip install ollama")
    sys.exit(1)

# Tester la connexion à Ollama
try:
    models = ollama.list()
    print(f"✅ Ollama fonctionne ! Modèles disponibles: {len(models['models'])}")
    
    if models['models']:
        print("\n📋 Modèles installés:")
        for model in models['models']:
            print(f"   - {model['name']} ({model['size'] / 1e9:.1f}GB)")
    else:
        print("\n⚠️  Aucun modèle installé")
        print("📥 Télécharger un modèle avec: ollama pull llama3.2:3b")
        
except Exception as e:
    print(f"❌ Ollama ne fonctionne pas: {e}")
    print("\n📥 Solutions:")
    print("   1. Télécharger Ollama: https://ollama.com/download")
    print("   2. Installer et démarrer l'application")
    print("   3. Télécharger un modèle: ollama pull llama3.2:3b")
    sys.exit(1)

# Test d'inférence simple
print("\n🧪 Test d'inférence avec le premier modèle disponible...")
try:
    if models['models']:
        model_name = models['models'][0]['name']
        print(f"   Modèle: {model_name}")
        
        response = ollama.chat(
            model=model_name,
            messages=[
                {'role': 'user', 'content': 'Dis bonjour en une phrase.'}
            ]
        )
        
        print(f"   Réponse: {response['message']['content']}")
        print("\n✅ Ollama est prêt à être utilisé !")
        print(f"\n💡 Pour l'utiliser dans le projet:")
        print(f"   from src.llm.explainer_ollama import RLExplainerOllama")
        print(f"   explainer = RLExplainerOllama(model_name='{model_name}')")
        
except Exception as e:
    print(f"❌ Erreur lors du test: {e}")
    sys.exit(1)
