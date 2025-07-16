# 📋 Module d'analyse de devis automobile et génération de matrices LLD

## 🚀 Fonctionnalités

### 1. 📋 Extraction de devis PDF
- Extraction automatique de données depuis des fichiers PDF de devis automobiles
- Analyse intelligente via LLM (OpenAI GPT)
- Détection automatique des loueurs
- Insertion des données dans un fichier Excel de suivi
- Support de multiples loueurs avec configuration personnalisable

### 2. 🚗 Génération de matrice Location Longue Durée (LLD)
**NOUVELLE FONCTIONNALITÉ**
- Chargement du fichier Excel de suivi des véhicules
- Sélection interactive des véhicules à inclure dans la matrice
- Génération automatique d'une matrice Excel LLD complète
- Support des templates de matrice personnalisés
- Calculs automatiques et formules intégrées

## 🏗️ Architecture

```
streamlit/
├── analyse_devis.py          # Application principale avec menu
├── modules/
│   ├── data_processor.py     # Traitement des données avec LLM
│   ├── excel_manager.py      # Gestion des fichiers Excel (suivi)
│   ├── lld_matrix_manager.py # Gestion des matrices LLD
│   ├── lld_app.py           # Application matrice LLD
│   ├── loueur_detector.py    # Détection des loueurs
│   ├── pdf_extractor.py      # Extraction PDF
│   └── ui_components.py      # Composants d'interface
├── config/
│   └── loueurs_config.py     # Configuration des loueurs
└── requirements.txt
```

## 🛠️ Installation

```bash
# Cloner le projet
git clone <url-du-repo>
cd streamlit

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt

# Configurer la clé API OpenAI
echo "OPENAI_API_KEY=votre_cle_api" > .env
```

## 🏃‍♂️ Utilisation

### Lancement de l'application
```bash
streamlit run analyse_devis.py
```

### Menu principal
L'application propose deux fonctionnalités principales :

#### 📋 Extraction de devis PDF
1. Sélectionner "📋 Extraction de devis PDF" dans le menu
2. Charger les fichiers PDF de devis
3. Charger le template Excel de suivi
4. Lancer l'extraction automatique
5. Télécharger le fichier Excel mis à jour

#### 🚗 Génération matrice LLD
1. Sélectionner "🚗 Génération matrice LLD" dans le menu
2. Charger le fichier Excel de suivi (généré par la première fonctionnalité)
3. Optionnel : Charger un template de matrice personnalisé
4. Sélectionner les véhicules à inclure dans la matrice
5. Générer et télécharger la matrice Excel LLD

## 📊 Structure de la matrice LLD

La matrice générée contient :

### VÉHICULE
- Marque, Modèle, Finition
- Caractéristiques techniques (puissance, CO2, consommation)
- Prix (catalogue, options, batterie électrique)
- Informations techniques (boîte, énergie, autonomie)

### PRESTATIONS
- Durée et kilométrage
- Services inclus/non inclus :
  - Loyer financier ✅
  - Entretien ✅
  - Assistance 24/24 ✅
  - Perte financière ✅
  - Véhicule de remplacement ❌
  - Pneumatiques ❌
  - Frais d'immatriculation ✅
  - Accompagnement LEASYGO ✅
  - Accès LEASYDRIVE ✅

### SECTIONS FUTURES
- Coûts autres (assurance, pneumatiques)
- Fiscalité (TVS, malus écologique, AND)
- Analyse TCO estimative
- Avantages en nature et cotisations sociales

## 🔧 Configuration

### Variables d'environnement (.env)
```
OPENAI_API_KEY=votre_cle_api_openai
```

### Configuration des loueurs
Modifier `config/loueurs_config.py` pour ajouter de nouveaux loueurs :

```python
LOUEURS_CONFIG = {
    "NOUVEAU_LOUEUR": {
        "termes_detection": ["terme1", "terme2"],
        "mapping": {
            "marque": ["marque", "brand"],
            "modele": ["modèle", "model"],
            # ...
        }
    }
}
```

## 📝 Logs et débogage

- L'application affiche des messages détaillés sur le processus d'extraction
- Les erreurs sont capturées et affichées à l'utilisateur
- Suggestion automatique de configuration pour les nouveaux loueurs

## 🤝 Contribution

1. Fork le projet
2. Créer une branche (`git checkout -b feature/amelioration`)
3. Commit (`git commit -am 'Ajout de fonctionnalité'`)
4. Push (`git push origin feature/amelioration`)
5. Créer une Pull Request

## 📄 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 🆘 Support

Pour toute question ou problème :
1. Vérifier que la clé API OpenAI est correctement configurée
2. Vérifier que tous les modules sont installés (`pip install -r requirements.txt`)
3. Consulter les logs d'erreur affichés dans l'interface Streamlit 

# Analyseur Automatisé de Devis LLD (Gemini 2.0 Flash-Lite)

Application Streamlit pour l'extraction automatique d'informations à partir de devis de location longue durée (LLD) en utilisant l'API Gemini 2.0 Flash-Lite de Google.

## Fonctionnalités

- **Extraction automatique** : Utilise l'IA Gemini 2.0 Flash-Lite pour extraire les données clés des devis PDF
- **Support multi-formats** : Compatible avec les devis de plusieurs loueurs (Arval, Alphabet, Ayvens, Athlon, etc.)
- **Interface intuitive** : Interface web Streamlit facile à utiliser
- **Export Excel** : Téléchargement des résultats au format Excel
- **Traitement par lots** : Analyse de plusieurs devis simultanément

## Installation

1. Cloner le repository :
```bash
git clone <repository-url>
cd streamlit
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

3. Configurer la clé API Gemini :
   - Créer un fichier `.env` dans le répertoire racine
   - Ajouter votre clé API Gemini :
```
GEMINI_API_KEY=votre_cle_api_gemini_ici
```

## Obtenir une clé API Gemini

1. Aller sur [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Créer un nouveau projet ou sélectionner un projet existant
3. Générer une nouvelle clé API
4. Copier la clé dans votre fichier `.env`

## Utilisation

1. Lancer l'application :
```bash
streamlit run analyse_devis_gemini.py
```

2. Dans l'interface web :
   - La clé API Gemini sera chargée automatiquement depuis le fichier `.env`
   - Ou entrer manuellement votre clé API Gemini dans la sidebar
   - Télécharger vos fichiers PDF de devis
   - Cliquer sur "🚀 Lancer l'analyse"
   - Consulter et éditer les résultats
   - Télécharger le fichier Excel

## Loueurs supportés

L'application reconnaît automatiquement les devis des loueurs suivants :

- **Arval** : Extraction optimisée pour les devis Arval
- **Alphabet** : Support des formats Alphabet
- **Ayvens** (ex-Temsys) : Reconnaissance des devis Ayvens
- **Athlon** : Extraction spécialisée Athlon
- **Agilauto** : Support des devis Agilauto
- **Olinn** : Compatibilité avec les formats Olinn

## Données extraites

L'application extrait automatiquement :

- **Véhicule** : Marque, modèle, finition
- **Prix catalogue** : Prix TTC avec options
- **Consommation** : kWh/100km ou L/100km
- **CO2** : Émissions en g/km
- **Malus écologique** : Montant du malus
- **TVS** : Taxe sur les Véhicules de Société
- **AND** : Amortissements Non Déductibles (mensuel)
- **Loi de roulage** : Durée et kilométrage
- **Prestations** : Services inclus
- **Loyer total** : Montant mensuel TTC
- **Date d'offre** : Date d'émission du devis

## Structure du projet

```
.
├── analyse_devis_gemini.py    # Application principale avec Gemini 2.0 Flash-Lite
├── analyse_devis_old.py       # Version OpenAI (obsolète)
├── requirements.txt           # Dépendances Python
├── .env.example              # Exemple de configuration
└── README.md                 # Documentation
```

## Dépendances principales

- `streamlit` : Interface web
- `google-genai` : SDK Gemini
- `pandas` : Manipulation de données
- `pdfplumber` / `PyMuPDF` : Extraction de texte PDF
- `xlsxwriter` : Export Excel

## Configuration

### Fichier .env

Créez un fichier `.env` dans le répertoire racine :

```bash
GEMINI_API_KEY=votre_cle_api_gemini_ici
```

L'application chargera automatiquement cette clé au démarrage.

## Développement

### Ajouter un nouveau loueur

1. Modifier `LOUEURS_CONFIG` dans `analyse_devis_gemini.py`
2. Ajouter les identifiants et mappings spécifiques
3. Tester avec des devis du nouveau loueur

### Améliorer l'extraction

1. Ajuster les prompts système dans `get_system_prompt()`
2. Modifier la fonction `extract_devis_data()` si nécessaire
3. Tester avec différents formats de devis

## Modèle utilisé

L'application utilise **Gemini 2.0 Flash-Lite** qui offre :
- **Rapidité d'exécution** optimisée
- **Coûts réduits** par rapport aux modèles plus lourds
- **Précision** maintenue pour l'extraction de données
- **Function calling** natif

## Limitations

- Nécessite une clé API Gemini valide
- Les PDF doivent contenir du texte extractible (pas d'images scannées)
- La précision dépend de la qualité du texte extrait

## Support

Pour des questions ou des problèmes :
1. Vérifier que la clé API Gemini est valide et configurée dans `.env`
2. S'assurer que les PDF contiennent du texte extractible
3. Consulter les logs pour identifier les erreurs

## Licence

Ce projet est sous licence MIT.

# 🚗 Extracteur de Devis Avancé - Stratégie Multi-Sources

## 📚 Inspiré par deux articles de référence

Ce projet combine les meilleures pratiques de deux articles experts :
- [Parseur.com - Extraire des données de factures avec Python](https://parseur.com/fr/blog/extraire-donnees-factures-python)
- [Cohorte.co - Streamlit App to Extract Tables from PDFs](https://www.cohorte.co/blog/docs-to-table-building-a-streamlit-app-to-extract-tables-from-pdfs-and-answer-questions)

## 🚀 Améliorations Apportées

### 1. **Stratégie d'Extraction 4 Niveaux**
- ✅ **unstructured** (extraction JSON tableaux - Cohorte.co)
- ✅ **pdftotext** (rapide et précis - Parseur.com)
- ✅ **PyMuPDF + OCR** (fallback pour PDFs scannés)
- ✅ **Expressions régulières** (fallback si LLM échoue)

### 2. **Extraction JSON Tableaux (Cohorte.co)**
- Détection automatique des tableaux dans les PDFs
- Conversion en JSON structuré pour améliorer l'analyse LLM
- Extraction HTML des tableaux complexes
- Contexte supplémentaire pour une précision maximale

### 3. **Schéma JSON Structuré (Parseur.com)**
```json
{
    "vehicule": "Modèle et marque du véhicule",
    "loueur": "ARVAL, ALPHABET, AYVENS...", 
    "prix_catalogue": "Prix TTC en euros",
    "loyers_v1": "Loyer mensuel tout compris",
    "and": "AND mensuel en euros",
    "tvs": "TVS annuelle en euros"
}
```

### 4. **Gestion Robuste des Erreurs**
- Fallback automatique entre 4 méthodes
- Extraction garantie même avec PDFs difficiles
- Logging détaillé des performances
- Données JSON optimisent la précision LLM

## 🔧 Installation

### 1. Dépendances Python
```bash
pip install -r requirements.txt
```

### 2. Dépendances Système

**unstructured (Recommandé par Cohorte.co):**
```bash
# Installation automatique avec pip install unstructured
# Aucune dépendance système supplémentaire requise
```

**pdftotext (Recommandé par Parseur.com):**
```bash
# MacOS
brew install poppler

# Ubuntu/Debian
sudo apt-get install poppler-utils

# Windows
# Téléchargez Poppler depuis: https://github.com/oschwartz10612/poppler-windows
```

**Tesseract OCR:**
```bash
# MacOS
brew install tesseract tesseract-lang

# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-fra

# Windows
# Téléchargez depuis: https://github.com/UB-Mannheim/tesseract/wiki
```

## 📦 Utilisation

### 1. Lancer l'application
```bash
streamlit run test_extraction_bmw.py
```

### 2. Configuration automatique
- L'application détecte automatiquement les dépendances disponibles
- Affiche le statut des méthodes d'extraction
- Guide l'installation des dépendances manquantes

### 3. Processus d'extraction
1. **Upload** de vos PDFs de devis
2. **Extraction automatique** avec stratégie en cascade
3. **Analyse LLM** avec fallback regex
4. **Injection Excel** avec mapping automatique

## 🎯 Fonctionnalités Avancées

### Multi-Devis par PDF
- ✅ Détection automatique de plusieurs devis dans un même PDF
- ✅ 1 ligne Excel par devis trouvé
- ✅ Numérotation automatique des devis

### Extraction Intelligente 4 Niveaux
- ✅ **unstructured** pour tableaux JSON (idéal pour devis structurés)
- ✅ **pdftotext** pour PDFs texte (plus rapide que PyMuPDF)
- ✅ **OCR** pour PDFs scannés/images
- ✅ **Regex** pour patterns spécifiques si LLM échoue

### Support Universel Loueurs
- ARVAL, ALPHABET, AYVENS, ATHLON
- Concessionnaires BMW, Mercedes, etc.
- Adaptation automatique aux nouveaux formats

## 📊 Avantages vs Code Original

| Fonctionnalité | Avant | Maintenant |
|----------------|-------|------------|
| Méthodes d'extraction | 1 (PyMuPDF) | 4 (cascade) |
| Extraction JSON | ❌ | ✅ Unstructured |
| Fallback si échec | ❌ | ✅ 3 niveaux |
| Performance | Moyenne | Optimisée |
| Multi-devis | ❌ | ✅ |
| Robustesse | Limitée | Maximale |
| Installation | Complexe | Guidée |

## 🔍 Diagnostic et Debug

### Statut des Dépendances
L'application affiche automatiquement :
- ✅/❌ Tesseract OCR
- ✅/❌ pdftotext
- ✅/❌ OpenAI API

### Logs Détaillés
- Méthode d'extraction utilisée pour chaque PDF
- Performance de chaque étape
- Fallback automatique en cas d'échec
- Statistiques d'injection Excel

## 🏆 Résultats

### Performance
- **Vitesse** : unstructured + pdftotext optimisent l'extraction
- **Précision** : JSON structuré améliore les résultats LLM
- **Robustesse** : Aucun échec total grâce aux 4 méthodes en cascade

### Extraction
- **Multi-devis** : Support natif
- **Tous formats** : Loueurs + concessionnaires
- **Données complètes** : Loyers, AND, TVS, prestations...

## 🤝 Crédits

**Articles de référence :**
- **Parseur.com** : [Extraire des données de factures avec Python](https://parseur.com/fr/blog/extraire-donnees-factures-python)
- **Cohorte.co** : [Building a Streamlit App to Extract Tables from PDFs](https://www.cohorte.co/blog/docs-to-table-building-a-streamlit-app-to-extract-tables-from-pdfs-and-answer-questions)

**Techniques intégrées :**
- **Méthode unstructured** : Extraction JSON tableaux (Cohorte.co)
- **Méthode pdftotext** : Extraction rapide (Parseur.com)
- **Stratégie cascade** : Combinaison des meilleures pratiques
- **Schéma JSON** : Structure optimisée pour LLM

## 📄 License

MIT - Libre d'utilisation et modification 