# 🎞️ DOF Tool HQ

**DOF Tool HQ** est un outil de simulation de profondeur de champ (*Depth of Field*) haute qualité avec interface graphique, conçu pour les artistes, photographes et créateurs 3D. Il prend en entrée une image RGB et une depth map 16 bits, et génère un rendu bokeh réaliste et sans halos.

La raison d'être de ce programme est le besoin de réaliser plus rapidement ce calcul que celui offert par **Mandelbulb 3D**, qui offre les deux images nécessaires, en des qualités parfaites : 

1. **Image en RVB** : PNG 8bits
2. **Z-DepthMap** : PGM 16bits / PNG Grayscale 16bits

Il est également possible de générer la depthmap d'une photo en se basant sur les modèles suivants : 

- Depth Anything v2 (Small, Normal et Large)
- Depth Anything v2 Metrics (Indoor et Outdoor)
- Depth Anything v3 (Small, Normal et Large)
- Marigold.

Attention, ces opérations demandent une carte graphique raisonnable mais sont parfaitement réalisibles sur une carte graphique "ancienne".

La configuration utilisée ici est un simple Laptop LDLC i5-9400 2,9Ghz avec nVidia GTX 1650.

---

###

---

## ✨ Fonctionnalités

### Rendu
- **Rendu par couches séparées** : zéro halo sur fond blanc ou coloré
- **Formes de bokeh** : Disque, Hexagone, Gaussien (doux), et plus
- **Zone de netteté** interactive avec limites proche/lointaine et transitions douces indépendantes
- **Flou de bord radial** par sujet (*radial edge blur*)
- **Courbe Z-depth** ajustable : linéaire, concave, convexe, en S, S inversé…
- **Auto-détection de l'inversion Z-depth** (0 = près ou 0 = loin)
- **Séparation focus-aware** pour un résultat précis au niveau des contours

### Interface
- **Aperçu en temps réel** avec modes : Original / Z-Depth / Carte de flou / Résultat DoF
- **Comparaison split-screen** glissable (avant/après)
- **Marqueur sujet** positionnable par clic pour le flou radial
- **Sélection des limites** proche/lointaine par clic directement sur l'image
- **Panneau gauche redimensionnable** par glisser-déposer du sash
- **Overlay coloré** des zones de netteté avec opacité réglable

### Workflow
- **Glisser-déposer** d'un ou deux fichiers simultanément (RGB + depth auto-détectés)
- **Détection automatique** NB / couleur pour assigner les fichiers
- Calcul en **pleine résolution** à la demande
- **Export PNG / JPEG**

### Accélération GPU
| Backend | Condition |
|---|---|
| ⚡ CuPy (CUDA) | `pip install cupy-cuda12x` + GPU NVIDIA |
| ⚡ PyTorch CUDA | `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| ⚡ PyTorch MPS | Apple Silicon (M1/M2/M3) avec PyTorch |
| 🖥️ SciPy CPU | Fallback par défaut |
| 🖥️ OpenCV CPU | Si SciPy absent |
| 🖥️ NumPy CPU | Fallback minimal |

Le diagnostic GPU est accessible depuis l'interface.

### Langues supportées
🇫🇷 Français · 🇬🇧 English · 🇧🇪 Nederlands · 🇩🇪 Deutsch · 🇨🇳 中文 · 🇯🇵 日本語 · 🇸🇦 العربية · 🇷🇺 Русский · 🇮🇳 हिन्दी · 🇪🇸 Español · 🇧🇷 Português · 🇧🇩 বাংলা · 🏴 Wallon · 🏴 Klingon

---

## 📦 Installation

### Dépendances requises
```bash
pip install numpy pillow scipy
```

### Dépendances optionnelles
```bash
# Glisser-déposer de fichiers
pip install tkinterdnd2

# Accélération CPU alternative
pip install opencv-python

# Accélération GPU NVIDIA (CUDA 12.x)
pip install cupy-cuda12x

# Accélération GPU NVIDIA via PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

> **Note :** L'application fonctionne entièrement sans GPU. Les dépendances optionnelles améliorent les performances ou ajoutent des fonctionnalités confort.

---

## 🚀 Lancement

```bash
python DOFTool_v13_2_.py
```

La fenêtre s'ouvre en 1440×860. Elle est librement redimensionnable.

---

## 🖱️ Utilisation rapide

1. **Charger** une image RGB (JPG, PNG…) et une depth map 16 bits (PNG, EXR…)
   — ou **glisser-déposer** les deux fichiers directement sur la fenêtre
2. Ajuster la **courbe Z-depth** si nécessaire (avec auto-détection d'inversion)
3. Définir la **zone de netteté** en déplaçant les poignées sur la courbe, ou en cliquant les boutons *Pick near / Pick far* directement sur l'image
4. Choisir la **forme du bokeh** et le **flou maximum**
5. (Optionnel) Cliquer sur le sujet pour activer le **flou de bord radial**
6. Cliquer **▶ Calculer (pleine résolution)**
7. Vérifier le résultat avec le **split-screen** et **exporter** en PNG ou JPEG

---

## 🗂️ Paramètres principaux

| Paramètre | Description |
|---|---|
| Inverser Z-depth | Inverse la convention de profondeur (0 = loin ↔ 0 = près) |
| Courbe Z-depth | Remapping non-linéaire de la profondeur |
| Limite proche / lointaine | Bornes de la zone nette |
| Transition proche / lointaine | Douceur du fondu de chaque côté de la zone nette |
| Flou maximum (px) | Rayon maximal du bokeh en pixels |
| Forme du bokeh | Disque / Hexagone / Gaussien / … |
| Couches (qualité) | Nombre de couches pour le rendu (+ = meilleur, + lent) |
| Flou de bord radial | Intensité et ellipticité du flou autour du sujet |

---

## 🔧 Compatibilité

- **Python** : 3.8+
- **Pillow** : compatible ≥ 10 (gestion automatique de `Image.Resampling`)
- **OS** : Windows, macOS, Linux
- **GPU** : NVIDIA (CUDA), Apple Silicon (MPS) — optionnel

---

## 📝 Changelog

| Version | Nouveautés principales |
|---|---|
| v13 | Auto-détection inversion Z-depth, aperçu depth sous la courbe, UI améliorée |
| v12 | Correctifs GPU critiques, kernels vectorisés, gaussien séparable, i18n complète |
| v11 | Glisser-déposer automatique, détection NB/couleur, dépôt multi-fichiers |
| v10 | Flou de bord radial par sujet (*edge blur*) |
| v9 | Séparation focus-aware |
| v8 | Rendu par couches (anti-halo) |

