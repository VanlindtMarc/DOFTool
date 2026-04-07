# DOFTool
Logiciel permettant de générer un flou de profondeur à partir d'une image et de sa carte de profondeur.

Ce logiciel a été écrit par **Claude.AI** en Python.

Les différentes dépendances imposent **Python 3.11**.

Vous devez également installer les dépendances via la commande suivante : 
```
python -m pip install numpy pillow scipy opencv-python
```

Pour effectuer le rendu par carte graphique vous devez installer **PyTorch** :
```
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Utilisation
![Interface utilisateur](https://github.com/VanlindtMarc/DOFTool/blob/main/DOFTool.png)

Double-cliquez sur le fichier DOFtool.py

Si rien ne se passe, utilisez la ligne de commande 
```
python DOFTool.py
```

Vous devez disposer de deux images : 
1. L'image sans flou
2. La carte de profondeur en noir et blanc

DOFTool vous proposera :
* Prévisualisation
* Inverser le z-depth...
* Indiquer les limites proches et lointaines
* Indiquer ces limites visuellement
* Appliquer une transition entre zones différentes
* Indiquer le niveau de flou
* Choisir son type de calcul de flou (Bokeh circulaire, hexagonal ou flou gaussien)
* Nombre de passes
* Anti-bleeding
* Barre d'avancement des calculs
* Export en JPG et PNG
* Plusieurs langues : français, anglais, néerlandais, allemand, espagnol, portugais, japonais, chinois, arabe, russe, hindi, bengali, klingon et wallon liégeois.

### Exemple d'utilisation : photo de famille
Je vais partir de la photo suivante.

