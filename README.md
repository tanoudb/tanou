# Webtoon Translator V5

Pipeline de traduction d’images manhwa/webtoon :
- détection zones texte (YOLO)
- OCR (`PP-OCRv5` en principal, `EasyOCR` en fallback)
- traduction (`NLLB`)
- rendu + inpainting

## Prérequis (Windows)

- Python **3.11** recommandé (3.10 possible)
- GPU NVIDIA recommandé (CUDA 12.x)
- VS Code + extension Python

## Environnement recommandé

Le projet contient actuellement `.venv` (Python 3.14) qui n’est pas compatible avec la stack OCR.
Utilise un env dédié Python 3.11 :

```powershell
cd "A:\manhwa trad v2"
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Lancement

```powershell
python main.py --show-config
python main.py --debug
```

## Configuration OCR

Dans `config/settings.py` :
- backend par défaut : `ppocr-v5`
- fallback automatique : `easyocr`
- chemin env NVIDIA : variable `PADDLE_ENV_PATH` (sinon `.venv` par défaut)

Exemple :

```powershell
$env:PADDLE_ENV_PATH = "A:\manhwa trad v2\.venv311"
python main.py --debug
```

## DLL Hell (NVIDIA)

Le module `core/dll_manager.py` :
- synchronise les DLL CUDA/cuDNN dans `torch\lib`
- enregistre les dossiers DLL dans `PATH`
- précharge les DLL critiques avant chargement OCR

Ce mécanisme est appelé automatiquement au démarrage du pipeline.

⚠️ Note importante (Windows, février 2026):
- la stack `PP-OCRv5` validée dans ce projet fonctionne avec `paddleocr==3.4.0` + `paddlepaddle==3.1.1`;
- `paddlepaddle-gpu` Windows n’est pas aligné sur cette version, donc il est volontairement retiré pour éviter les conflits DLL.

## Dépendances clés

- `paddleocr` + `paddlepaddle`
- `easyocr`
- `torch`, `torchvision`, `torchaudio`
- `ultralytics`
- `transformers`, `sentencepiece`
- `opencv-python`, `pillow`, `numpy`, `psutil`
- `simple-lama-inpainting`

