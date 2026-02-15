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

## Docker (Linux CUDA)

Objectif: environnement stable GPU (Torch/OCR) sans gestion manuelle des DLL Windows.

Build + run:

```powershell
docker compose build
docker compose up
```

Les dossiers suivants sont montés en volume:
- `input/` → `/app/input`
- `output/` → `/app/output`
- `logs/` → `/app/logs`
- `assets/cache/`, `assets/models/`, `assets/fonts/`

Le service utilise `gpus: all` (NVIDIA Container Toolkit requis).

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

## Variables d'environnement utiles

### Détection

- `WEBTOON_USE_BLACK_PADDING=false` (défaut) : désactive le hack barres noires
- `WEBTOON_BLACK_PADDING_RATIO=0.03` : ratio si padding activé

### Traduction (VRAM)

- `WEBTOON_USE_BITSANDBYTES=true|false` : active la quantification NLLB
- `WEBTOON_BNB_4BIT=true|false` : mode 4-bit (recommandé)
- `WEBTOON_BNB_8BIT=true|false` : mode 8-bit (alternative)
- `WEBTOON_AUTO_DETECT_SOURCE_LANG=true|false` : détecte automatiquement la langue OCR
- `WEBTOON_FALLBACK_SOURCE_LANG=en` : langue source de secours si détection incertaine

Exemple PowerShell:

```powershell
$env:WEBTOON_USE_BITSANDBYTES = "true"
$env:WEBTOON_BNB_4BIT = "true"
$env:WEBTOON_BNB_8BIT = "false"
python main.py --show-config
```

### Regroupement contextuel (ordre de lecture)

- `WEBTOON_ENABLE_CONTEXT_GROUPING=true|false`
- `WEBTOON_CONTEXT_DISTANCE_THRESHOLD=300` (augmenter pour fusionner plus de bulles proches)
- `WEBTOON_MAX_GROUP_SIZE=5`

Preset conseillé pour webtoons longs:

```powershell
$env:WEBTOON_ENABLE_CONTEXT_GROUPING = "true"
$env:WEBTOON_CONTEXT_DISTANCE_THRESHOLD = "420"
$env:WEBTOON_MAX_GROUP_SIZE = "7"
python main.py --show-config
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

