# Webtoon Translator V5

Pipeline de traduction d’images manhwa/webtoon :
- détection zones texte (YOLO)
- segmentation précise guidée par YOLO (`SmartSegmenter`, multi-masques)
- OCR (`PaddleOCR-VL v1.5` en principal, `PP-OCRv5` en fallback, puis `EasyOCR` secours)
- traduction (`LLM local` par défaut, `NLLB` en option)
- rendu + inpainting (masques pixel-level, conservation de couleur/style)

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

Par défaut Docker lance la segmentation en mode `hybrid` (stable, sans checkpoint SAM2).

Pour activer SAM2 dans Docker:

1) Place un checkpoint en local dans `assets/cache/ocr_weights/sam2_b.pt`
2) Lance avec override d'env:

```powershell
$env:WEBTOON_SEGMENTATION_BACKEND = "sam2"
docker compose up --build
```

Pour revenir au mode stable:

```powershell
$env:WEBTOON_SEGMENTATION_BACKEND = "hybrid"
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
- backend principal par défaut : `paddleocr-vl-v1.5`
- fallback automatique : `ppocr-v5`, puis `easyocr`
- chemin env NVIDIA : variable `PADDLE_ENV_PATH` (sinon `.venv` par défaut)

Note dépendances OCR-VL:
- `PaddleOCR-VL v1.5` nécessite les extras `paddlex[ocr]`.
- L'image Docker les installe automatiquement.

Variables OCR utiles:

- `WEBTOON_OCR_PRIMARY=paddleocr-vl-v1.5`
- `WEBTOON_OCR_FALLBACKS=ppocr-v5,easyocr`
- `WEBTOON_OCR_FALLBACK_MIN_CONF=0.72` (si confiance plus basse, fallback)

Exemple :

```powershell
$env:PADDLE_ENV_PATH = "A:\manhwa trad v2\.venv311"
python main.py --debug
```

## Variables d'environnement utiles

### Détection

- `WEBTOON_USE_BLACK_PADDING=false` (défaut) : désactive le hack barres noires
- `WEBTOON_BLACK_PADDING_RATIO=0.03` : ratio si padding activé

### Segmentation précise (YOLO → masques)

- `WEBTOON_ENABLE_PRECISE_MASKS=true|false` : active les masques fins pour l’inpainting
- `WEBTOON_SEGMENTATION_BACKEND=hybrid|sam2|ocr_regions` : backend segmentation
- `WEBTOON_SAM2_CHECKPOINT=...` : checkpoint SAM/SAM2 (défaut: `assets/cache/ocr_weights/sam2_b.pt`)
- `WEBTOON_SEGMENTATION_MULTIMASK=true|false` : sépare les composants (bulles fusionnées)
- `WEBTOON_SEGMENTATION_MIN_COMPONENT=24` : aire minimale d’un composant masque
- `WEBTOON_SEGMENTATION_DILATE=9` : dilation du masque texte
- `WEBTOON_SEGMENTATION_PROMPT_MARGIN=4` : marge de prompt bbox

Notes:
- `sam2` est optionnel. Si indisponible, fallback automatique en mode `hybrid`.
- Le backend `hybrid` utilise les régions OCR + raffinage morphologique.

Activation SAM2 (si checkpoint présent):

```powershell
$env:WEBTOON_SEGMENTATION_BACKEND = "sam2"
$env:WEBTOON_SAM2_CHECKPOINT = "A:\manhwa trad v2\assets\cache\ocr_weights\sam2_b.pt"
python main.py --show-config
python main.py --debug
```

### Traduction (LLM local, sans API)

- `WEBTOON_TRANSLATION_BACKEND=local_llm|nllb` : backend de traduction (`local_llm` par défaut)
- `WEBTOON_LLM_MODEL=Qwen/Qwen2.5-3B-Instruct` : modèle LLM téléchargé localement
- `WEBTOON_LLM_REQUIRE_CUDA=true|false` : refuse le fallback CPU si CUDA indisponible
- `WEBTOON_LLM_MAX_NEW_TOKENS=220` : longueur max de sortie
- `WEBTOON_LLM_TEMPERATURE=0.0` : 0 = déterministe
- `WEBTOON_LLM_TOP_P=1.0` : nucleus sampling si température > 0
- `WEBTOON_LLM_REPETITION_PENALTY=1.05` : limite les répétitions

Le modèle est téléchargé dans `assets/cache/translation_models` via `transformers` au premier lancement.

Exemple local GPU (Windows):

```powershell
$env:WEBTOON_TRANSLATION_BACKEND = "local_llm"
$env:WEBTOON_LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
$env:WEBTOON_LLM_REQUIRE_CUDA = "true"
python main.py --debug
```

Option plus performante (si VRAM suffisante, surtout en Docker Linux):

```powershell
$env:WEBTOON_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
```

### Traduction (VRAM / quantification)

- `WEBTOON_USE_BITSANDBYTES=true|false` : active la quantification du modèle de traduction
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

### Rendu (style + couleur)

- `WEBTOON_PRESERVE_TEXT_COLOR=true|false` : reprend la couleur dominante du texte source
- `WEBTOON_AUTO_STYLE_TYPESETTING=true|false` : auto-style basique (dialogue/narration/cri/chuchotement)
- `WEBTOON_LOCK_TEXT_TO_OCR_REGIONS=true|false` : place le texte traduit à partir des régions OCR d'origine
- `WEBTOON_LOCK_TEXT_SYSTEM_ONLY=true|false` : limite ce verrouillage à la classe `System` (recommandé)

## Debug pipeline pro

En `--debug`, le dossier `output/debug/<image>_pipeline/` contient par zone:
- crop source
- masque segmenté
- rendu avant/après
- texte OCR + texte traduit + style

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

