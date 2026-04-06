"""
Depth of Field Tool  –  High Quality  (v9)
==========================================
• Rendu par couches séparées : zéro halo sur fond blanc ou coloré.
• v9 : Séparation focus-aware — les couches respectent l'ordre de profondeur.
• Pre-fill anti-saignement — les bords nets restent nets.
• Panneau gauche REDIMENSIONNABLE par glisser-déposer.
• Tous les 14 drapeaux toujours visibles (défilement si nécessaire).

Langues : 🇫🇷 Français   🇬🇧 English    🇧🇪 Nederlands  🇩🇪 Deutsch
          🇨🇳 中文       🇯🇵 日本語     🇸🇦 العربية     🏴 Klingon
          🏴 Wallon li.  🇷🇺 Русский    🇮🇳 हिन्दी       🇪🇸 Español
          🇧🇷 Português  🇧🇩 বাংলা

Dépendances : pip install numpy pillow scipy
              (optionnel)  pip install opencv-python
"""

import os
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import math

# ── Compatibilité Pillow ≥ 10  (Image.BILINEAR → Image.Resampling.BILINEAR) ──
_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR
_NEAREST  = getattr(Image, "Resampling", Image).NEAREST
_LANCZOS  = getattr(Image, "Resampling", Image).LANCZOS

try:
    from scipy.ndimage import (convolve, maximum_filter, minimum_filter,
                                gaussian_filter, uniform_filter)
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# =============================================================================
# GPU Backend  (CuPy → PyTorch CUDA/MPS → SciPy/CPU)
# =============================================================================

import subprocess as _subprocess
import json       as _json

GPU_BACKEND     = "cpu"
GPU_DEVICE      = ""
_TORCH_DEVICE   = None
_GPU_FAIL_CUPY  = ""
_GPU_FAIL_TORCH = ""
_CUPY_OK        = False

# ─────────────────────────────────────────────────────────────────────────────
# Probe CuPy dans un SOUS-PROCESSUS isolé.
#
# Pourquoi ? Certaines erreurs CUDA (version mismatch, DLL corrompue, driver
# incompatible) ne lèvent PAS d'exception Python : elles provoquent un crash
# C-level (segfault / access violation) qui tue le process sans message.
# try/except est alors inutile.  En isolant le test dans un subprocess,
# le crash éventuel n'affecte que l'enfant — le process principal reste intact.
# ─────────────────────────────────────────────────────────────────────────────
_CUPY_PROBE = """
import json, sys
r = {"ok": False, "device": "", "error": ""}
try:
    import cupy as cp
    from cupyx.scipy.ndimage import (
        convolve, maximum_filter, minimum_filter,
        gaussian_filter, uniform_filter,
    )
    a = cp.zeros((8, 8), dtype=cp.float32)
    _ = float((a + 1.0).sum())       # déclenche la compilation JIT (NVRTC)
    del a, _
    p = cp.cuda.runtime.getDeviceProperties(0)
    r["ok"]     = True
    r["device"] = p["name"].decode(errors="replace")
except ImportError as e:
    r["error"] = "not_installed:" + str(e)
except Exception as e:
    r["error"] = str(e)
print(json.dumps(r), flush=True)
"""

def _probe_cupy_safe() -> dict:
    """Lance le probe CuPy dans un subprocess; retourne {"ok":bool, ...}."""
    import sys  # import local — robuste même si le namespace global est partiel
    try:
        proc = _subprocess.run(
            [sys.executable, "-c", _CUPY_PROBE],
            capture_output=True, text=True, timeout=25,
        )
        # Prendre la dernière ligne non-vide (des warnings peuvent précéder)
        for line in reversed(proc.stdout.strip().split("\n")):
            if line.strip().startswith("{"):
                return _json.loads(line.strip())
        # Stdout vide ou non-JSON → le process a crashé
        stderr = (proc.stderr or "").strip()
        code   = proc.returncode
        detail = (f": {stderr[:400]}" if stderr else "")
        return {"ok": False,
                "error": f"crash (exit {code}){detail}"}
    except _subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout (>25 s) — init CuPy bloquée"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── Tentative CuPy ────────────────────────────────────────────────────────────
_cupy_probe = _probe_cupy_safe()

if _cupy_probe["ok"]:
    # Le subprocess a confirmé que CuPy fonctionne → import dans le process
    try:
        import cupy as cp
        from cupyx.scipy.ndimage import (
            convolve       as _cp_convolve,
            maximum_filter as _cp_maxfilt,
            minimum_filter as _cp_minfilt,
            gaussian_filter as _cp_gaussian,
            uniform_filter  as _cp_uniform,
        )
        GPU_BACKEND = "cupy"
        GPU_DEVICE  = "GPU CUDA – " + _cupy_probe["device"]
        _CUPY_OK    = True
    except Exception as _e:
        _GPU_FAIL_CUPY = f"Import CuPy échoué après probe réussi : {_e}"
else:
    _err = _cupy_probe.get("error", "")
    if "not_installed" in _err:
        _GPU_FAIL_CUPY = "CuPy non installé  →  pip install cupy-cuda12x"
    elif "nvrtc" in _err.lower() or "DynamicLibNotFoundError" in _err:
        _GPU_FAIL_CUPY = (
            "CuPy : nvrtc*.dll introuvable (CUDA Toolkit incomplet).\n"
            "► Solution rapide : PyTorch embarque ses propres DLLs CUDA :\n"
            "  pip install torch --index-url "
            "https://download.pytorch.org/whl/cu121\n"
            "► Ou installer le CUDA Toolkit complet :\n"
            "  https://developer.nvidia.com/cuda-downloads")
    elif "crash" in _err or "exit" in _err:
        _GPU_FAIL_CUPY = (
            f"CuPy : crash au démarrage ({_err[:200]}).\n"
            "Causes fréquentes :\n"
            "  • Version CuPy incompatible avec la version CUDA installée\n"
            "    (cupy-cuda12x nécessite CUDA 12.x, etc.)\n"
            "  • Driver NVIDIA trop ancien\n"
            "  • Conflit de DLLs CUDA\n"
            "Vérifier : nvcc --version  et  nvidia-smi\n"
            "Puis réinstaller : pip install cupy-cuda12x --force-reinstall")
    else:
        _GPU_FAIL_CUPY = f"CuPy erreur : {_err}"

# ── Tentative PyTorch ─────────────────────────────────────────────────────────
if GPU_BACKEND == "cpu":
    try:
        import torch
        import torch.nn.functional as _TF
        if torch.cuda.is_available():
            _TORCH_DEVICE = torch.device("cuda")
            GPU_BACKEND   = "torch_cuda"
            GPU_DEVICE    = "GPU CUDA – " + torch.cuda.get_device_name(0)
        elif getattr(getattr(torch, "backends", None), "mps", None) and \
                torch.backends.mps.is_available():
            _TORCH_DEVICE = torch.device("mps")
            GPU_BACKEND   = "torch_mps"
            GPU_DEVICE    = "GPU Apple MPS"
        else:
            _TORCH_DEVICE = torch.device("cpu")
            GPU_BACKEND   = "torch_cpu"
            GPU_DEVICE    = "CPU (PyTorch)"
            _GPU_FAIL_TORCH = (
                f"PyTorch v{torch.__version__} installé MAIS "
                f"torch.cuda.is_available()=False.\n"
                f"Réinstaller avec CUDA :\n"
                f"  pip install torch --index-url "
                f"https://download.pytorch.org/whl/cu121")
    except ImportError:
        _GPU_FAIL_TORCH = (
            "PyTorch non installé  →\n"
            "  pip install torch --index-url "
            "https://download.pytorch.org/whl/cu121")
    except Exception as _e:
        _GPU_FAIL_TORCH = f"PyTorch erreur : {_e}"

# ── Résumé CPU ────────────────────────────────────────────────────────────────
if GPU_BACKEND == "cpu":
    if HAS_SCIPY:
        GPU_DEVICE = "CPU (SciPy)"
    elif HAS_CV2:
        GPU_DEVICE = "CPU (OpenCV)"
    else:
        GPU_DEVICE = "CPU (NumPy)"


def gpu_diagnostic() -> str:
    """Retourne un texte complet expliquant l'état du GPU et comment l'activer."""
    sep        = "─" * 46
    gpu_active = GPU_BACKEND in ("cupy", "torch_cuda", "torch_mps")
    lines = [
        sep,
        f"  Backend actif : {GPU_BACKEND}",
        f"  Device        : {GPU_DEVICE}",
        sep, "",
    ]
    if gpu_active:
        lines += ["  ✓  GPU ACTIF — rendu accéléré.", ""]
    else:
        lines += ["  ✗  GPU NON ACTIF — rendu sur CPU.", "", sep, ""]

        if _GPU_FAIL_CUPY:
            lines += ["  [CuPy — probe subprocess]"]
            for l in _GPU_FAIL_CUPY.split("\n"):
                lines.append(f"  {l}")
            lines.append("")

        if _GPU_FAIL_TORCH:
            lines += ["  [PyTorch]"]
            for l in _GPU_FAIL_TORCH.split("\n"):
                lines.append(f"  {l}")
            lines.append("")

        # Conseil commun si crash détecté
        if _GPU_FAIL_CUPY and ("crash" in _GPU_FAIL_CUPY or "exit" in _GPU_FAIL_CUPY):
            lines += [
                sep,
                "  ► Vérifier la compatibilité de version :",
                "    nvcc --version          ← version CUDA installée",
                "    nvidia-smi              ← version driver",
                "    pip show cupy-cuda12x   ← version CuPy",
                "  cupy-cuda12x  → CUDA 12.x",
                "  cupy-cuda11x  → CUDA 11.x",
                "  cupy-cuda102  → CUDA 10.2",
                "",
                "  ► Alternative sans Toolkit (recommandé) :",
                "    pip uninstall cupy-cuda12x",
                "    pip install torch --index-url "
                "https://download.pytorch.org/whl/cu121",
                "",
            ]

        # nvidia-smi pour confirmer la présence GPU
        try:
            import subprocess
            smi = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=name,driver_version,memory.total",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5)
            if smi.returncode == 0 and smi.stdout.strip():
                lines += [sep, "  GPU détecté par nvidia-smi :"]
                for row in smi.stdout.strip().split("\n"):
                    lines.append(f"    {row}")
                lines.append("")
            else:
                lines += [sep,
                          "  nvidia-smi : GPU NVIDIA non détecté.", ""]
        except Exception:
            pass
    lines.append(sep)
    return "\n".join(lines)


# =============================================================================
# Langues  (2 rangées de 7)
# =============================================================================

LANGUAGES = ["fr", "en", "nl", "de", "zh", "ja", "ar",
             "kl", "wa", "ru", "hi", "es", "pt", "bn"]

FLAG_ROWS = [LANGUAGES[:7], LANGUAGES[7:]]


# =============================================================================
# Traductions
# =============================================================================

T = {
    "title": {
        "fr": "── DOF TOOL HQ v9 ──",    "en": "── DOF TOOL HQ v9 ──",
        "nl": "── DOF TOOL HQ v9 ──",    "de": "── DOF TOOL HQ v9 ──",
        "zh": "── DOF 工具 HQ v9 ──",    "ja": "── DOF ツール HQ v9 ──",
        "ar": "── أداة DOF HQ v9 ──",    "kl": "── DOF jan HQ v9 ──",
        "wa": "── OUTIL DOF HQ v9 ──",   "ru": "── ИНСТРУМЕНТ DOF v9 ──",
        "hi": "── DOF टूल HQ v9 ──",     "es": "── HERRAMIENTA DOF v9 ──",
        "pt": "── FERRAMENTA DOF v9 ──", "bn": "── DOF টুল HQ v9 ──",
    },
    "btn_load_rgb": {
        "fr": "📂  Image RGB",           "en": "📂  RGB Image",
        "nl": "📂  RGB-afbeelding",      "de": "📂  RGB-Bild",
        "zh": "📂  RGB 图像",            "ja": "📂  RGB画像を開く",
        "ar": "📂  صورة RGB",           "kl": "📂  RGB nagh beQ",
        "wa": "📂  Imådje RGB",          "ru": "📂  RGB-изображение",
        "hi": "📂  RGB छवि",            "es": "📂  Imagen RGB",
        "pt": "📂  Imagem RGB",          "bn": "📂  RGB ছবি",
    },
    "btn_load_depth": {
        "fr": "📂  Z-Depth 16-bit PNG",  "en": "📂  Z-Depth 16-bit PNG",
        "nl": "📂  Z-Diepte 16-bit PNG", "de": "📂  Z-Tiefe 16-Bit PNG",
        "zh": "📂  Z深度 16位 PNG",      "ja": "📂  Zデプス 16ビット PNG",
        "ar": "📂  عمق Z 16 بت PNG",    "kl": "📂  Z-maS 16-jaj PNG",
        "wa": "📂  Z-Profond 16-bit PNG","ru": "📂  Z-глубина 16-бит PNG",
        "hi": "📂  Z-गहराई 16-बिट PNG", "es": "📂  Z-Profundidad 16-bit PNG",
        "pt": "📂  Z-Profundidade 16bit","bn": "📂  Z-গভীরতা 16-বিট PNG",
    },
    "no_file": {
        "fr": "(aucune)",   "en": "(none)",      "nl": "(geen)",
        "de": "(keine)",    "zh": "（无）",      "ja": "（なし）",
        "ar": "(لا شيء)",  "kl": "(pagh)",      "wa": "(rin)",
        "ru": "(нет)",      "hi": "(कोई नहीं)", "es": "(ninguno)",
        "pt": "(nenhum)",   "bn": "(কিছু নেই)",
    },
    "chk_invert": {
        "fr": "↕  Inverser Z-depth  (0 = loin)",
        "en": "↕  Invert Z-depth  (0 = far)",
        "nl": "↕  Z-diepte omdraaien  (0 = ver)",
        "de": "↕  Z-Tiefe umkehren  (0 = fern)",
        "zh": "↕  反转Z深度  (0 = 远)",
        "ja": "↕  Zデプスを反転  (0 = 遠)",
        "ar": "↕  عكس عمق Z  (0 = بعيد)",
        "kl": "↕  Z-maS nIHghoS  (0 = Hop)",
        "wa": "↕  Rivirer Z-profond  (0 = lon)",
        "ru": "↕  Инверт. Z-глубину  (0 = далеко)",
        "hi": "↕  Z-गहराई पलटें  (0 = दूर)",
        "es": "↕  Invertir Z-prof.  (0 = lejos)",
        "pt": "↕  Inverter Z-prof.  (0 = longe)",
        "bn": "↕  Z-গভীরতা উল্টান  (0 = দূর)",
    },
    "focus_zone_title": {
        "fr": "Zone de netteté  —  glisser les poignées",
        "en": "Focus zone  —  drag the handles",
        "nl": "Scherptezone  —  sleep de grepen",
        "de": "Schärfebereich  —  Griffe ziehen",
        "zh": "对焦区域  —  拖动控制柄",
        "ja": "合焦ゾーン  —  ハンドルをドラッグ",
        "ar": "منطقة التركيز  —  اسحب المقابض",
        "kl": "nIH Daq  —  Hoch vIt",
        "wa": "Zon di clårté  —  glèyî li håve",
        "ru": "Зона резкости  —  тяни ручки",
        "hi": "फ़ोकस क्षेत्र  —  हैंडल खींचें",
        "es": "Zona de enfoque  —  arrastrar asas",
        "pt": "Zona de foco  —  arraste as alças",
        "bn": "ফোকাস অঞ্চল  —  হ্যান্ডেল টানুন",
    },
    "lbl_near": {
        "fr": "Limite proche",   "en": "Near limit",      "nl": "Nabije grens",
        "de": "Nahgrenze",       "zh": "近端界限",         "ja": "手前の境界",
        "ar": "الحد القريب",    "kl": "Qav tuqDaq",      "wa": "Limita di près",
        "ru": "Ближний предел",  "hi": "निकट सीमा",       "es": "Límite cercano",
        "pt": "Limite próximo",  "bn": "কাছের সীমা",
    },
    "lbl_far": {
        "fr": "Limite lointaine","en": "Far limit",       "nl": "Verre grens",
        "de": "Ferngrenze",      "zh": "远端界限",         "ja": "遠方の境界",
        "ar": "الحد البعيد",    "kl": "Qav HopDaq",      "wa": "Limita di lon",
        "ru": "Дальний предел",  "hi": "दूर सीमा",        "es": "Límite lejano",
        "pt": "Limite distante", "bn": "দূরের সীমা",
    },
    "lbl_falloff": {
        "fr": "Transition douce","en": "Smooth falloff",  "nl": "Zachte overgang",
        "de": "Weicher Übergang","zh": "平滑过渡",         "ja": "なめらかな移行",
        "ar": "تلاشٍ سلس",      "kl": "bIQ SaS",         "wa": "Douce passaedje",
        "ru": "Плавный спад",    "hi": "नरम संक्रमण",    "es": "Transición suave",
        "pt": "Transição suave", "bn": "মসৃণ পরিবর্তন",
    },
    "lbl_falloff_near": {
        "fr": "↖ Transition proche", "en": "↖ Near falloff",
        "nl": "↖ Nabije overgang",   "de": "↖ Naher Übergang",
        "zh": "↖ 近端过渡",          "ja": "↖ 手前トランジション",
        "ar": "↖ تلاشٍ قريب",      "kl": "↖ tuq bIQ SaS",
        "wa": "↖ Passaedje proche",  "ru": "↖ Ближн. спад",
        "hi": "↖ निकट संक्रमण",    "es": "↖ Transición cercana",
        "pt": "↖ Transição próxima", "bn": "↖ কাছের পরিবর্তন",
    },
    "lbl_falloff_far": {
        "fr": "↗ Transition lointaine", "en": "↗ Far falloff",
        "nl": "↗ Verre overgang",       "de": "↗ Ferner Übergang",
        "zh": "↗ 远端过渡",             "ja": "↗ 遠方トランジション",
        "ar": "↗ تلاشٍ بعيد",         "kl": "↗ Hop bIQ SaS",
        "wa": "↗ Passaedje lon",        "ru": "↗ Дальн. спад",
        "hi": "↗ दूर संक्रमण",        "es": "↗ Transición lejana",
        "pt": "↗ Transição distante",   "bn": "↗ দূরের পরিবর্তন",
    },
    "lbl_max_blur": {
        "fr": "Flou maximum (px)",        "en": "Maximum blur (px)",
        "nl": "Maximale vervaging (px)",  "de": "Maximale Unschärfe (px)",
        "zh": "最大模糊 (px)",             "ja": "最大ぼかし (px)",
        "ar": "أقصى ضبابية (px)",        "kl": "Qav bIQ (px)",
        "wa": "Mås-imum brouyant (px)",   "ru": "Макс. размытие (px)",
        "hi": "अधिकतम धुंध (px)",        "es": "Desenfoque máx. (px)",
        "pt": "Desfoque máximo (px)",     "bn": "সর্বোচ্চ ঝাপসা (px)",
    },
    "quality_title": {
        "fr": "── Qualité ──",   "en": "── Quality ──",   "nl": "── Kwaliteit ──",
        "de": "── Qualität ──",  "zh": "── 品质 ──",      "ja": "── 品質 ──",
        "ar": "── الجودة ──",   "kl": "── quv ──",       "wa": "── Qualité ──",
        "ru": "── Качество ──",  "hi": "── गुणवत्ता ──", "es": "── Calidad ──",
        "pt": "── Qualidade ──", "bn": "── মান ──",
    },
    "lbl_bokeh_shape": {
        "fr": "Forme du bokeh",  "en": "Bokeh shape",     "nl": "Bokehvorm",
        "de": "Bokeh-Form",      "zh": "虚焦形状",         "ja": "ボケの形状",
        "ar": "شكل البوكيه",    "kl": "bokeh mIw",       "wa": "Fôme do bokeh",
        "ru": "Форма боке",      "hi": "बोके आकार",       "es": "Forma del bokeh",
        "pt": "Forma do bokeh",  "bn": "বোকে আকৃতি",
    },
    "lbl_layers": {
        "fr": "Couches (qualité)",     "en": "Layers (quality)",
        "nl": "Lagen (kwaliteit)",     "de": "Ebenen (Qualität)",
        "zh": "层数（品质）",           "ja": "レイヤー数（品質）",
        "ar": "طبقات (الجودة)",       "kl": "tep (quv)",
        "wa": "Coûtches (qualité)",    "ru": "Слои (качество)",
        "hi": "परतें (गुणवत्ता)",     "es": "Capas (calidad)",
        "pt": "Camadas (qualidade)",   "bn": "স্তর (মান)",
    },
    "preview_title": {
        "fr": "Aperçu",          "en": "Preview",         "nl": "Voorbeeld",
        "de": "Vorschau",        "zh": "预览",             "ja": "プレビュー",
        "ar": "معاينة",          "kl": "leghlaH",         "wa": "Aperwçu",
        "ru": "Предпросмотр",    "hi": "पूर्वावलोकन",    "es": "Vista previa",
        "pt": "Pré-visualização","bn": "পূর্বরূপ",
    },
    "view_original": {
        "fr": "Original",        "en": "Original",        "nl": "Origineel",
        "de": "Original",        "zh": "原图",             "ja": "オリジナル",
        "ar": "الأصلي",          "kl": "wa'DIch",         "wa": "Ôrîdjinål",
        "ru": "Оригинал",        "hi": "मूल",             "es": "Original",
        "pt": "Original",        "bn": "মূল",
    },
    "view_depth": {
        "fr": "Z-Depth",         "en": "Z-Depth",         "nl": "Z-Diepte",
        "de": "Z-Tiefe",         "zh": "Z深度",            "ja": "Zデプス",
        "ar": "عمق Z",           "kl": "Z-maS",           "wa": "Z-Profond",
        "ru": "Z-глубина",       "hi": "Z-गहराई",         "es": "Z-Profundidad",
        "pt": "Z-Profundidade",  "bn": "Z-গভীরতা",
    },
    "view_blur_map": {
        "fr": "Carte de flou",   "en": "Blur map",        "nl": "Vervagingskaart",
        "de": "Unschärfekarte",  "zh": "模糊图",           "ja": "ぼかしマップ",
        "ar": "خريطة الضبابية", "kl": "bIQ rIS",         "wa": "Mape do brouyant",
        "ru": "Карта размытия",  "hi": "धुंध मानचित्र",  "es": "Mapa de desenfoque",
        "pt": "Mapa de desfoque","bn": "ঝাপসা মানচিত্র",
    },
    "view_result": {
        "fr": "Résultat DoF",    "en": "DoF Result",      "nl": "DoF-resultaat",
        "de": "DoF-Ergebnis",    "zh": "景深结果",         "ja": "被写界深度 結果",
        "ar": "نتيجة DoF",       "kl": "DoF chev",        "wa": "Rizultat DoF",
        "ru": "Результат DoF",   "hi": "DoF परिणाम",      "es": "Resultado DoF",
        "pt": "Resultado DoF",   "bn": "DoF ফলাফল",
    },
    "btn_compute": {
        "fr": "▶  Calculer (pleine résolution)",
        "en": "▶  Compute (full resolution)",
        "nl": "▶  Berekenen (volledige resolutie)",
        "de": "▶  Berechnen (volle Auflösung)",
        "zh": "▶  计算（完整分辨率）",
        "ja": "▶  計算（フル解像度）",
        "ar": "▶  احسب (الدقة الكاملة)",
        "kl": "▶  maq (Hoch Qav)",
        "wa": "▶  Cålculer (plinte rizolution)",
        "ru": "▶  Вычислить (полное разрешение)",
        "hi": "▶  गणना करें (पूर्ण रिज़ॉल्यूशन)",
        "es": "▶  Calcular (resolución completa)",
        "pt": "▶  Calcular (resolução completa)",
        "bn": "▶  গণনা করুন (পূর্ণ রেজোলিউশন)",
    },
    "btn_export": {
        "fr": "💾  Exporter résultat",   "en": "💾  Export result",
        "nl": "💾  Resultaat exporteren","de": "💾  Ergebnis exportieren",
        "zh": "💾  导出结果",             "ja": "💾  結果をエクスポート",
        "ar": "💾  تصدير النتيجة",      "kl": "💾  Hoch ngeH",
        "wa": "💾  Espôrter li rizultat","ru": "💾  Экспорт результата",
        "hi": "💾  परिणाम निर्यात करें","es": "💾  Exportar resultado",
        "pt": "💾  Exportar resultado",  "bn": "💾  ফলাফল রপ্তানি করুন",
    },
    "lbl_panel_width": {
        "fr": "Largeur panneau",     "en": "Panel width",
        "nl": "Paneel breedte",      "de": "Panel-Breite",
        "zh": "面板宽度",             "ja": "パネル幅",
        "ar": "عرض اللوحة",          "kl": "ghomDaq Saw",
        "wa": "Lådje do pannea",     "ru": "Ширина панели",
        "hi": "पैनल चौड़ाई",        "es": "Ancho del panel",
        "pt": "Largura do painel",   "bn": "প্যানেল প্রস্থ",
    },
    "status_ready": {
        "fr": "Prêt.",        "en": "Ready.",       "nl": "Klaar.",
        "de": "Bereit.",      "zh": "就绪。",       "ja": "準備完了。",
        "ar": "جاهز.",        "kl": "vItlhutlh.",  "wa": "Prustî.",
        "ru": "Готово.",      "hi": "तैयार।",      "es": "Listo.",
        "pt": "Pronto.",      "bn": "প্রস্তুত।",
    },
    "status_rgb_loaded": {
        "fr": "Image RGB chargée.",      "en": "RGB image loaded.",
        "nl": "RGB-afbeelding geladen.", "de": "RGB-Bild geladen.",
        "zh": "RGB图像已加载。",          "ja": "RGB画像を読み込みました。",
        "ar": "تم تحميل صورة RGB.",     "kl": "RGB nagh beQ tI'lu'.",
        "wa": "Imådje RGB tchèrdjîye.", "ru": "RGB-изображение загружено.",
        "hi": "RGB छवि लोड हुई।",      "es": "Imagen RGB cargada.",
        "pt": "Imagem RGB carregada.",  "bn": "RGB ছবি লোড হয়েছে।",
    },
    "status_depth_loaded": {
        "fr": "Z-Depth chargé.",        "en": "Z-Depth loaded.",
        "nl": "Z-Diepte geladen.",      "de": "Z-Tiefe geladen.",
        "zh": "Z深度已加载。",           "ja": "Zデプスを読み込みました。",
        "ar": "تم تحميل عمق Z.",       "kl": "Z-maS tI'lu'.",
        "wa": "Z-Profond tchèrdjî.",    "ru": "Z-глубина загружена.",
        "hi": "Z-गहराई लोड हुई।",     "es": "Z-Profundidad cargada.",
        "pt": "Z-Profundidade carregada.","bn": "Z-গভীরতা লোড হয়েছে।",
    },
    "status_computing": {
        "fr": "Calcul pleine résolution…",
        "en": "Computing full resolution…",
        "nl": "Volledige resolutie berekenen…",
        "de": "Berechne volle Auflösung…",
        "zh": "正在计算完整分辨率…",
        "ja": "フル解像度で計算中…",
        "ar": "جارٍ الحساب بالدقة الكاملة…",
        "kl": "Hoch Qav maq…",
        "wa": "Cålcul plinte rizolution…",
        "ru": "Вычисление полного разрешения…",
        "hi": "पूर्ण रिज़ॉल्यूशन गणना…",
        "es": "Calculando resolución completa…",
        "pt": "Calculando resolução completa…",
        "bn": "পূর্ণ রেজোলিউশন গণনা…",
    },
    "status_done_full": {
        "fr": "✓ Calcul terminé.",          "en": "✓ Computation done.",
        "nl": "✓ Berekening voltooid.",     "de": "✓ Berechnung abgeschlossen.",
        "zh": "✓ 计算完成。",               "ja": "✓ 計算完了。",
        "ar": "✓ اكتمل الحساب.",           "kl": "✓ maq rIn.",
        "wa": "✓ Cålcul terminé.",          "ru": "✓ Вычисление завершено.",
        "hi": "✓ गणना पूर्ण।",            "es": "✓ Cálculo terminado.",
        "pt": "✓ Cálculo concluído.",       "bn": "✓ গণনা সম্পন্ন।",
    },
    "status_done_preview": {
        "fr": "✓ Aperçu prêt.",            "en": "✓ Preview ready.",
        "nl": "✓ Voorbeeld klaar.",        "de": "✓ Vorschau fertig.",
        "zh": "✓ 预览就绪。",              "ja": "✓ プレビュー完了。",
        "ar": "✓ المعاينة جاهزة.",        "kl": "✓ leghlaH vItlhutlh.",
        "wa": "✓ Aperwçu prustî.",         "ru": "✓ Предпросмотр готов.",
        "hi": "✓ पूर्वावलोकन तैयार।",    "es": "✓ Vista previa lista.",
        "pt": "✓ Pré-visualização pronta.","bn": "✓ পূর্বরূপ প্রস্তুত।",
    },
    "status_starting": {
        "fr": "Démarrage…",   "en": "Starting…",    "nl": "Starten…",
        "de": "Starte…",      "zh": "启动中…",      "ja": "開始中…",
        "ar": "جارٍ البدء…", "kl": "taH…",         "wa": "Démåradje…",
        "ru": "Запуск…",      "hi": "शुरू हो रहा है…","es": "Iniciando…",
        "pt": "Iniciando…",   "bn": "শুরু হচ্ছে…",
    },
    "status_exported": {
        "fr": "Exporté → ",       "en": "Exported → ",     "nl": "Geëxporteerd → ",
        "de": "Exportiert → ",    "zh": "已导出 → ",        "ja": "エクスポート済み → ",
        "ar": "تم التصدير → ",   "kl": "ngeH → ",         "wa": "Espôrté → ",
        "ru": "Экспортировано → ","hi": "निर्यात हुआ → ",  "es": "Exportado → ",
        "pt": "Exportado → ",     "bn": "রপ্তানি হয়েছে → ",
    },
    "progress_bg": {
        "fr": "Fond",      "en": "Background", "nl": "Achtergrond",
        "de": "Hintergrund","zh": "背景",        "ja": "背景",
        "ar": "الخلفية",   "kl": "bIngDaq",    "wa": "Fond",
        "ru": "Фон",       "hi": "पृष्ठभूमि", "es": "Fondo",
        "pt": "Fundo",     "bn": "পটভূমি",
    },
    "progress_fg": {
        "fr": "Avant-plan", "en": "Foreground","nl": "Voorgrond",
        "de": "Vordergrund","zh": "前景",        "ja": "前景",
        "ar": "المقدمة",   "kl": "pemDaq",     "wa": "Divant-plan",
        "ru": "Передний план","hi": "अग्रभूमि", "es": "Primer plano",
        "pt": "Primeiro plano","bn": "অগ্রভূমি",
    },
    "progress_comp": {
        "fr": "Composition",  "en": "Compositing", "nl": "Compositie",
        "de": "Komposition",  "zh": "合成",          "ja": "合成",
        "ar": "التأليف",      "kl": "DaH",          "wa": "Composition",
        "ru": "Композиция",   "hi": "संरचना",       "es": "Composición",
        "pt": "Composição",   "bn": "সংযোজন",
    },
    "dlg_missing_title": {
        "fr": "Données manquantes",    "en": "Missing data",
        "nl": "Ontbrekende gegevens",  "de": "Fehlende Daten",
        "zh": "数据缺失",               "ja": "データ不足",
        "ar": "بيانات مفقودة",        "kl": "De' Hutlh",
        "wa": "Doneyes manquantes",    "ru": "Данные отсутствуют",
        "hi": "डेटा अनुपलब्ध",        "es": "Datos faltantes",
        "pt": "Dados em falta",        "bn": "ডেটা অনুপস্থিত",
    },
    "dlg_missing_body": {
        "fr": "Chargez l'image et le z-depth d'abord.",
        "en": "Please load the image and z-depth first.",
        "nl": "Laad eerst de afbeelding en de z-diepte.",
        "de": "Bitte zuerst Bild und Z-Tiefe laden.",
        "zh": "请先加载图像和Z深度。",
        "ja": "先に画像とZデプスを読み込んでください。",
        "ar": "يرجى تحميل الصورة وعمق Z أولاً.",
        "kl": "wa'DIch nagh beQ je Z-maS yItI'.",
        "wa": "Tchèrdjoz l'imådje et l'Z-profond d'åbôrd.",
        "ru": "Сначала загрузите изображение и Z-глубину.",
        "hi": "पहले छवि और Z-गहराई लोड करें।",
        "es": "Cargue primero la imagen y el z-depth.",
        "pt": "Carregue primeiro a imagem e o z-depth.",
        "bn": "আগে ছবি এবং Z-গভীরতা লোড করুন।",
    },
    "dlg_nothing_title": {
        "fr": "Rien à exporter",        "en": "Nothing to export",
        "nl": "Niets te exporteren",    "de": "Nichts zu exportieren",
        "zh": "无可导出内容",             "ja": "エクスポートするものがありません",
        "ar": "لا شيء للتصدير",        "kl": "pagh ngeH",
        "wa": "Rén à espôrter",         "ru": "Нечего экспортировать",
        "hi": "निर्यात के लिए कुछ नहीं","es": "Nada que exportar",
        "pt": "Nada para exportar",     "bn": "রপ্তানির কিছু নেই",
    },
    "dlg_nothing_body": {
        "fr": "Lancez d'abord le calcul pleine résolution.",
        "en": "Run the full resolution computation first.",
        "nl": "Voer eerst de volledige resolutieberekening uit.",
        "de": "Zuerst die Vollauflösungsberechnung ausführen.",
        "zh": "请先运行完整分辨率计算。",
        "ja": "先にフル解像度の計算を実行してください。",
        "ar": "قم بتشغيل حساب الدقة الكاملة أولاً.",
        "kl": "wa'DIch Hoch Qav maq yImaq.",
        "wa": "Lancoz d'åbôrd li cålcul plinte rizolution.",
        "ru": "Сначала выполните вычисление полного разрешения.",
        "hi": "पहले पूर्ण रिज़ॉल्यूशन गणना चलाएँ।",
        "es": "Ejecute primero el cálculo de resolución completa.",
        "pt": "Execute primeiro o cálculo de resolução completa.",
        "bn": "আগে পূর্ণ রেজোলিউশন গণনা চালান।",
    },
    "dlg_save_title": {
        "fr": "Enregistrer", "en": "Save",       "nl": "Opslaan",
        "de": "Speichern",   "zh": "保存",        "ja": "保存",
        "ar": "حفظ",         "kl": "choH",       "wa": "Sôvegarder",
        "ru": "Сохранить",   "hi": "सहेजें",    "es": "Guardar",
        "pt": "Guardar",     "bn": "সংরক্ষণ করুন",
    },
    "dlg_rgb_title": {
        "fr": "Image RGB",         "en": "RGB Image",       "nl": "RGB-afbeelding",
        "de": "RGB-Bild",          "zh": "RGB图像",          "ja": "RGB画像",
        "ar": "صورة RGB",          "kl": "RGB nagh beQ",    "wa": "Imådje RGB",
        "ru": "RGB-изображение",   "hi": "RGB छवि",         "es": "Imagen RGB",
        "pt": "Imagem RGB",        "bn": "RGB ছবি",
    },
    "dlg_depth_title": {
        "fr": "Z-Depth 16-bit PNG",  "en": "Z-Depth 16-bit PNG",
        "nl": "Z-Diepte 16-bit PNG", "de": "Z-Tiefe 16-Bit PNG",
        "zh": "Z深度 16位 PNG",      "ja": "Zデプス 16ビット PNG",
        "ar": "عمق Z 16 بت PNG",    "kl": "Z-maS 16-jaj PNG",
        "wa": "Z-Profond 16-bit PNG","ru": "Z-глубина 16-бит PNG",
        "hi": "Z-गहराई 16-बिट PNG", "es": "Z-Profundidad 16-bit PNG",
        "pt": "Z-Profundidade 16bit","bn": "Z-গভীরতা 16-বিট PNG",
    },
    "dlg_all_files": {
        "fr": "Tous",              "en": "All files",        "nl": "Alle bestanden",
        "de": "Alle Dateien",      "zh": "所有文件",          "ja": "すべてのファイル",
        "ar": "كل الملفات",       "kl": "Hoch De'",         "wa": "Tos les fitchîs",
        "ru": "Все файлы",         "hi": "सभी फ़ाइलें",    "es": "Todos los archivos",
        "pt": "Todos os ficheiros","bn": "সমস্ত ফাইল",
    },
    "kernel_disk": {
        "fr": "Disque (Bokeh)",    "en": "Disk (Bokeh)",     "nl": "Schijf (Bokeh)",
        "de": "Kreis (Bokeh)",     "zh": "圆形（虚焦）",     "ja": "円形（ボケ）",
        "ar": "قرص (بوكيه)",      "kl": "gho (bokeh)",      "wa": "Disque (Bokeh)",
        "ru": "Диск (боке)",       "hi": "वृत्त (बोके)",    "es": "Disco (Bokeh)",
        "pt": "Disco (Bokeh)",     "bn": "ডিস্ক (বোকে)",
    },
    "kernel_hex": {
        "fr": "Hexagone (Bokeh)",  "en": "Hexagon (Bokeh)",  "nl": "Zeshoek (Bokeh)",
        "de": "Hexagon (Bokeh)",   "zh": "六边形（虚焦）",   "ja": "六角形（ボケ）",
        "ar": "سداسي (بوكيه)",    "kl": "jav mIw (bokeh)",  "wa": "Hexagone (Bokeh)",
        "ru": "Шестиугольник (боке)","hi": "षट्भुज (बोके)", "es": "Hexágono (Bokeh)",
        "pt": "Hexágono (Bokeh)",  "bn": "ষড়ভুজ (বোকে)",
    },
    "kernel_gauss": {
        "fr": "Gaussien (doux)",   "en": "Gaussian (soft)",  "nl": "Gaussisch (zacht)",
        "de": "Gaußförmig (weich)","zh": "高斯（柔和）",     "ja": "ガウシアン（ソフト）",
        "ar": "غاوسي (ناعم)",     "kl": "SaS (bIQ)",        "wa": "Gaussyin (doûs)",
        "ru": "Гауссиан (мягкий)", "hi": "गाउसियन (नरम)",  "es": "Gaussiano (suave)",
        "pt": "Gaussiano (suave)", "bn": "গাউসিয়ান (নরম)",
    },
    "lbl_render_mode": {
        "fr": "Mode de rendu",        "en": "Render mode",
        "nl": "Rendermodus",           "de": "Render-Modus",
        "zh": "渲染模式",              "ja": "レンダーモード",
        "ar": "وضع العرض",            "kl": "mIw cha'",
        "wa": "Mode di rindou",        "ru": "Режим рендера",
        "hi": "रेंडर मोड",            "es": "Modo de render",
        "pt": "Modo de renderização",  "bn": "রেন্ডার মোড",
    },
    "render_two_layers": {
        "fr": "Deux couches (scènes)", "en": "Two layers (scenes)",
        "nl": "Twee lagen (scènes)",   "de": "Zwei Ebenen (Szenen)",
        "zh": "双层（场景）",           "ja": "2レイヤー（シーン）",
        "ar": "طبقتان (مشاهد)",       "kl": "cha' tep (mIw)",
        "wa": "Deûs coûtches (sênes)", "ru": "Два слоя (сцены)",
        "hi": "दो परतें (दृश्य)",      "es": "Dos capas (escenas)",
        "pt": "Duas camadas (cenas)",  "bn": "দুটি স্তর (দৃশ্য)",
    },
    "render_single": {
        "fr": "Couche unique (fractales)","en": "Single layer (fractals)",
        "nl": "Enkele laag (fractals)",   "de": "Einzelebene (Fraktale)",
        "zh": "单层（分形）",              "ja": "単一レイヤー（フラクタル）",
        "ar": "طبقة واحدة (فراكتال)",    "kl": "wa' tep (fractals)",
        "wa": "Seûle coûtche (fractåles)","ru": "Один слой (фракталы)",
        "hi": "एक परत (फ़्रैक्टल)",       "es": "Capa única (fractales)",
        "pt": "Camada única (fractais)",  "bn": "একটি স্তর (ফ্র্যাক্টাল)",
    },
    "btn_gpu_diag": {
        "fr": "🔍 Diagnostic GPU",  "en": "🔍 GPU Diagnostic",
        "nl": "🔍 GPU Diagnose",    "de": "🔍 GPU Diagnose",
        "zh": "🔍 GPU 诊断",        "ja": "🔍 GPU 診断",
        "ar": "🔍 تشخيص GPU",      "kl": "🔍 GPU Soj",
        "wa": "🔍 Diagnose GPU",    "ru": "🔍 Диагностика GPU",
        "hi": "🔍 GPU जाँच",       "es": "🔍 Diagnóstico GPU",
        "pt": "🔍 Diagnóstico GPU", "bn": "🔍 GPU নির্ণয়",
    },
    "dlg_gpu_title": {
        "fr": "Diagnostic GPU",  "en": "GPU Diagnostic",
        "nl": "GPU Diagnose",    "de": "GPU Diagnose",
        "zh": "GPU 诊断",        "ja": "GPU 診断",
        "ar": "تشخيص GPU",      "kl": "GPU Soj",
        "wa": "Diagnose GPU",    "ru": "Диагностика GPU",
        "hi": "GPU जाँच",       "es": "Diagnóstico GPU",
        "pt": "Diagnóstico GPU", "bn": "GPU নির্ণয়",
    },
    "passes_suffix": {
        "fr": " passes",   "en": " layers",   "nl": " lagen",
        "de": " Ebenen",   "zh": " 层",       "ja": " レイヤー",
        "ar": " طبقات",   "kl": " tep",      "wa": " passêyes",
        "ru": " слоёв",    "hi": " परतें",   "es": " capas",
        "pt": " camadas",  "bn": " স্তর",
    },
    "focus_zone_net": {
        "fr": "NET",     "en": "SHARP",   "nl": "SCHERP",
        "de": "SCHARF",  "zh": "清晰",    "ja": "合焦",
        "ar": "حاد",     "kl": "nIH",    "wa": "NET",
        "ru": "РЕЗКО",   "hi": "तीक्ष्ण","es": "NÍTIDO",
        "pt": "NÍTIDO",  "bn": "স্পষ্ট",
    },
    "chk_overlay": {
        "fr": "🔴🟢🔵  Afficher zones DoF",
        "en": "🔴🟢🔵  Show DoF zones",
        "nl": "🔴🟢🔵  DoF-zones tonen",
        "de": "🔴🟢🔵  DoF-Zonen anzeigen",
        "zh": "🔴🟢🔵  显示景深区域",
        "ja": "🔴🟢🔵  DoFゾーン表示",
        "ar": "🔴🟢🔵  إظهار مناطق DoF",
        "kl": "🔴🟢🔵  DoF Daq legh",
        "wa": "🔴🟢🔵  Môtrer zones DoF",
        "ru": "🔴🟢🔵  Показать зоны DoF",
        "hi": "🔴🟢🔵  DoF क्षेत्र दिखाएँ",
        "es": "🔴🟢🔵  Mostrar zonas DoF",
        "pt": "🔴🟢🔵  Mostrar zonas DoF",
        "bn": "🔴🟢🔵  DoF অঞ্চল দেখান",
    },
    "lbl_overlay_opacity": {
        "fr": "Opacité overlay",      "en": "Overlay opacity",
        "nl": "Overlay-opaciteit",    "de": "Overlay-Deckkraft",
        "zh": "叠加透明度",            "ja": "オーバーレイ不透明度",
        "ar": "شفافية التراكب",       "kl": "bIQ Qav",
        "wa": "Opacité overlay",      "ru": "Прозрачность оверлея",
        "hi": "ओवरले अपारदर्शिता",   "es": "Opacidad overlay",
        "pt": "Opacidade overlay",    "bn": "ওভারলে অস্বচ্ছতা",
    },
    "overlay_legend_far":   {
        "fr": "■ Flou loin",  "en": "■ Far blur",   "nl": "■ Ver onscherp",
        "de": "■ Fern-unsch.","zh": "■ 远景虚焦",   "ja": "■ 遠景ぼかし",
        "ar": "■ ضبابية بعيد","kl": "■ Hop bIQ",   "wa": "■ Flou lon",
        "ru": "■ Дальн. расм","hi": "■ दूर का धुंध","es": "■ Desenfoque lejos",
        "pt": "■ Desfoque lon","bn": "■ দূর ঝাপসা",
    },
    "overlay_legend_sharp": {
        "fr": "■ Net",        "en": "■ Sharp",      "nl": "■ Scherp",
        "de": "■ Scharf",     "zh": "■ 清晰",       "ja": "■ 合焦",
        "ar": "■ حاد",        "kl": "■ nIH",        "wa": "■ Net",
        "ru": "■ Резко",      "hi": "■ तीक्ष्ण",   "es": "■ Nítido",
        "pt": "■ Nítido",     "bn": "■ স্পষ্ট",
    },
    "overlay_legend_near":  {
        "fr": "■ Flou près",  "en": "■ Near blur",  "nl": "■ Dichtbij onscherp",
        "de": "■ Nah-unsch.", "zh": "■ 近景虚焦",   "ja": "■ 近景ぼかし",
        "ar": "■ ضبابية قريب","kl": "■ tuq bIQ",   "wa": "■ Flou près",
        "ru": "■ Ближн. расм","hi": "■ पास का धुंध","es": "■ Desenfoque cerca",
        "pt": "■ Desfoque próx","bn": "■ কাছের ঝাপসা",
    },
}


def t(key: str, lang: str) -> str:
    d = T.get(key, {})
    return d.get(lang, d.get("en", key))


# =============================================================================
# Noyaux Bokeh
# =============================================================================

def make_disk_kernel(radius: float) -> np.ndarray:
    r = max(int(math.ceil(radius)), 1)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    mask = (x * x + y * y) <= radius * radius
    k = mask.astype(np.float32)
    k /= k.sum()
    return k


def make_hex_kernel(radius: float) -> np.ndarray:
    r = max(int(math.ceil(radius)), 1)
    k = np.zeros((2*r+1, 2*r+1), dtype=np.float32)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            x, y = dx / radius, dy / radius
            if (abs(x) <= 1.0 and abs(y) <= 1.0 and
                    abs(x + y) <= 1.0 / (math.sqrt(3) / 2) and
                    abs(x - y) <= 1.0 / (math.sqrt(3) / 2)):
                k[dy + r, dx + r] = 1.0
    s = k.sum()
    return k / s if s > 0 else make_disk_kernel(radius)


def make_gaussian_kernel(radius: float) -> np.ndarray:
    r = max(int(math.ceil(radius * 2)), 1)
    sigma = radius / 2.0
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    k = np.exp(-(x * x + y * y) / (2 * sigma * sigma)).astype(np.float32)
    k /= k.sum()
    return k


KERNEL_FN   = {"disk": make_disk_kernel, "hex": make_hex_kernel, "gauss": make_gaussian_kernel}
KERNEL_KEYS = ["disk", "hex", "gauss"]
KERNEL_T    = ["kernel_disk", "kernel_hex", "kernel_gauss"]


def _cupy_failed(exc: Exception) -> None:
    """Appelé si une opération CuPy échoue à l'exécution.
    Désactive CuPy pour la session et mémorise la raison."""
    global GPU_BACKEND, GPU_DEVICE, _CUPY_OK, _GPU_FAIL_CUPY
    _CUPY_OK        = False
    GPU_BACKEND     = "cpu"
    GPU_DEVICE      = "CPU (SciPy) [CuPy désactivé]"
    _GPU_FAIL_CUPY  = f"Erreur CuPy à l'exécution : {exc}\n{_GPU_FAIL_CUPY}"


def apply_kernel(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve img (HxWx3 float32) avec kernel (KxK float32) — GPU si dispo."""
    if GPU_BACKEND == "cupy" and _CUPY_OK:
        try:
            img_cp = cp.asarray(img)
            k_cp   = cp.asarray(kernel)
            result = cp.stack([
                _cp_convolve(img_cp[:, :, c], k_cp, mode="reflect")
                for c in range(3)
            ], axis=2)
            return cp.asnumpy(result)
        except Exception as _e:
            _cupy_failed(_e)

    elif GPU_BACKEND in ("torch_cuda", "torch_mps", "torch_cpu"):
        kH, kW = kernel.shape
        pH, pW = kH // 2, kW // 2
        t = (torch.from_numpy(img.transpose(2, 0, 1))
             .unsqueeze(0).float().to(_TORCH_DEVICE))
        t = _TF.pad(t, (pW, pW, pH, pH), mode="reflect")
        k = (torch.from_numpy(kernel).float()
             .unsqueeze(0).unsqueeze(0)
             .expand(3, 1, kH, kW).contiguous().to(_TORCH_DEVICE))
        out = _TF.conv2d(t, k, groups=3)
        return out.squeeze(0).permute(1, 2, 0).cpu().numpy()

    elif HAS_SCIPY:
        return np.stack([
            convolve(img[:, :, c].astype(np.float32), kernel, mode="reflect")
            for c in range(3)
        ], axis=2)

    elif HAS_CV2:
        return cv2.filter2D(img.astype(np.float32), -1, kernel,
                            borderType=cv2.BORDER_REFLECT)

    else:
        from PIL import ImageFilter
        r = int((kernel.shape[0] - 1) / 4)
        pil = Image.fromarray(img.astype(np.uint8))
        return np.array(pil.filter(ImageFilter.GaussianBlur(radius=max(r, 1))),
                        dtype=np.float32)


# =============================================================================
# Chargement depth
# =============================================================================

def load_depth(path: str, invert: bool = False) -> np.ndarray:
    """
    Chargement robuste d'une depth map :
    • 16-bit PNG (mode I;16, I;16B, I) → précision complète 65 535 niveaux
    • 8-bit PNG/JPG (mode L)           → 255 niveaux
    • RGB/RGBA                          → converti en niveaux de gris
    Retourne un float32 normalisé [0.0, 1.0].
    """
    # ── Priorité 1 : OpenCV lit les PNG 16-bit nativement ────────────────────
    if HAS_CV2:
        arr_cv = cv2.imread(path,
                            cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        if arr_cv is not None:
            if arr_cv.ndim == 3:
                arr_cv = arr_cv[:, :, 0].copy()
            arr = arr_cv.astype(np.float32)
            arr -= arr.min()
            if arr.max() > 0:
                arr /= arr.max()
            return 1.0 - arr if invert else arr

    # ── Priorité 2 : PIL avec gestion explicite de tous les modes ─────────────
    img = Image.open(path)

    if img.mode in ("I", "F"):
        # Mode I = int32 (PIL stocke les 16-bit en int32 sur certaines versions)
        # Mode F = float32
        arr = np.array(img, dtype=np.float32)

    elif img.mode in ("I;16", "I;16B", "I;16S", "I;16BS"):
        # Mode 16-bit brut : np.array donne uint16 directement
        arr = np.array(img, dtype=np.float32)

    elif img.mode == "L":
        arr = np.array(img, dtype=np.float32)          # 8-bit, 0-255

    elif img.mode in ("RGB", "RGBA", "LA"):
        arr = np.array(img.convert("L"), dtype=np.float32)

    else:
        # Dernier recours : forcer la conversion en niveaux de gris
        arr = np.array(img.convert("L"), dtype=np.float32)

    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()
    return 1.0 - arr if invert else arr


# =============================================================================
# Carte de rayon de flou
# =============================================================================

def compute_blur_radius_map(depth, focus_near, focus_far, max_blur,
                            falloff_near, falloff_far=None):
    """
    Calcule la carte de rayon de flou.
    falloff_near : largeur de transition côté proche  (avant focus_near)
    falloff_far  : largeur de transition côté lointain (après focus_far)
                   Si None, identique à falloff_near (comportement legacy).
    """
    if falloff_far is None:
        falloff_far = falloff_near

    d, blur = depth.copy(), np.zeros_like(depth)

    # ── Côté proche (d < focus_near) ─────────────────────────────────────────
    sn = (d < focus_near) & (d >= focus_near - falloff_near)
    if sn.any() and falloff_near > 1e-6:
        tv = np.clip((focus_near - d[sn]) / falloff_near, 0.0, 1.0)
        blur[sn] = tv * tv * (3.0 - 2.0 * tv)
    hn = d < (focus_near - falloff_near)
    if hn.any() and (focus_near - falloff_near) > 1e-6:
        tv = np.clip(
            (focus_near - falloff_near - d[hn]) / max(focus_near - falloff_near, 1e-6),
            0.0, 1.0)
        blur[hn] = 1.0 + tv

    # ── Côté lointain (d > focus_far) ────────────────────────────────────────
    sf = (d > focus_far) & (d <= focus_far + falloff_far)
    if sf.any() and falloff_far > 1e-6:
        tv = np.clip((d[sf] - focus_far) / falloff_far, 0.0, 1.0)
        blur[sf] = tv * tv * (3.0 - 2.0 * tv)
    hf = d > (focus_far + falloff_far)
    rf  = 1.0 - (focus_far + falloff_far)
    if hf.any() and rf > 1e-6:
        tv = np.clip((d[hf] - focus_far - falloff_far) / rf, 0.0, 1.0)
        blur[hf] = 1.0 + tv

    blur = np.clip(blur, 0.0, None)
    if blur.max() > 0:
        blur = blur / blur.max() * max_blur
    return blur


# =============================================================================
# Helpers masques
# =============================================================================

def _dilate(mask: np.ndarray, radius: float) -> np.ndarray:
    r = max(int(math.ceil(radius)), 1)
    ks = 2 * r + 1
    if GPU_BACKEND == "cupy" and _CUPY_OK:
        try:
            return cp.asnumpy(_cp_maxfilt(cp.asarray(mask), size=ks))
        except Exception as _e:
            _cupy_failed(_e)
    elif GPU_BACKEND in ("torch_cuda", "torch_mps", "torch_cpu"):
        t = (torch.from_numpy(mask).float()
             .unsqueeze(0).unsqueeze(0).to(_TORCH_DEVICE))
        t = _TF.pad(t, (r, r, r, r), mode="replicate")
        t = _TF.max_pool2d(t, kernel_size=ks, stride=1, padding=0)
        return t.squeeze().cpu().numpy()
    elif HAS_SCIPY:
        return maximum_filter(mask, size=ks)
    elif HAS_CV2:
        k = np.ones((ks, ks), dtype=np.uint8)
        return cv2.dilate((mask * 255).astype(np.uint8), k).astype(np.float32) / 255.0
    else:
        return mask


def _erode(mask: np.ndarray, radius: float) -> np.ndarray:
    r = max(int(math.ceil(radius)), 1)
    ks = 2 * r + 1
    if GPU_BACKEND == "cupy" and _CUPY_OK:
        try:
            return cp.asnumpy(_cp_minfilt(cp.asarray(mask), size=ks))
        except Exception as _e:
            _cupy_failed(_e)
    elif GPU_BACKEND in ("torch_cuda", "torch_mps", "torch_cpu"):
        t = (torch.from_numpy(mask).float()
             .unsqueeze(0).unsqueeze(0).to(_TORCH_DEVICE))
        t = _TF.pad(t, (r, r, r, r), mode="replicate")
        t = -_TF.max_pool2d(-t, kernel_size=ks, stride=1, padding=0)
        return t.squeeze().cpu().numpy()
    elif HAS_SCIPY:
        return minimum_filter(mask, size=ks)
    elif HAS_CV2:
        k = np.ones((ks, ks), dtype=np.uint8)
        return cv2.erode((mask * 255).astype(np.uint8), k).astype(np.float32) / 255.0
    else:
        return mask


def _smooth(mask: np.ndarray, sigma: float) -> np.ndarray:
    if GPU_BACKEND == "cupy" and _CUPY_OK:
        try:
            return cp.asnumpy(_cp_gaussian(cp.asarray(mask), sigma=sigma))
        except Exception as _e:
            _cupy_failed(_e)
    elif GPU_BACKEND in ("torch_cuda", "torch_mps", "torch_cpu"):
        ks  = max(int(math.ceil(sigma * 3)) * 2 + 1, 3)
        dev = _TORCH_DEVICE
        x   = torch.arange(ks, dtype=torch.float32, device=dev) - ks // 2
        gk  = torch.exp(-x * x / (2.0 * sigma * sigma))
        gk  = gk / gk.sum()
        gk2 = (gk.unsqueeze(0) * gk.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        t   = (torch.from_numpy(mask).float()
               .unsqueeze(0).unsqueeze(0).to(dev))
        t   = _TF.pad(t, (ks // 2, ks // 2, ks // 2, ks // 2), mode="reflect")
        out = _TF.conv2d(t, gk2)
        return out.squeeze().cpu().numpy()
    elif HAS_SCIPY:
        return gaussian_filter(mask, sigma=sigma)
    elif HAS_CV2:
        ks = max(int(math.ceil(sigma * 3)) * 2 + 1, 3)
        return cv2.GaussianBlur(mask, (ks, ks), sigma)
    else:
        return mask


# =============================================================================
# Overlay zones DoF
# =============================================================================

def make_zone_overlay(depth, focus_near, focus_far, falloff_near, falloff_far=None):
    if falloff_far is None:
        falloff_far = falloff_near
    h, w = depth.shape
    out  = np.zeros((h, w, 4), dtype=np.uint8)
    d    = depth.copy()

    in_focus = (d >= focus_near) & (d <= focus_far)
    out[in_focus, 0] = 60
    out[in_focus, 1] = 120
    out[in_focus, 2] = 255
    out[in_focus, 3] = 80

    trans_near = (d < focus_near) & (d >= max(focus_near - falloff_near, 0.0))
    if trans_near.any() and falloff_near > 1e-6:
        tv = np.clip((focus_near - d[trans_near]) / falloff_near, 0.0, 1.0)
        tv = tv * tv * (3.0 - 2.0 * tv)
        out[trans_near, 1] = (200 * tv).astype(np.uint8)
        out[trans_near, 3] = (160 * tv).astype(np.uint8)

    hard_near = d < max(focus_near - falloff_near, 0.0)
    if hard_near.any():
        fn  = focus_near - falloff_near
        tv  = np.clip((fn - d[hard_near]) / max(fn, 1e-6), 0.0, 1.0)
        intensity = 0.5 + 0.5 * tv
        out[hard_near, 1] = np.clip(intensity * 255, 0, 255).astype(np.uint8)
        out[hard_near, 3] = np.clip(intensity * 200, 80, 200).astype(np.uint8)

    trans_far = (d > focus_far) & (d <= min(focus_far + falloff_far, 1.0))
    if trans_far.any() and falloff_far > 1e-6:
        tv = np.clip((d[trans_far] - focus_far) / falloff_far, 0.0, 1.0)
        tv = tv * tv * (3.0 - 2.0 * tv)
        out[trans_far, 0] = (220 * tv).astype(np.uint8)
        out[trans_far, 3] = (160 * tv).astype(np.uint8)

    hard_far = d > min(focus_far + falloff_far, 1.0)
    if hard_far.any():
        ff = focus_far + falloff_far
        tv = np.clip((d[hard_far] - ff) / max(1.0 - ff, 1e-6), 0.0, 1.0)
        intensity = 0.5 + 0.5 * tv
        out[hard_far, 0] = np.clip(intensity * 255, 0, 255).astype(np.uint8)
        out[hard_far, 3] = np.clip(intensity * 200, 80, 200).astype(np.uint8)

    return out


def composite_overlay(base_rgb, overlay_rgba, opacity=0.5):
    base  = base_rgb.astype(np.float32)
    ov_c  = overlay_rgba[:, :, :3].astype(np.float32)
    ov_a  = overlay_rgba[:, :, 3].astype(np.float32) / 255.0 * opacity
    a     = ov_a[:, :, np.newaxis]
    return np.clip(base * (1.0 - a) + ov_c * a, 0, 255).astype(np.uint8)


# =============================================================================
# Moteur DoF — rendu en couches séparées (v9 : séparation focus-aware)
# =============================================================================

def render_dof(rgb, blur_map, depth, kernel_key="disk", steps=16,
               bleed_correction=True, progress_cb=None, lang="fr",
               focus_near=0.35, focus_far=0.65):
    """
    Rendu DOF en couches séparées  —  v9

    Corrections v9 :
    ─ depth_mid basé sur le CENTRE DE LA ZONE DE FOCUS au lieu de la médiane
      globale → la séparation near/far suit l'intention de l'utilisateur.
    ─ Couches assignées dans le bon ordre de profondeur :
        • bg_layer = objets LOIN de la caméra (depth faible) → composité EN PREMIER
        • fg_layer = objets PROCHES de la caméra (depth élevé) → composité EN DERNIER
      Avant : le flou du ciel était composité PAR-DESSUS le terrain → halo.
    ─ PRE-FILL anti-contamination (v8)
    ─ Masques de protection proportionnels au rayon de flou max (v8)
    """
    h, w  = rgb.shape[:2]
    max_r = blur_map.max()
    rgb_f = rgb.astype(np.float32)
    maker = KERNEL_FN.get(kernel_key, make_disk_kernel)

    if max_r < 0.5:
        return rgb.copy()

    # ── Lissage du depth ──────────────────────────────────────────────────────
    depth_s = depth
    if GPU_BACKEND == "cupy" and _CUPY_OK:
        try:
            depth_s = cp.asnumpy(_cp_uniform(cp.asarray(depth), size=3))
        except Exception as _e:
            _cupy_failed(_e)
    if depth_s is depth:   # fallback si cupy a échoué ou non dispo
        if GPU_BACKEND in ("torch_cuda", "torch_mps", "torch_cpu"):
            depth_s = uniform_filter(depth, size=3) if HAS_SCIPY else (
                      cv2.blur(depth, (3, 3)) if HAS_CV2 else depth)
        elif HAS_SCIPY:
            depth_s = uniform_filter(depth, size=3)
        elif HAS_CV2:
            depth_s = cv2.blur(depth, (3, 3))

    # ── v9 : seuil basé sur le centre de la zone de focus ─────────────────────
    depth_mid = (focus_near + focus_far) / 2.0

    # ── Séparation — DOUCE (sigmoid) ──────────────────────────────────────────
    _d_range     = max(float(depth_s.max() - depth_s.min()), 1e-6)
    _softness    = _d_range * 0.08
    # _depth_blend ≈ 1.0 pour depth élevé (proche caméra)
    # _depth_blend ≈ 0.0 pour depth faible (loin de la caméra)
    _depth_blend = 1.0 / (1.0 + np.exp(-(depth_s - depth_mid) / _softness))
    blurred_mask = (blur_map >= 0.5).astype(np.float32)

    # ── v9 : couches dans le bon ordre de profondeur ──────────────────────────
    # bg = loin de la caméra (depth faible, ciel) → composité en 1er
    # fg = proche de la caméra (depth élevé, terrain) → composité en dernier
    bg_pixels = (1.0 - _depth_blend) * blurred_mask
    fg_pixels = _depth_blend          * blurred_mask

    # ── Masque net ────────────────────────────────────────────────────────────
    sharp_mask_raw = (blur_map < 0.5).astype(np.float32)

    # ── Protection proportionnelle, PLAFONNÉE pour les gros flous ───────────
    # Sans plafond, blur=72 → protection 25px + prefill 43px → image lavée.
    _PROT_CAP  = 8.0    # px max de dilatation du masque de protection
    _SIGMA_CAP = 5.0    # px max de sigma pour le lissage des masques
    _FILL_CAP  = 12.0   # px max de rayon du noyau de pre-fill

    prot_radius  = min(max(max_r * 0.35, 2.0), _PROT_CAP)
    prot_sigma   = min(max(max_r * 0.25, 1.5), _SIGMA_CAP)

    sharp_protection = _smooth(_dilate(sharp_mask_raw, prot_radius),
                               sigma=prot_sigma)
    sharp_protection = np.clip(sharp_protection, 0.0, 1.0)

    # bg bloqué par les pixels nets ET par le premier plan (fg)
    fg_solid       = (_depth_blend >= 0.5).astype(np.float32)
    fg_protection  = _smooth(_dilate(fg_solid,
                                     min(max(max_r * 0.30, 2.0), _PROT_CAP)),
                             sigma=prot_sigma)
    fg_protection  = np.clip(fg_protection, 0.0, 1.0)
    bg_allowed     = np.clip(1.0 - sharp_protection - fg_protection, 0.0, 1.0)

    # fg bloqué par les pixels nets uniquement (il est devant)
    fg_allowed     = 1.0 - sharp_protection

    # ── PRE-FILL — empêche la contamination croisée des couleurs ──────────────
    bg_source = np.clip(
        bg_pixels + (1.0 - blurred_mask) * (1.0 - _depth_blend), 0.0, 1.0)
    fg_source = np.clip(
        fg_pixels + (1.0 - blurred_mask) * _depth_blend, 0.0, 1.0)

    fill_kernel = maker(min(max(max_r * 0.6, 3.0), _FILL_CAP))

    rgb_for_bg = rgb_f.copy()
    bg_src_3d  = bg_source[:, :, np.newaxis]
    for _ in range(3):
        filled     = apply_kernel(rgb_for_bg, fill_kernel)
        rgb_for_bg = rgb_for_bg * bg_src_3d + filled * (1.0 - bg_src_3d)

    rgb_for_fg = rgb_f.copy()
    fg_src_3d  = fg_source[:, :, np.newaxis]
    for _ in range(3):
        filled     = apply_kernel(rgb_for_fg, fill_kernel)
        rgb_for_fg = rgb_for_fg * fg_src_3d + filled * (1.0 - fg_src_3d)

    # ── Pré-calcul des noyaux ─────────────────────────────────────────────────
    radii   = np.linspace(1.0, max_r, max(steps, 4))
    kernels = [maker(float(r)) for r in radii]
    _layer_sigma = max_r / len(radii) * 1.2

    # Nombre total d'étapes pour la barre de progression :
    # pre-fill (1) + BG layers + FG layers + composition (1)
    total = 1 + len(radii) * 2 + 1
    done  = [0]

    def _progress(label):
        done[0] += 1
        if progress_cb:
            progress_cb(done[0], total, label)

    _progress("Pre-fill")

    # ── Couche BG (loin de la caméra : ciel, arrière-plan) ──────────────────
    bg_accum  = np.zeros_like(rgb_f)
    bg_weight = np.zeros((h, w), dtype=np.float32)

    for i, r in enumerate(radii):
        gw      = np.exp(-0.5 * ((blur_map - r) / _layer_sigma) ** 2)
        layer_w = gw * bg_pixels
        if layer_w.sum() < 1e-3:
            _progress(f"{t('progress_bg', lang)} {i + 1}/{len(radii)}")
            continue
        dilated    = _dilate(layer_w, r) * bg_allowed
        blurred    = apply_kernel(rgb_for_bg, kernels[i])
        bg_accum  += blurred * dilated[:, :, np.newaxis]
        bg_weight += dilated
        _progress(f"{t('progress_bg', lang)} {i + 1}/{len(radii)}")

    mask_no_bg = (bg_weight < 1e-6)[:, :, np.newaxis]
    bg_layer   = np.clip(
        bg_accum / np.maximum(bg_weight, 1e-6)[:, :, np.newaxis], 0.0, 255.0)
    bg_layer   = np.where(mask_no_bg, rgb_f, bg_layer)

    # ── Couche FG (proche de la caméra : terrain, premier plan) ─────────────
    fg_accum  = np.zeros_like(rgb_f)
    fg_weight = np.zeros((h, w), dtype=np.float32)
    fg_alpha  = np.zeros((h, w), dtype=np.float32)

    for i, r in enumerate(radii):
        gw      = np.exp(-0.5 * ((blur_map - r) / _layer_sigma) ** 2)
        layer_w = gw * fg_pixels
        if layer_w.sum() < 1e-3:
            _progress(f"{t('progress_fg', lang)} {i + 1}/{len(radii)}")
            continue
        dilated    = _dilate(layer_w, r) * fg_allowed
        blurred    = apply_kernel(rgb_for_fg, kernels[i])
        fg_accum  += blurred * dilated[:, :, np.newaxis]
        fg_weight += dilated
        fg_alpha   = np.maximum(fg_alpha, dilated)
        _progress(f"{t('progress_fg', lang)} {i + 1}/{len(radii)}")

    mask_no_fg = (fg_weight < 1e-6)[:, :, np.newaxis]
    fg_layer   = np.clip(
        fg_accum / np.maximum(fg_weight, 1e-6)[:, :, np.newaxis], 0.0, 255.0)
    fg_layer   = np.where(mask_no_fg, rgb_f, fg_layer)

    # ── Couche NETTE ──────────────────────────────────────────────────────────
    sharp_eroded = _erode(sharp_mask_raw, min(max(max_r * 0.05, 0.5), 1.5))
    sharp_alpha  = _smooth(sharp_eroded, sigma=min(max(max_r * 0.04, 0.5), 1.5))
    sharp_alpha  = np.clip(sharp_alpha, 0.0, 1.0)

    # ── Composition back-to-front ─────────────────────────────────────────────
    # 1. BG (arrière-plan lointain) en premier
    result = bg_layer.copy()

    # 2. Zone nette par-dessus
    result = (result * (1.0 - sharp_protection[:, :, np.newaxis])
              + rgb_f  *  sharp_protection[:, :, np.newaxis])
    result = (result * (1.0 - sharp_alpha[:, :, np.newaxis])
              + rgb_f  *  sharp_alpha[:, :, np.newaxis])

    # 3. FG (premier plan proche) par-dessus tout
    fg_alpha_s = _smooth(fg_alpha, sigma=1.0)
    fg_alpha_s = np.clip(fg_alpha_s, 0.0, 1.0)
    result = (result * (1.0 - fg_alpha_s[:, :, np.newaxis])
              + fg_layer * fg_alpha_s[:, :, np.newaxis])

    _progress(t("progress_comp", lang))
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


# =============================================================================
# Moteur DoF — rendu couche unique (fractales / scènes complexes)
# =============================================================================

def render_dof_single(rgb, blur_map, depth, kernel_key="disk", steps=16,
                      progress_cb=None, lang="fr", **_kw):
    """
    Rendu DOF couche unique — idéal pour les fractales.

    Pas de séparation near/far.  Pour chaque rayon, l'image entière est floutée
    et chaque pixel est pondéré selon la correspondance blur_map ↔ rayon.
    Le masque net est composité par-dessus à la fin.

    Avantages : zéro artefact de couches, transitions parfaitement lisses.
    Inconvénient : pas de gestion d'occlusion (le flou d'arrière-plan peut
    saigner légèrement sur un bord net de premier plan).
    """
    h, w  = rgb.shape[:2]
    max_r = blur_map.max()
    rgb_f = rgb.astype(np.float32)
    maker = KERNEL_FN.get(kernel_key, make_disk_kernel)

    if max_r < 0.5:
        return rgb.copy()

    sharp_mask   = (blur_map < 0.5).astype(np.float32)
    blurred_mask = 1.0 - sharp_mask

    radii   = np.linspace(1.0, max_r, max(steps, 4))
    kernels = [maker(float(r)) for r in radii]
    _layer_sigma = max_r / len(radii) * 1.2

    total = len(radii) + 1
    done  = [0]

    def _progress(label):
        done[0] += 1
        if progress_cb:
            progress_cb(done[0], total, label)

    # ── Accumulateur unique ────────────────────────────────────────────────────
    accum  = np.zeros_like(rgb_f)
    weight = np.zeros((h, w), dtype=np.float32)

    for i, r in enumerate(radii):
        gw      = np.exp(-0.5 * ((blur_map - r) / _layer_sigma) ** 2)
        layer_w = gw * blurred_mask
        if layer_w.sum() < 1e-3:
            _progress(f"Blur {i + 1}/{len(radii)}")
            continue
        dilated = _dilate(layer_w, min(r, 8.0))
        blurred = apply_kernel(rgb_f, kernels[i])
        accum  += blurred * dilated[:, :, np.newaxis]
        weight += dilated
        _progress(f"Blur {i + 1}/{len(radii)}")

    no_blur    = (weight < 1e-6)[:, :, np.newaxis]
    blur_layer = np.clip(
        accum / np.maximum(weight, 1e-6)[:, :, np.newaxis], 0.0, 255.0)
    blur_layer = np.where(no_blur, rgb_f, blur_layer)

    # ── Masque net par-dessus ──────────────────────────────────────────────────
    sharp_alpha = _smooth(
        _erode(sharp_mask, min(max(max_r * 0.05, 0.5), 1.5)),
        sigma=min(max(max_r * 0.04, 0.5), 1.0))
    sharp_alpha = np.clip(sharp_alpha, 0.0, 1.0)

    result = (blur_layer * (1.0 - sharp_alpha[:, :, np.newaxis])
              + rgb_f     *  sharp_alpha[:, :, np.newaxis])

    _progress(t("progress_comp", lang))
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


# =============================================================================
# Widget drapeaux
# =============================================================================

class FlagButton(tk.Canvas):
    W, H = 36, 24

    def __init__(self, master, lang_code, command, **kw):
        super().__init__(master, width=self.W, height=self.H,
                         highlightthickness=2, cursor="hand2", **kw)
        self.lang_code = lang_code
        self.command   = command
        self._selected = False
        self._draw()
        self.bind("<Button-1>", lambda _: self.command(self.lang_code))

    def set_selected(self, v):
        self._selected = v
        self.config(highlightbackground="#a0c4ff" if v else "#2a2a4a")

    def _star(self, cx, cy, r, color):
        pts = []
        for i in range(10):
            a  = math.radians(i * 36 - 90)
            rl = r if i % 2 == 0 else r * 0.42
            pts += [cx + math.cos(a) * rl, cy + math.sin(a) * rl]
        self.create_polygon(pts, fill=color, outline="")

    def _hband(self, c1, c2, c3):
        W, H = self.W, self.H
        self.create_rectangle(0,       0, W, H//3,   fill=c1, outline="")
        self.create_rectangle(0,   H//3, W, 2*H//3,  fill=c2, outline="")
        self.create_rectangle(0, 2*H//3, W, H,       fill=c3, outline="")

    def _vband(self, c1, c2, c3):
        W, H = self.W, self.H
        self.create_rectangle(0,      0, W//3,   H, fill=c1, outline="")
        self.create_rectangle(W//3,   0, 2*W//3, H, fill=c2, outline="")
        self.create_rectangle(2*W//3, 0, W,      H, fill=c3, outline="")

    def _draw(self):
        W, H = self.W, self.H
        self.delete("all")
        lc = self.lang_code

        if lc == "fr":
            self._vband("#002395", "#FFFFFF", "#ED2939")
        elif lc == "en":
            self.create_rectangle(0, 0, W, H, fill="#012169", outline="")
            for w, c in [(5, "#FFFFFF"), (2, "#C8102E")]:
                self.create_line(0, 0, W, H, fill=c, width=w)
                self.create_line(W, 0, 0, H, fill=c, width=w)
            self.create_rectangle(W//2-3, 0, W//2+3, H, fill="#FFFFFF", outline="")
            self.create_rectangle(0, H//2-2, W, H//2+2, fill="#FFFFFF", outline="")
            self.create_rectangle(W//2-2, 0, W//2+2, H, fill="#C8102E", outline="")
            self.create_rectangle(0, H//2-1, W, H//2+1, fill="#C8102E", outline="")
        elif lc == "nl":
            self._hband("#AE1C28", "#FFFFFF", "#21468B")
        elif lc == "de":
            self._hband("#000000", "#DD0000", "#FFCE00")
        elif lc == "zh":
            self.create_rectangle(0, 0, W, H, fill="#DE2910", outline="")
            self._star(W*0.28, H*0.38, 5.0, "#FFDE00")
            for sx, sy in [(0.54, 0.15), (0.65, 0.28), (0.65, 0.52), (0.54, 0.65)]:
                self._star(W*sx, H*sy, 2.5, "#FFDE00")
        elif lc == "ja":
            self.create_rectangle(0, 0, W, H, fill="#FFFFFF", outline="")
            r = min(W, H) // 3
            cx, cy = W//2, H//2
            self.create_oval(cx-r, cy-r, cx+r, cy+r, fill="#BC002D", outline="")
        elif lc == "ar":
            self.create_rectangle(0, 0, W, H, fill="#006C35", outline="")
            self.create_line(4, H//2+2, W-4, H//2+2, fill="#FFFFFF", width=2)
            self.create_arc(4, H//2-2, 14, H//2+8, start=0, extent=180,
                            outline="#FFFFFF", width=2, style="arc")
            self.create_line(6, H//2-4, W-6, H//2-4, fill="#FFFFFF", width=1)
            self.create_line(9, H//2-7, W-9, H//2-7, fill="#FFFFFF", width=1)
        elif lc == "kl":
            self.create_rectangle(0, 0, W, H, fill="#111111", outline="")
            cx, cy = W//2, H//2
            self.create_polygon([cx,2, W-3,cy, cx,H-2, 3,cy],
                                fill="#8B0000", outline="#B8860B", width=1)
            for angle in [0, 120, 240]:
                rad = math.radians(angle - 90)
                self.create_line(cx + math.cos(rad)*3, cy + math.sin(rad)*3,
                                 cx + math.cos(rad)*7, cy + math.sin(rad)*7,
                                 fill="#FFD700", width=2)
        elif lc == "wa":
            self.create_rectangle(0, 0, W, H, fill="#FBDB00", outline="")
            self.create_oval(8, 7, 21, 19, fill="#CC0000", outline="")
            self.create_oval(19, 4, 27, 12, fill="#CC0000", outline="")
            self.create_polygon([20,4, 24,1, 27,5], fill="#CC0000", outline="")
            self.create_polygon([26,7, 31,9, 26,10], fill="#FBDB00", outline="")
            self.create_polygon([8,10, 2,5, 2,18], fill="#CC0000", outline="")
            self.create_line(14, 19, 12, 23, fill="#CC0000", width=2)
            self.create_line(14, 19, 16, 23, fill="#CC0000", width=2)
        elif lc == "ru":
            self._hband("#FFFFFF", "#0039A6", "#D52B1E")
        elif lc == "hi":
            self._hband("#FF9933", "#FFFFFF", "#138808")
            cx, cy, r = W//2, H//2, 4
            self.create_oval(cx-r, cy-r, cx+r, cy+r, outline="#000080", width=1, fill="")
            for i in range(8):
                a = math.radians(i * 45)
                self.create_line(cx, cy, cx+math.cos(a)*r, cy+math.sin(a)*r,
                                 fill="#000080", width=1)
        elif lc == "es":
            self.create_rectangle(0,       0, W, H//4,   fill="#AA151B", outline="")
            self.create_rectangle(0,   H//4, W, 3*H//4,  fill="#F1BF00", outline="")
            self.create_rectangle(0, 3*H//4, W, H,       fill="#AA151B", outline="")
        elif lc == "pt":
            self.create_rectangle(0,    0, W//3, H, fill="#006600", outline="")
            self.create_rectangle(W//3, 0, W,    H, fill="#FF0000", outline="")
            cx = W//3
            self.create_oval(cx-4, H//2-4, cx+4, H//2+4,
                             fill="#FFD700", outline="#003399", width=1)
        elif lc == "bn":
            self.create_rectangle(0, 0, W, H, fill="#006A4E", outline="")
            cx, cy = W//2 - 2, H//2
            r = min(W, H) // 3
            self.create_oval(cx-r, cy-r, cx+r, cy+r, fill="#F42A41", outline="")

        self.create_rectangle(0, 0, W-1, H-1, outline="#333333", fill="")


# =============================================================================
# Widget zone de netteté
# =============================================================================

class FocusZoneCanvas(tk.Canvas):
    H   = 82
    PAD = 16

    def __init__(self, master, var_near, var_far, var_falloff_near, var_falloff_far,
                 callback, lang_getter, **kw):
        super().__init__(master, height=self.H, bg="#0d0d1a",
                         highlightthickness=1, highlightbackground="#2a2a4a", **kw)
        self.var_near         = var_near
        self.var_far          = var_far
        self.var_falloff_near = var_falloff_near
        self.var_falloff_far  = var_falloff_far
        self.callback         = callback
        self.lang_getter      = lang_getter
        self._drag            = None
        self.bind("<Configure>",       lambda _: self.redraw())
        self.bind("<ButtonPress-1>",   self._on_press)
        self.bind("<B1-Motion>",       self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)

    def redraw(self):
        self.delete("all")
        W = self.winfo_width()
        if W < 30:
            return
        H, P  = self.H, self.PAD
        IW    = W - 2 * P
        near  = self.var_near.get()
        far   = self.var_far.get()
        fo_n  = self.var_falloff_near.get()   # transition côté proche
        fo_f  = self.var_falloff_far.get()    # transition côté lointain
        lang  = self.lang_getter()

        def xp(d):  return P + d * IW
        def yp(tv): return H - P - tv * (H - 2*P)

        self.create_rectangle(xp(near), P+2, xp(far), H-P, fill="#132613", outline="")

        N = max(int(IW * 2), 120)
        for side in ("left", "right"):
            fo  = fo_n if side == "left" else fo_f
            col = "#4a9eff" if side == "left" else "#ff9f4a"  # bleu=proche, orange=loin
            pts = []
            for i in range(N + 1):
                d = i / N
                if side == "left"  and d > near: continue
                if side == "right" and d < far:  continue
                gap  = (near - d) if side == "left" else (d - far)
                tv   = min(gap / max(fo, 1e-6), 1.0)
                tv   = tv * tv * (3.0 - 2.0 * tv)
                pts += [xp(d), yp(tv)]
            if len(pts) < 4:
                continue
            poly = ([xp(0), yp(0)] + pts + [xp(near), yp(0)] if side == "left"
                    else [xp(far), yp(0)] + pts + [xp(1), yp(0)])
            self.create_polygon(poly, fill="#0a2040", outline="")
            self.create_line(pts, fill=col, width=2, smooth=True)

        self.create_line(P, H-P, W-P, H-P, fill="#2a2a5a", width=1)
        self.create_line(xp(near), H-P, xp(far), H-P, fill="#2aaa5a", width=2)

        for val, color in [(near, "#60d394"), (far, "#ff9f6b")]:
            cx = xp(val)
            self.create_line(cx, P+2, cx, H-P, fill=color, width=2, dash=(4, 3))
            self.create_oval(cx-6, H-P-6, cx+6, H-P+6,
                             fill=color, outline="#1a1a2e", width=2)
            self.create_text(cx, P+10, text=f"{val:.2f}",
                             fill=color, font=("Courier New", 8, "bold"))

        self.create_text(xp((near+far)/2), H//2+4,
                         text=t("focus_zone_net", lang),
                         fill="#60d394", font=("Courier New", 8, "bold"))
        self.create_text(P,   H-4, text="0", fill="#444466",
                         font=("Courier New", 8), anchor="w")
        self.create_text(W-P, H-4, text="1", fill="#444466",
                         font=("Courier New", 8), anchor="e")

    def _d_from_x(self, ex):
        W, P = self.winfo_width(), self.PAD
        return max(0.0, min(1.0, (ex - P) / max(W - 2*P, 1)))

    def _on_press(self, e):
        W, P = self.winfo_width(), self.PAD
        IW   = W - 2*P
        xn   = P + self.var_near.get() * IW
        xf   = P + self.var_far.get()  * IW
        dn, df = abs(e.x - xn), abs(e.x - xf)
        if min(dn, df) > 16:
            return
        self._drag = "near" if dn <= df else "far"

    def _on_drag(self, e):
        if not self._drag:
            return
        d    = self._d_from_x(e.x)
        near = self.var_near.get()
        far  = self.var_far.get()
        if self._drag == "near": self.var_near.set(min(d, far  - 0.02))
        else:                    self.var_far.set( max(d, near + 0.02))
        self.redraw()
        self.callback()

    def _on_release(self, e):
        self._drag = None


# =============================================================================
# Application principale
# =============================================================================

class DofApp(tk.Tk):

    PANEL_MIN = 260
    PANEL_MAX = 600
    PANEL_DEF = 320

    def __init__(self):
        super().__init__()
        self.title("Depth of Field Tool  —  HQ  v9")
        self.configure(bg="#1a1a2e")
        self.resizable(True, True)

        self.lang        = "fr"
        self.rgb_img     = None
        self.depth_img   = None
        self.result_img  = None
        self._depth_path = None
        self._render_job = None
        self._computing  = False
        self._kernel_key = "disk"

        # Largeur courante du panneau gauche
        self._panel_width = self.PANEL_DEF

        self._build_ui()

    # =========================================================================
    # Construction UI
    # =========================================================================

    def _build_ui(self):
        st = ttk.Style(self)
        st.theme_use("clam")
        BG = "#1a1a2e"
        st.configure("TFrame",       background=BG)
        st.configure("TLabel",       background=BG, foreground="#e0e0f0",
                     font=("Courier New", 10))
        st.configure("TButton",      background="#16213e", foreground="#a0c4ff",
                     font=("Courier New", 10, "bold"), borderwidth=0, padding=6)
        st.map("TButton",            background=[("active", "#0f3460")])
        st.configure("TScale",       background=BG, troughcolor="#16213e",
                     sliderlength=18)
        st.configure("TCheckbutton", background=BG, foreground="#c0c0ff",
                     font=("Courier New", 10))
        st.configure("TCombobox",    fieldbackground="#16213e",
                     background="#16213e", foreground="#a0c4ff",
                     font=("Courier New", 10))
        st.configure("Sash",         sashthickness=6, background="#2a2a6a")

        # ── Conteneur principal avec PanedWindow (redimensionnable) ──────────
        self._paned = tk.PanedWindow(self, orient=tk.HORIZONTAL,
                                     bg="#0d0d1a",
                                     sashwidth=6,
                                     sashrelief=tk.FLAT,
                                     sashpad=0,
                                     handlepad=80,
                                     handlesize=10)
        self._paned.pack(fill=tk.BOTH, expand=True)

        # ── Panneau gauche (scrollable) ───────────────────────────────────────
        left_outer = tk.Frame(self._paned, bg=BG,
                              width=self._panel_width)
        left_outer.pack_propagate(False)

        vsb = tk.Scrollbar(left_outer, orient="vertical", bg="#16213e",
                           troughcolor="#0d0d1a", activebackground="#0f3460")
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        scroll_cv = tk.Canvas(left_outer, bg=BG, highlightthickness=0,
                              yscrollcommand=vsb.set)
        scroll_cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.config(command=scroll_cv.yview)

        ctrl = ttk.Frame(scroll_cv, padding=12)
        win_id = scroll_cv.create_window((0, 0), window=ctrl, anchor="nw")

        def _on_ctrl_configure(e):
            scroll_cv.configure(scrollregion=scroll_cv.bbox("all"))
        ctrl.bind("<Configure>", _on_ctrl_configure)

        def _on_canvas_resize(e):
            scroll_cv.itemconfig(win_id, width=e.width)
        scroll_cv.bind("<Configure>", _on_canvas_resize)

        def _on_mousewheel(e):
            if e.num == 4:
                scroll_cv.yview_scroll(-1, "units")
            elif e.num == 5:
                scroll_cv.yview_scroll(1, "units")
            else:
                # Windows: delta ~120, macOS: delta ~1
                units = -e.delta // 120 if abs(e.delta) >= 120 else -e.delta
                scroll_cv.yview_scroll(units or -1, "units")

        for w in (scroll_cv, ctrl):
            w.bind("<MouseWheel>", _on_mousewheel)
            w.bind("<Button-4>",   _on_mousewheel)
            w.bind("<Button-5>",   _on_mousewheel)

        # ── Drapeaux — grille 2×7, toujours visible, wraplength adaptatif ────
        flag_outer = ttk.Frame(ctrl)
        flag_outer.pack(fill=tk.X, pady=(0, 6))

        self._flag_btns = {}
        for row_langs in FLAG_ROWS:
            row = ttk.Frame(flag_outer)
            row.pack(fill=tk.X, pady=2)
            for lc in row_langs:
                fb = FlagButton(row, lc, self._switch_lang, bg=BG)
                fb.pack(side=tk.LEFT, padx=2, pady=1)
                self._flag_btns[lc] = fb
        self._flag_btns[self.lang].set_selected(True)

        # ── Titre ─────────────────────────────────────────────────────────────
        self.lbl_title = ttk.Label(ctrl, font=("Courier New", 11, "bold"),
                                   foreground="#a0c4ff")
        self.lbl_title.pack(pady=(0, 8))

        # ── Chargement ───────────────────────────────────────────────────────
        self.btn_load_rgb = ttk.Button(ctrl, command=self._load_rgb)
        self.btn_load_rgb.pack(fill=tk.X, pady=3)
        self.lbl_rgb = ttk.Label(ctrl, foreground="#555577", wraplength=10)
        self.lbl_rgb.pack()

        self.btn_load_depth = ttk.Button(ctrl, command=self._load_depth)
        self.btn_load_depth.pack(fill=tk.X, pady=(8, 3))
        self.lbl_depth = ttk.Label(ctrl, foreground="#555577", wraplength=10)
        self.lbl_depth.pack()

        self.var_invert = tk.BooleanVar(value=False)
        self.chk_invert = ttk.Checkbutton(ctrl, variable=self.var_invert,
                                           command=self._reload_depth)
        self.chk_invert.pack(anchor="w", pady=(5, 0))

        ttk.Separator(ctrl, orient="horizontal").pack(fill=tk.X, pady=8)

        # ── Zone de netteté ───────────────────────────────────────────────────
        self.lbl_focus_title = ttk.Label(ctrl, foreground="#a0c4ff")
        self.lbl_focus_title.pack(anchor="w")

        self.var_near         = tk.DoubleVar(value=0.35)
        self.var_far          = tk.DoubleVar(value=0.65)
        self.var_falloff_near = tk.DoubleVar(value=0.06)
        self.var_falloff_far  = tk.DoubleVar(value=0.06)

        self.focus_viz = FocusZoneCanvas(
            ctrl,
            self.var_near, self.var_far,
            self.var_falloff_near, self.var_falloff_far,
            callback=self._on_focus_changed,
            lang_getter=lambda: self.lang)
        self.focus_viz.pack(fill=tk.X, pady=(4, 8))

        def _slider_row(lbl_attr, var, from_, to_, color):
            lbl = ttk.Label(ctrl)
            setattr(self, lbl_attr + "_k", lbl)
            lbl.pack(anchor="w")
            row = ttk.Frame(ctrl)
            row.pack(fill=tk.X, pady=(0, 4))
            ttk.Scale(row, from_=from_, to=to_, orient=tk.HORIZONTAL,
                      variable=var,
                      command=lambda _: self._on_slider()).pack(
                          side=tk.LEFT, fill=tk.X, expand=True)
            val_lbl = ttk.Label(row, foreground=color, width=9,
                                font=("Courier New", 10))
            val_lbl.pack(side=tk.LEFT)
            setattr(self, lbl_attr + "_v", val_lbl)

        _slider_row("lbl_near",         self.var_near,         0.0,  1.0,  "#60d394")
        _slider_row("lbl_far",          self.var_far,          0.0,  1.0,  "#ff9f6b")
        _slider_row("lbl_fo_near",      self.var_falloff_near, 0.0,  0.25, "#4a9eff")
        _slider_row("lbl_fo_far",       self.var_falloff_far,  0.0,  0.25, "#ff9f4a")

        ttk.Separator(ctrl, orient="horizontal").pack(fill=tk.X, pady=6)

        self.var_blur = tk.DoubleVar(value=22.0)
        _slider_row("lbl_blur",  self.var_blur,    1.0,   100.0, "#a0c4ff")

        ttk.Separator(ctrl, orient="horizontal").pack(fill=tk.X, pady=6)

        # ── Qualité ───────────────────────────────────────────────────────────
        self.lbl_quality_title = ttk.Label(ctrl, foreground="#a0c4ff")
        self.lbl_quality_title.pack(anchor="w")

        self.lbl_bokeh_shape_k = ttk.Label(ctrl)
        self.lbl_bokeh_shape_k.pack(anchor="w", pady=(4, 0))

        self.var_kernel_display = tk.StringVar()
        self.cb_kernel = ttk.Combobox(ctrl, textvariable=self.var_kernel_display,
                                       state="readonly")
        self.cb_kernel.pack(fill=tk.X, pady=(2, 5))
        self.cb_kernel.bind("<<ComboboxSelected>>", self._on_kernel_select)

        self.var_steps = tk.DoubleVar(value=16.0)
        _slider_row("lbl_layers", self.var_steps, 4, 128, "#a0c4ff")

        self.lbl_render_mode_k = ttk.Label(ctrl)
        self.lbl_render_mode_k.pack(anchor="w", pady=(4, 0))

        self._render_mode = "two_layers"   # valeur interne
        self.var_render_mode_display = tk.StringVar()
        self.cb_render_mode = ttk.Combobox(ctrl,
                                            textvariable=self.var_render_mode_display,
                                            state="readonly")
        self.cb_render_mode.pack(fill=tk.X, pady=(2, 5))
        self.cb_render_mode.bind("<<ComboboxSelected>>", self._on_render_mode_select)

        ttk.Separator(ctrl, orient="horizontal").pack(fill=tk.X, pady=7)

        # ── Aperçu ────────────────────────────────────────────────────────────
        self.lbl_preview_title = ttk.Label(ctrl)
        self.lbl_preview_title.pack(anchor="w")

        self.var_view = tk.StringVar(value="result")
        for val, attr in [("original", "rb_original"), ("depth", "rb_depth"),
                          ("blur_map", "rb_blur_map"), ("result", "rb_result")]:
            rb = ttk.Radiobutton(ctrl, variable=self.var_view, value=val,
                                 command=self._refresh_preview, style="TLabel")
            rb.pack(anchor="w")
            setattr(self, attr, rb)

        ttk.Separator(ctrl, orient="horizontal").pack(fill=tk.X, pady=(7, 4))

        # ── Overlay DoF ───────────────────────────────────────────────────────
        self.var_overlay = tk.BooleanVar(value=False)
        self.chk_overlay = ttk.Checkbutton(ctrl, variable=self.var_overlay,
                                            command=self._refresh_preview)
        self.chk_overlay.pack(anchor="w")

        self.lbl_overlay_opacity_k = ttk.Label(ctrl)
        self.lbl_overlay_opacity_k.pack(anchor="w", pady=(4, 0))

        ov_row = ttk.Frame(ctrl)
        ov_row.pack(fill=tk.X, pady=(0, 2))
        self.var_overlay_opacity = tk.DoubleVar(value=0.55)
        ttk.Scale(ov_row, from_=0.05, to=1.0, orient=tk.HORIZONTAL,
                  variable=self.var_overlay_opacity,
                  command=lambda _: self._refresh_preview()).pack(
                      side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_overlay_opacity_v = ttk.Label(ov_row, foreground="#c0c0ff",
                                                width=5, font=("Courier New", 10))
        self.lbl_overlay_opacity_v.pack(side=tk.LEFT)

        leg = ttk.Frame(ctrl)
        leg.pack(anchor="w", pady=(2, 0))
        self.lbl_leg_far   = ttk.Label(leg, foreground="#ff5555", font=("Courier New", 9))
        self.lbl_leg_sharp = ttk.Label(leg, foreground="#5599ff", font=("Courier New", 9))
        self.lbl_leg_near  = ttk.Label(leg, foreground="#44cc66", font=("Courier New", 9))
        self.lbl_leg_far.pack(anchor="w")
        self.lbl_leg_sharp.pack(anchor="w")
        self.lbl_leg_near.pack(anchor="w")

        ttk.Separator(ctrl, orient="horizontal").pack(fill=tk.X, pady=7)

        # ── Largeur panneau (slider) ──────────────────────────────────────────
        self.lbl_panel_width_k = ttk.Label(ctrl)
        self.lbl_panel_width_k.pack(anchor="w")

        pw_row = ttk.Frame(ctrl)
        pw_row.pack(fill=tk.X, pady=(0, 4))
        self.var_panel_width = tk.IntVar(value=self._panel_width)
        ttk.Scale(pw_row, from_=self.PANEL_MIN, to=self.PANEL_MAX,
                  orient=tk.HORIZONTAL,
                  variable=self.var_panel_width,
                  command=self._on_panel_width_slider).pack(
                      side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_panel_width_v = ttk.Label(pw_row, foreground="#a0c4ff",
                                            width=6, font=("Courier New", 10))
        self.lbl_panel_width_v.pack(side=tk.LEFT)

        ttk.Separator(ctrl, orient="horizontal").pack(fill=tk.X, pady=7)

        # ── Calcul / export ───────────────────────────────────────────────────
        self.btn_compute = ttk.Button(ctrl, command=self._compute_full)
        self.btn_compute.pack(fill=tk.X, pady=3)
        self.btn_export  = ttk.Button(ctrl, command=self._export)
        self.btn_export.pack(fill=tk.X, pady=3)

        self.var_progress = tk.DoubleVar(value=0.0)
        self.progressbar  = ttk.Progressbar(ctrl, variable=self.var_progress,
                                             maximum=100.0,
                                             mode="determinate")
        self.progressbar.pack(fill=tk.X, pady=(8, 2))
        self.lbl_progress = ttk.Label(ctrl, text="", foreground="#8888aa",
                                       font=("Courier New", 9))
        self.lbl_progress.pack(anchor="w")

        self.lbl_status = ttk.Label(ctrl, foreground="#60d394", wraplength=10)
        self.lbl_status.pack(pady=(8, 0))

        # ── Info GPU + bouton diagnostic ──────────────────────────────────────
        gpu_active = GPU_BACKEND not in ("cpu", "torch_cpu")
        gpu_color  = "#a0c4ff" if gpu_active else "#666688"
        gpu_icon   = "⚡" if gpu_active else "🖥"
        gpu_row    = ttk.Frame(ctrl)
        gpu_row.pack(fill=tk.X, pady=(4, 0))
        self.lbl_gpu = ttk.Label(gpu_row,
                                  text=f"{gpu_icon} {GPU_DEVICE}",
                                  foreground=gpu_color,
                                  font=("Courier New", 9))
        self.lbl_gpu.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.btn_gpu_diag = ttk.Button(
            gpu_row,
            text="🔍",
            width=3,
            command=self._show_gpu_diag)
        self.btn_gpu_diag.pack(side=tk.RIGHT, padx=(4, 0))

        # ── Panneau droit (canvas prévisualisation) ───────────────────────────
        right_frame = tk.Frame(self._paned, bg="#0d0d1a")

        zoom_bar = ttk.Frame(right_frame)
        zoom_bar.pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Label(zoom_bar, text="🔍", font=("Courier New", 9)).pack(side=tk.LEFT)
        self.var_zoom = tk.DoubleVar(value=1.0)
        ttk.Scale(zoom_bar, from_=0.25, to=4.0, orient=tk.HORIZONTAL,
                  variable=self.var_zoom, length=200,
                  command=lambda _: self._refresh_preview()).pack(
                      side=tk.LEFT, padx=(4, 4))
        self.lbl_zoom_v = ttk.Label(zoom_bar, foreground="#a0c4ff", width=6,
                                     font=("Courier New", 9))
        self.lbl_zoom_v.pack(side=tk.LEFT)
        ttk.Button(zoom_bar, text="1:1",
                   command=lambda: [self.var_zoom.set(1.0),
                                    self._refresh_preview()]).pack(
                       side=tk.LEFT, padx=(6, 0))

        self.canvas = tk.Canvas(right_frame, bg="#0d0d1a", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self._tk_img = None

        # Ajout des deux panneaux au PanedWindow
        self._paned.add(left_outer,  minsize=self.PANEL_MIN, width=self._panel_width)
        self._paned.add(right_frame, minsize=300)

        # Liaison sash → slider (synchronisation inverse)
        self._paned.bind("<ButtonRelease-1>", self._on_sash_release)

        # Lancer l'interface
        self._apply_lang()

    # =========================================================================
    # Redimensionnement du panneau gauche
    # =========================================================================

    def _on_panel_width_slider(self, val=None):
        """Slider → redimensionner le panneau via le sash du PanedWindow."""
        w = int(self.var_panel_width.get())
        w = max(self.PANEL_MIN, min(self.PANEL_MAX, w))
        self._panel_width = w
        self.lbl_panel_width_v.config(text=f"{w} px")
        # Mettre à jour wraplength des labels dépendants
        wl = max(w - 30, 60)
        for lbl in (self.lbl_status, self.lbl_rgb, self.lbl_depth):
            lbl.config(wraplength=wl)
        # Repositionner le sash
        try:
            self._paned.sash_place(0, w, 0)
        except Exception:
            pass

    def _on_sash_release(self, event):
        """Sash déplacé à la souris → synchroniser le slider."""
        try:
            x, _ = self._paned.sash_coord(0)
            x = max(self.PANEL_MIN, min(self.PANEL_MAX, x))
            self._panel_width = x
            self.var_panel_width.set(x)
            self.lbl_panel_width_v.config(text=f"{x} px")
            wl = max(x - 30, 60)
            for lbl in (self.lbl_status, self.lbl_rgb, self.lbl_depth):
                lbl.config(wraplength=wl)
        except Exception:
            pass

    # =========================================================================
    # Langue
    # =========================================================================

    def _switch_lang(self, code):
        if code == self.lang:
            return
        for lc, fb in self._flag_btns.items():
            fb.set_selected(lc == code)
        self.lang = code
        self._apply_lang()

    def _apply_lang(self):
        L  = self.lang
        wl = max(self._panel_width - 30, 60)
        self.title(f"DOF Tool HQ  v9  —  {L.upper()}")
        self.lbl_title.config(text=t("title", L))
        self.btn_load_rgb.config(text=t("btn_load_rgb", L))
        self.btn_load_depth.config(text=t("btn_load_depth", L))
        if not self._has_rgb():
            self.lbl_rgb.config(text=t("no_file", L), wraplength=wl)
        if not self._has_depth():
            self.lbl_depth.config(text=t("no_file", L), wraplength=wl)
        self.chk_invert.config(text=t("chk_invert", L))
        self.lbl_focus_title.config(text=t("focus_zone_title", L))
        self.lbl_near_k.config(text=t("lbl_near", L))
        self.lbl_far_k.config( text=t("lbl_far",  L))
        self.lbl_fo_near_k.config(text=t("lbl_falloff_near", L))
        self.lbl_fo_far_k.config( text=t("lbl_falloff_far",  L))
        self.lbl_blur_k.config(text=t("lbl_max_blur", L))
        self.lbl_quality_title.config(text=t("quality_title", L))
        self.lbl_bokeh_shape_k.config(text=t("lbl_bokeh_shape", L))
        self.lbl_layers_k.config(     text=t("lbl_layers", L))
        kl = [t(k, L) for k in KERNEL_T]
        self.cb_kernel.config(values=kl)
        self.var_kernel_display.set(kl[KERNEL_KEYS.index(self._kernel_key)])
        self.lbl_render_mode_k.config(text=t("lbl_render_mode", L))
        rm_vals = [t("render_two_layers", L), t("render_single", L)]
        self.cb_render_mode.config(values=rm_vals)
        self.var_render_mode_display.set(
            rm_vals[0] if self._render_mode == "two_layers" else rm_vals[1])
        self.lbl_preview_title.config(text=t("preview_title", L))
        self.rb_original.config(text=t("view_original", L))
        self.rb_depth.config(   text=t("view_depth",    L))
        self.rb_blur_map.config(text=t("view_blur_map", L))
        self.rb_result.config(  text=t("view_result",   L))
        self.btn_compute.config(text=t("btn_compute", L))
        self.btn_export.config( text=t("btn_export",  L))
        self.lbl_status.config( text=t("status_ready", L), wraplength=wl)
        self.chk_overlay.config(          text=t("chk_overlay", L))
        self.lbl_overlay_opacity_k.config(text=t("lbl_overlay_opacity", L))
        self.lbl_leg_far.config(          text=t("overlay_legend_far",   L))
        self.lbl_leg_sharp.config(        text=t("overlay_legend_sharp", L))
        self.lbl_leg_near.config(         text=t("overlay_legend_near",  L))
        self.lbl_panel_width_k.config(    text=t("lbl_panel_width", L))
        self.lbl_panel_width_v.config(    text=f"{self._panel_width} px")
        self._update_labels()
        self.focus_viz.redraw()

    def _has_rgb(self):   return self.rgb_img   is not None
    def _has_depth(self): return self.depth_img  is not None

    # =========================================================================
    # Chargement fichiers
    # =========================================================================

    def _load_rgb(self):
        L = self.lang
        path = filedialog.askopenfilename(
            title=t("dlg_rgb_title", L),
            filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff"),
                       (t("dlg_all_files", L), "*.*")])
        if not path: return
        self.rgb_img = np.array(Image.open(path).convert("RGB"))
        wl = max(self._panel_width - 30, 60)
        self.lbl_rgb.config(text=os.path.basename(path)[-40:],
                            foreground="#60d394", wraplength=wl)
        self._status(t("status_rgb_loaded", L))
        self._auto_compute()

    def _load_depth(self):
        L = self.lang
        path = filedialog.askopenfilename(
            title=t("dlg_depth_title", L),
            filetypes=[("PNG", "*.png"), (t("dlg_all_files", L), "*.*")])
        if not path: return
        self._depth_path = path
        self.depth_img   = load_depth(path, self.var_invert.get())
        # ── Redimensionner si la depth map ne correspond pas à l'image RGB ──
        if self._has_rgb() and self.depth_img.shape[:2] != self.rgb_img.shape[:2]:
            h, w = self.rgb_img.shape[:2]
            if HAS_CV2:
                self.depth_img = cv2.resize(self.depth_img, (w, h),
                                            interpolation=cv2.INTER_LINEAR)
            else:
                self.depth_img = np.array(
                    Image.fromarray(self.depth_img, mode='F')
                        .resize((w, h), _BILINEAR),
                    dtype=np.float32)
        wl = max(self._panel_width - 30, 60)
        self.lbl_depth.config(text=os.path.basename(path)[-40:],
                              foreground="#60d394", wraplength=wl)
        self._status(t("status_depth_loaded", L))
        self._auto_compute()

    def _reload_depth(self):
        if self._depth_path:
            self.depth_img = load_depth(self._depth_path, self.var_invert.get())
            if self._has_rgb() and self.depth_img.shape[:2] != self.rgb_img.shape[:2]:
                h, w = self.rgb_img.shape[:2]
                if HAS_CV2:
                    self.depth_img = cv2.resize(self.depth_img, (w, h),
                                                interpolation=cv2.INTER_LINEAR)
                else:
                    self.depth_img = np.array(
                        Image.fromarray(self.depth_img, mode='F')
                            .resize((w, h), _BILINEAR),
                        dtype=np.float32)
            self._auto_compute()

    # =========================================================================
    # Noyau Bokeh
    # =========================================================================

    def _on_kernel_select(self, _=None):
        display = self.var_kernel_display.get()
        L = self.lang
        for key, tk_key in zip(KERNEL_KEYS, KERNEL_T):
            if t(tk_key, L) == display:
                self._kernel_key = key
                break
        self._schedule_compute()

    def _on_render_mode_select(self, _=None):
        display = self.var_render_mode_display.get()
        L = self.lang
        if display == t("render_single", L):
            self._render_mode = "single"
        else:
            self._render_mode = "two_layers"
        self._schedule_compute()

    # =========================================================================
    # Sliders
    # =========================================================================

    def _on_slider(self):
        n, f = self.var_near.get(), self.var_far.get()
        if n >= f:
            self.var_far.set(min(n + 0.02, 1.0))
        self._update_labels()
        self.focus_viz.redraw()
        if self.var_overlay.get() and hasattr(self, "_depth_preview"):
            self._refresh_preview()
        self._schedule_compute()

    def _on_focus_changed(self):
        self._update_labels()
        if self.var_overlay.get() and hasattr(self, "_depth_preview"):
            self._refresh_preview()
        self._schedule_compute()

    def _update_labels(self):
        L = self.lang
        self.lbl_near_v.config(    text=f"{self.var_near.get():.2f}")
        self.lbl_far_v.config(     text=f"{self.var_far.get():.2f}")
        self.lbl_fo_near_v.config( text=f"{self.var_falloff_near.get():.2f}")
        self.lbl_fo_far_v.config(  text=f"{self.var_falloff_far.get():.2f}")
        self.lbl_blur_v.config(    text=f"{self.var_blur.get():.0f} px")
        self.lbl_layers_v.config(  text=f"{int(self.var_steps.get())}{t('passes_suffix', L)}")
        self.lbl_overlay_opacity_v.config(
            text=f"{int(self.var_overlay_opacity.get() * 100)}%")

    def _schedule_compute(self):
        self._update_labels()
        if self._render_job:
            self.after_cancel(self._render_job)
        self._render_job = self.after(420, self._auto_compute)

    # =========================================================================
    # Progression
    # =========================================================================

    def _set_progress(self, done, total, label):
        self.var_progress.set(done / total * 100.0)
        self.lbl_progress.config(text=f"{label}  ({done}/{total})")

    def _reset_progress(self, msg=""):
        self.var_progress.set(0.0)
        self.lbl_progress.config(text=msg)

    # =========================================================================
    # Calcul DoF
    # =========================================================================

    def _auto_compute(self):
        if not self._has_rgb() or not self._has_depth() or self._computing:
            return
        threading.Thread(target=self._run_compute, args=(True,), daemon=True).start()

    def _compute_full(self):
        L = self.lang
        if not self._has_rgb() or not self._has_depth():
            messagebox.showwarning(t("dlg_missing_title", L),
                                   t("dlg_missing_body",  L))
            return
        self._status(t("status_computing", L))
        threading.Thread(target=self._run_compute, args=(False,), daemon=True).start()

    def _run_compute(self, preview_mode):
        self._computing = True
        L = self.lang
        self.after(0, self._reset_progress, t("status_starting", L))
        try:
            rgb   = self.rgb_img
            depth = self.depth_img

            if preview_mode:
                scale = min(1.0, 560 / max(rgb.shape[:2]))
                if scale < 1.0:
                    h = int(rgb.shape[0] * scale)
                    w = int(rgb.shape[1] * scale)
                    rgb   = np.array(
                        Image.fromarray(rgb).resize((w, h), _BILINEAR))
                    # Resize depth en float32 natif (mode 'F') — aucune perte
                    # de précision : évite le passage uint16 → I;16 → float
                    if HAS_CV2:
                        depth = cv2.resize(depth, (w, h),
                                           interpolation=cv2.INTER_LINEAR)
                    else:
                        depth = np.array(
                            Image.fromarray(depth, mode='F')
                                .resize((w, h), _BILINEAR),
                            dtype=np.float32)

            near         = self.var_near.get()
            far          = self.var_far.get()
            falloff_near = self.var_falloff_near.get()
            falloff_far  = self.var_falloff_far.get()
            max_b        = self.var_blur.get()
            if preview_mode:
                s     = min(1.0, 560 / max(self.rgb_img.shape[:2]))
                max_b = max(max_b * s, 2.0)

            steps    = int(self.var_steps.get())
            blur_map = compute_blur_radius_map(depth, near, far, max_b,
                                               falloff_near, falloff_far)

            def progress_cb(done, total, label):
                self.after(0, self._set_progress, done, total, label)

            if self._render_mode == "single":
                result = render_dof_single(rgb, blur_map, depth,
                                           kernel_key=self._kernel_key,
                                           steps=steps,
                                           progress_cb=progress_cb,
                                           lang=L)
            else:
                result = render_dof(rgb, blur_map, depth,
                                    kernel_key=self._kernel_key,
                                    steps=steps,
                                    bleed_correction=True,
                                    progress_cb=progress_cb,
                                    lang=L,
                                    focus_near=near,
                                    focus_far=far)

            self.result_img        = result
            self._depth_preview    = depth
            self._blur_map_preview = blur_map

            done_msg = (t("status_done_full", L) if not preview_mode
                        else t("status_done_preview", L))
            self.after(0, self._refresh_preview)
            self.after(0, self._status, done_msg)
            self.after(0, self._reset_progress, "")

        except Exception as e:
            import traceback
            self.after(0, self._status, f"Erreur : {e}")
            self.after(0, self._reset_progress, "")
            traceback.print_exc()
        finally:
            self._computing = False

    # =========================================================================
    # Affichage
    # =========================================================================

    def _refresh_preview(self, _=None):
        self.lbl_overlay_opacity_v.config(
            text=f"{int(self.var_overlay_opacity.get() * 100)}%")

        mode = self.var_view.get()
        if   mode == "original" and self._has_rgb():
            arr = self.rgb_img
        elif mode == "depth"    and hasattr(self, "_depth_preview"):
            d   = (self._depth_preview * 255).astype(np.uint8)
            arr = np.stack([d, d, d], axis=2)
        elif mode == "blur_map" and hasattr(self, "_blur_map_preview"):
            bm  = self._blur_map_preview
            bmn = (bm / max(bm.max(), 1e-6) * 255).astype(np.uint8)
            arr = np.stack([bmn, bmn, bmn], axis=2)
        elif mode == "result"   and self.result_img is not None:
            arr = self.result_img
        else:
            return

        if self.var_overlay.get() and hasattr(self, "_depth_preview"):
            overlay = make_zone_overlay(
                self._depth_preview,
                self.var_near.get(), self.var_far.get(),
                self.var_falloff_near.get(), self.var_falloff_far.get())
            if overlay.shape[:2] != arr.shape[:2]:
                h_a, w_a = arr.shape[:2]
                overlay = np.array(
                    Image.fromarray(overlay, mode="RGBA").resize(
                        (w_a, h_a), _NEAREST))
            arr = composite_overlay(arr, overlay,
                                    opacity=self.var_overlay_opacity.get())

        self._show_array(arr)

    def _show_array(self, arr):
        h, w  = arr.shape[:2]
        cw    = self.canvas.winfo_width()  or 900
        ch    = self.canvas.winfo_height() or 600
        zoom  = self.var_zoom.get()
        base  = min(cw / w, ch / h, 1.0)
        scale = base * zoom
        nw    = max(int(w * scale), 1)
        nh    = max(int(h * scale), 1)
        pil   = Image.fromarray(arr.astype(np.uint8)).resize((nw, nh), _BILINEAR)
        self._tk_img = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER,
                                  image=self._tk_img)
        self.lbl_zoom_v.config(text=f"×{zoom:.2f}")

    # =========================================================================
    # Export
    # =========================================================================

    def _export(self):
        L = self.lang
        if self.result_img is None:
            messagebox.showwarning(t("dlg_nothing_title", L),
                                   t("dlg_nothing_body",  L))
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
            title=t("dlg_save_title", L))
        if not path:
            return
        Image.fromarray(self.result_img).save(path)
        self._status(t("status_exported", L) + os.path.basename(path))

    def _show_gpu_diag(self):
        L = self.lang
        # Rafraîchir le label si CuPy a échoué à l'exécution depuis le démarrage
        gpu_active = GPU_BACKEND in ("cupy", "torch_cuda", "torch_mps")
        gpu_icon   = "⚡" if gpu_active else "🖥"
        gpu_color  = "#a0c4ff" if gpu_active else "#666688"
        self.lbl_gpu.config(text=f"{gpu_icon} {GPU_DEVICE}", foreground=gpu_color)
        messagebox.showinfo(t("dlg_gpu_title", L), gpu_diagnostic())

    def _status(self, msg):
        wl = max(self._panel_width - 30, 60)
        self.lbl_status.config(text=msg, wraplength=wl)


# =============================================================================

if __name__ == "__main__":
    app = DofApp()
    app.geometry("1440x860")
    app.mainloop()
