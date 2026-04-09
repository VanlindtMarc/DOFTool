"""
Depth of Field Tool  –  High Quality  (v13)
==========================================
• v13 : Auto-détection de l'inversion Z-depth, aperçu de la depth modifiée
        sous la courbe, slider « Couches (qualité) » déplacé juste au-dessus
        du bouton de calcul, suppression du slider « Largeur panneau »
        (le panneau reste redimensionnable par glisser-déposer du sash).
• Rendu par couches séparées : zéro halo sur fond blanc ou coloré.
• v9  : Séparation focus-aware.
• v10 : Flou de bord radial par sujet (edge blur).
• v11 : Glisser-déposer automatique — détection NB/couleur automatique,
        dépôt d'un ou deux fichiers en même temps.
• v12 : Correctifs critiques (fall-through GPU, race conditions,
        clipping falloff), kernels vectorisés, gaussien séparable,
        i18n complète des libellés split-compare.
• Pre-fill anti-saignement — les bords nets restent nets.
• Panneau gauche REDIMENSIONNABLE par glisser-déposer.

Langues : 🇫🇷 Français   🇬🇧 English    🇧🇪 Nederlands  🇩🇪 Deutsch
          🇨🇳 中文       🇯🇵 日本語     🇸🇦 العربية     🏴 Klingon
          🏴 Wallon li.  🇷🇺 Русский    🇮🇳 हिन्दी       🇪🇸 Español
          🇧🇷 Português  🇧🇩 বাংলা

Dépendances : pip install numpy pillow scipy
              (optionnel)  pip install opencv-python
              (DnD)        pip install tkinterdnd2
"""

import os
import re
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

# ── Drag-and-Drop (tkinterdnd2 optionnel) ────────────────────────────────────
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD as _TkDnD
    _DND_BASE = _TkDnD.Tk
    HAS_DND   = True
except ImportError:
    _DND_BASE = tk.Tk
    HAS_DND   = False
    DND_FILES = None


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
        "fr": "── DOF TOOL HQ v13 ──",    "en": "── DOF TOOL HQ v13 ──",
        "nl": "── DOF TOOL HQ v13 ──",    "de": "── DOF TOOL HQ v13 ──",
        "zh": "── DOF 工具 HQ v13 ──",    "ja": "── DOF ツール HQ v13 ──",
        "ar": "── أداة DOF HQ v13 ──",    "kl": "── DOF jan HQ v13 ──",
        "wa": "── OUTIL DOF HQ v13 ──",   "ru": "── ИНСТРУМЕНТ DOF v13 ──",
        "hi": "── DOF टूल HQ v13 ──",     "es": "── HERRAMIENTA DOF v13 ──",
        "pt": "── FERRAMENTA DOF v13 ──", "bn": "── DOF টুল HQ v13 ──",
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
        "fr": "📂  16-Bit Depth Map",  "en": "📂  16-Bit Depth Map",
        "nl": "📂  16-Bit Depth Map",  "de": "📂  16-Bit Depth Map",
        "zh": "📂  16位深度图",         "ja": "📂  16ビット深度マップ",
        "ar": "📂  خريطة عمق 16 بت",  "kl": "📂  16-jaj maS",
        "wa": "📂  Mape di profond 16-bit","ru": "📂  16-бит карта глубины",
        "hi": "📂  16-बिट डेप्थ मैप", "es": "📂  Mapa de profundidad 16-bit",
        "pt": "📂  Mapa de profundidade 16-bit","bn": "📂  ১৬-বিট ডেপ্থ ম্যাপ",
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
    "lbl_depth_curve": {
        "fr": "Courbe Z-depth",   "en": "Z-depth curve",
        "nl": "Z-diepte curve",   "de": "Z-Tiefe-Kurve",
        "zh": "Z深度曲线",         "ja": "Zデプス曲線",
        "ar": "منحنى عمق Z",       "kl": "Z-maS pagh",
        "wa": "Coûbe Z-profond",  "ru": "Кривая Z-глубины",
        "hi": "Z-गहराई वक्र",     "es": "Curva Z-prof.",
        "pt": "Curva Z-prof.",    "bn": "Z-গভীরতা বক্ররেখা",
    },
    "curve_hint": {
        "fr": "glisser · poignées = courbure · Alt = casser · dbl-clic = ajouter · clic-droit = supprimer",
        "en": "drag · handles = curvature · Alt = break · dbl-click = add · right-click = remove",
        "nl": "slepen · grepen = kromming · Alt = breken · dbl-klik = toevoegen · rmk = verwijderen",
        "de": "ziehen · Griffe = Krümmung · Alt = brechen · Dppl-Klick = hinzu · Rklick = löschen",
        "zh": "拖动 · 手柄=曲率 · Alt=断开 · 双击=添加 · 右键=删除",
        "ja": "ドラッグ · ハンドル=曲率 · Alt=分離 · ダブルクリック=追加 · 右クリック=削除",
        "ar": "اسحب · المقابض = انحناء · Alt = فصل · نقر مزدوج = إضافة · أيمن = حذف",
        "kl": "vIt · ghom = bend · Alt = chev · cha'-click = chel · nIH = Qaw'",
        "wa": "glèyî · håves = coûbeure · Alt = rompe · dbl-clik = radjouter · dr. = rissaetchî",
        "ru": "тянуть · ручки = кривизна · Alt = разорвать · 2×клик = добавить · ПКМ = удалить",
        "hi": "खींचें · हैंडल = वक्रता · Alt = तोड़ें · डबल-क्लिक = जोड़ें · राइट = हटाएँ",
        "es": "arrastrar · asas = curvatura · Alt = romper · dbl-clic = añadir · der = quitar",
        "pt": "arrastar · alças = curvatura · Alt = quebrar · dbl = add · dir = remover",
        "bn": "টানুন · হ্যান্ডেল = বক্রতা · Alt = ভাঙুন · ডাবল-ক্লিক = যোগ · রাইট = মুছুন",
    },
    "lbl_curve_quality": {
        "fr": "Qualité courbe",    "en": "Curve quality",
        "nl": "Curve kwaliteit",   "de": "Kurvenqualität",
        "zh": "曲线质量",            "ja": "曲線品質",
        "ar": "جودة المنحنى",     "kl": "pagh quv",
        "wa": "Qualité coûbe",    "ru": "Качество кривой",
        "hi": "वक्र गुणवत्ता",    "es": "Calidad curva",
        "pt": "Qualidade curva",  "bn": "বক্ররেখা মান",
    },
    "btn_reset_curve": {
        "fr": "↺ Reset",   "en": "↺ Reset",   "nl": "↺ Reset",
        "de": "↺ Reset",   "zh": "↺ 重置",     "ja": "↺ リセット",
        "ar": "↺ إعادة",   "kl": "↺ choH",    "wa": "↺ Rimete",
        "ru": "↺ Сброс",   "hi": "↺ रीसेट",  "es": "↺ Reset",
        "pt": "↺ Reset",   "bn": "↺ রিসেট",
    },
    "lbl_curve_preset": {
        "fr": "Préréglage",   "en": "Preset",       "nl": "Voorinstelling",
        "de": "Vorgabe",      "zh": "预设",          "ja": "プリセット",
        "ar": "الإعداد",     "kl": "wa'DIch",      "wa": "Préréglaedje",
        "ru": "Пресет",       "hi": "प्रीसेट",     "es": "Preajuste",
        "pt": "Predefinição", "bn": "প্রিসেট",
    },
    "curve_linear": {
        "fr": "Linéaire (neutre)",    "en": "Linear (neutral)",
        "nl": "Lineair (neutraal)",   "de": "Linear (neutral)",
        "zh": "线性 (中性)",           "ja": "リニア (中立)",
        "ar": "خطي (محايد)",          "kl": "linear",
        "wa": "Linéaire (neûte)",     "ru": "Линейная (нейтр.)",
        "hi": "रैखिक (तटस्थ)",       "es": "Lineal (neutral)",
        "pt": "Linear (neutro)",      "bn": "রৈখিক (নিরপেক্ষ)",
    },
    "curve_concave_soft": {
        "fr": "Concave légère",       "en": "Concave (soft)",
        "nl": "Concaaf (zacht)",      "de": "Konkav (sanft)",
        "zh": "凹形 (轻)",             "ja": "凹型 (弱)",
        "ar": "مقعّر (خفيف)",        "kl": "concave Sum",
        "wa": "Concåve ledjîre",      "ru": "Вогнутая (слабо)",
        "hi": "अवतल (हल्का)",        "es": "Cóncava (suave)",
        "pt": "Côncava (suave)",      "bn": "অবতল (হালকা)",
    },
    "curve_concave": {
        "fr": "Concave",              "en": "Concave",
        "nl": "Concaaf",              "de": "Konkav",
        "zh": "凹形",                  "ja": "凹型",
        "ar": "مقعّر",                "kl": "concave",
        "wa": "Concåve",              "ru": "Вогнутая",
        "hi": "अवतल",                 "es": "Cóncava",
        "pt": "Côncava",              "bn": "অবতল",
    },
    "curve_concave_strong": {
        "fr": "Concave forte",        "en": "Concave (strong)",
        "nl": "Concaaf (sterk)",      "de": "Konkav (stark)",
        "zh": "凹形 (强)",             "ja": "凹型 (強)",
        "ar": "مقعّر (قوي)",         "kl": "concave HoS",
        "wa": "Concåve foute",        "ru": "Вогнутая (сильно)",
        "hi": "अवतल (तीव्र)",         "es": "Cóncava (fuerte)",
        "pt": "Côncava (forte)",      "bn": "অবতল (শক্তিশালী)",
    },
    "curve_convex_soft": {
        "fr": "Convexe légère",       "en": "Convex (soft)",
        "nl": "Convex (zacht)",       "de": "Konvex (sanft)",
        "zh": "凸形 (轻)",             "ja": "凸型 (弱)",
        "ar": "محدّب (خفيف)",        "kl": "convex Sum",
        "wa": "Convexe ledjîre",      "ru": "Выпуклая (слабо)",
        "hi": "उत्तल (हल्का)",       "es": "Convexa (suave)",
        "pt": "Convexa (suave)",      "bn": "উত্তল (হালকা)",
    },
    "curve_convex": {
        "fr": "Convexe",              "en": "Convex",
        "nl": "Convex",               "de": "Konvex",
        "zh": "凸形",                  "ja": "凸型",
        "ar": "محدّب",                "kl": "convex",
        "wa": "Convexe",              "ru": "Выпуклая",
        "hi": "उत्तल",                "es": "Convexa",
        "pt": "Convexa",              "bn": "উত্তল",
    },
    "curve_convex_strong": {
        "fr": "Convexe forte",        "en": "Convex (strong)",
        "nl": "Convex (sterk)",       "de": "Konvex (stark)",
        "zh": "凸形 (强)",             "ja": "凸型 (強)",
        "ar": "محدّب (قوي)",         "kl": "convex HoS",
        "wa": "Convexe foute",        "ru": "Выпуклая (сильно)",
        "hi": "उत्तल (तीव्र)",        "es": "Convexa (fuerte)",
        "pt": "Convexa (forte)",      "bn": "উত্তল (শক্তিশালী)",
    },
    "curve_s": {
        "fr": "Courbe en S",          "en": "S-curve",
        "nl": "S-curve",              "de": "S-Kurve",
        "zh": "S 曲线",                "ja": "S字カーブ",
        "ar": "منحنى S",              "kl": "S pagh",
        "wa": "Coûbe è S",            "ru": "S-кривая",
        "hi": "S-वक्र",               "es": "Curva en S",
        "pt": "Curva em S",           "bn": "S-বক্ররেখা",
    },
    "curve_s_strong": {
        "fr": "Courbe en S forte",    "en": "S-curve (strong)",
        "nl": "S-curve (sterk)",      "de": "S-Kurve (stark)",
        "zh": "S 曲线 (强)",           "ja": "S字カーブ (強)",
        "ar": "منحنى S (قوي)",       "kl": "S pagh HoS",
        "wa": "Coûbe è S foute",      "ru": "S-кривая (сильно)",
        "hi": "S-वक्र (तीव्र)",       "es": "Curva en S (fuerte)",
        "pt": "Curva em S (forte)",   "bn": "S-বক্ররেখা (শক্তিশালী)",
    },
    "curve_inverse_s": {
        "fr": "S inversé",            "en": "Inverse S",
        "nl": "Omgekeerde S",         "de": "Inverses S",
        "zh": "反 S",                  "ja": "逆S字",
        "ar": "S معكوس",              "kl": "S nIHghoS",
        "wa": "S å rvier",            "ru": "Обратная S",
        "hi": "उलटा S",              "es": "S invertida",
        "pt": "S invertido",          "bn": "বিপরীত S",
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
    "dnd_hint": {
        "fr": "⬇  Déposez image(s) ici",
        "en": "⬇  Drop image(s) here",
        "nl": "⬇  Sleep afbeelding(en) hier",
        "de": "⬇  Bild(er) hier ablegen",
        "zh": "⬇  将图像拖放到此处",
        "ja": "⬇  ここに画像をドロップ",
        "ar": "⬇  أسقط الصور هنا",
        "kl": "⬇  nagh beQ vIngHa'",
        "wa": "⬇  Låche li(s) imådje(s) ci",
        "ru": "⬇  Перетащите изображения сюда",
        "hi": "⬇  यहाँ छवि खींचें",
        "es": "⬇  Arrastra imagen(es) aquí",
        "pt": "⬇  Solte imagem(ns) aqui",
        "bn": "⬇  এখানে ছবি টানুন",
    },
    "dnd_no_lib": {
        "fr": "(pip install tkinterdnd2 pour activer le glisser-déposer)",
        "en": "(pip install tkinterdnd2 to enable drag & drop)",
        "nl": "(pip install tkinterdnd2 voor drag & drop)",
        "de": "(pip install tkinterdnd2 für Drag & Drop)",
        "zh": "(pip install tkinterdnd2 启用拖放)",
        "ja": "(pip install tkinterdnd2 でDnDを有効化)",
        "ar": "(pip install tkinterdnd2 لتفعيل السحب والإفلات)",
        "kl": "(pip install tkinterdnd2 DnD)", "wa": "(pip install tkinterdnd2)",
        "ru": "(pip install tkinterdnd2 для DnD)",
        "hi": "(pip install tkinterdnd2)",     "es": "(pip install tkinterdnd2)",
        "pt": "(pip install tkinterdnd2)",     "bn": "(pip install tkinterdnd2)",
    },
    "dnd_loaded_rgb":   {
        "fr": "✓ RGB chargé automatiquement",   "en": "✓ RGB loaded automatically",
        "nl": "✓ RGB automatisch geladen",       "de": "✓ RGB automatisch geladen",
        "zh": "✓ RGB 自动加载",                  "ja": "✓ RGB 自動読み込み",
        "ar": "✓ RGB محمّل تلقائياً",           "kl": "✓ RGB automatic",
        "wa": "✓ RGB tchèrdjî automaticmint",   "ru": "✓ RGB загружен автоматически",
        "hi": "✓ RGB स्वतः लोड",                "es": "✓ RGB cargado automáticamente",
        "pt": "✓ RGB carregado automaticamente","bn": "✓ RGB স্বয়ংক্রিয় লোড",
    },
    "dnd_loaded_depth": {
        "fr": "✓ Z-Depth chargé automatiquement","en": "✓ Z-Depth loaded automatically",
        "nl": "✓ Z-Diepte automatisch geladen",  "de": "✓ Z-Tiefe automatisch geladen",
        "zh": "✓ Z深度自动加载",                  "ja": "✓ Zデプス自動読み込み",
        "ar": "✓ عمق Z محمّل تلقائياً",         "kl": "✓ Z-maS automatic",
        "wa": "✓ Z-Profond tchèrdjî automatic", "ru": "✓ Z-глубина загружена автоматически",
        "hi": "✓ Z-गहराई स्वतः लोड",           "es": "✓ Z-Profundidad cargada automáticamente",
        "pt": "✓ Z-Profundidade carregada auto.","bn": "✓ Z-গভীরতা স্বয়ংক্রিয় লোড",
    },
    "dnd_ambiguous": {
        "fr": "⚠ Les deux images semblent identiques — vérifiez l'assignation",
        "en": "⚠ Both images look similar — please check the assignment",
        "nl": "⚠ Beide afbeeldingen zien er hetzelfde uit — controleer de toewijzing",
        "de": "⚠ Beide Bilder sehen gleich aus — Zuweisung prüfen",
        "zh": "⚠ 两张图像看起来相似 — 请检查分配",
        "ja": "⚠ 両方の画像が似ています — 割り当てを確認してください",
        "ar": "⚠ الصورتان متشابهتان — يرجى التحقق من التعيين",
        "kl": "⚠ nagh beQ cha' ghap", "wa": "⚠ Li deus imådjes si resemblèt",
        "ru": "⚠ Оба изображения похожи — проверьте назначение",
        "hi": "⚠ दोनों छवियाँ समान दिखती हैं",
        "es": "⚠ Ambas imágenes parecen iguales — verifique la asignación",
        "pt": "⚠ Ambas as imagens parecem iguais — verifique a atribuição",
        "bn": "⚠ উভয় ছবি একই দেখাচ্ছে",
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
        "fr": "Charger une Depth Map",    "en": "Load Depth Map",
        "nl": "Depth Map laden",          "de": "Depth Map laden",
        "zh": "加载深度图",               "ja": "深度マップを開く",
        "ar": "تحميل خريطة العمق",       "kl": "maS tI'",
        "wa": "Tchèrdjî Mape di profond","ru": "Загрузить карту глубины",
        "hi": "डेप्थ मैप लोड करें",     "es": "Cargar mapa de profundidad",
        "pt": "Carregar mapa de profundidade","bn": "ডেপ্থ ম্যাপ লোড করুন",
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
    "kernel_pentagon": {
        "fr": "Pentagone (5 lames)",    "en": "Pentagon (5 blades)",
        "nl": "Vijfhoek (5 lamellen)",  "de": "Pentagon (5 Lamellen)",
        "zh": "五边形（5叶）",           "ja": "ペンタゴン（5枚羽）",
        "ar": "خماسي (5 شفرات)",       "kl": "vagh mIr",
        "wa": "Pentagone (5 lames)",    "ru": "Пентагон (5 лепестков)",
        "hi": "पंचभुज (5 ब्लेड)",     "es": "Pentágono (5 lamas)",
        "pt": "Pentágono (5 lâminas)",  "bn": "পেন্টাগন (৫ পাপড়ি)",
    },
    "kernel_octagon": {
        "fr": "Octogone (8 lames)",    "en": "Octagon (8 blades)",
        "nl": "Achthoek (8 lamellen)", "de": "Oktagon (8 Lamellen)",
        "zh": "八边形（8叶）",           "ja": "オクタゴン（8枚羽）",
        "ar": "ثماني (8 شفرات)",      "kl": "chorgh mIr",
        "wa": "Octogone (8 lames)",    "ru": "Октагон (8 лепестков)",
        "hi": "अष्टभुज (8 ब्लेड)",   "es": "Octágono (8 lamas)",
        "pt": "Octógono (8 lâminas)",  "bn": "অক্টাগন (৮ পাপড়ি)",
    },
    "kernel_ring": {
        "fr": "Anneau — catadioptre",      "en": "Ring — mirror lens",
        "nl": "Ring — spiegelobjectief",   "de": "Ring — Spiegelobjektiv",
        "zh": "环形 — 折返镜头",            "ja": "リング — 反射屈折レンズ",
        "ar": "حلقة — عدسة مرآة",        "kl": "gho — mIn taj",
        "wa": "Anea — catadioptre",        "ru": "Кольцо — катадиоптрик",
        "hi": "रिंग — मिरर लेंस",        "es": "Anillo — catadióptrico",
        "pt": "Anel — catadióptico",       "bn": "রিং — মিরর লেন্স",
    },
    "kernel_anamorphic": {
        "fr": "Anamorphique (cinéma)",     "en": "Anamorphic (cinema)",
        "nl": "Anamorfisch (bioscoop)",    "de": "Anamorphotisch (Kino)",
        "zh": "变形宽荧幕（椭圆）",         "ja": "アナモルフィック（映画）",
        "ar": "أنامورفي (سينما)",         "kl": "anamorphic",
        "wa": "Anamorphique (cinéma)",     "ru": "Анаморфный (кино)",
        "hi": "एनामॉर्फिक (सिनेमा)",     "es": "Anamórfico (cine)",
        "pt": "Anamórfico (cinema)",       "bn": "অ্যানামরফিক (সিনেমা)",
    },
    "kernel_star": {
        "fr": "Étoile ★ (6 branches)", "en": "Star ★ (6 points)",
        "nl": "Ster ★ (6 punten)",     "de": "Stern ★ (6 Zacken)",
        "zh": "星形 ★ (6角)",           "ja": "スター ★ (6先)",
        "ar": "نجمة ★ (6 رؤوس)",       "kl": "Hovtay' ★",
        "wa": "Étole ★ (6 branches)",  "ru": "Звезда ★ (6 лучей)",
        "hi": "स्टार ★ (6 नोक)",       "es": "Estrella ★ (6 puntas)",
        "pt": "Estrela ★ (6 pontas)",  "bn": "তারা ★ (৬ প্রান্ত)",
    },
    "kernel_heart": {
        "fr": "Cœur ♥",     "en": "Heart ♥",
        "nl": "Hart ♥",     "de": "Herz ♥",
        "zh": "心形 ♥",     "ja": "ハート ♥",
        "ar": "قلب ♥",     "kl": "muSHa' ♥",
        "wa": "Keûr ♥",    "ru": "Сердце ♥",
        "hi": "दिल ♥",     "es": "Corazón ♥",
        "pt": "Coração ♥", "bn": "হৃদয় ♥",
    },
    "kernel_custom": {
        "fr": "✏  Personnalisé (éditeur)",  "en": "✏  Custom (editor)",
        "nl": "✏  Aangepast (editor)",      "de": "✏  Benutzerdefiniert (Editor)",
        "zh": "✏  自定义（编辑器）",          "ja": "✏  カスタム（エディタ）",
        "ar": "✏  مخصص (محرر)",            "kl": "✏  Custom",
        "wa": "✏  Personåjhî (èditeur)",   "ru": "✏  Свой (редактор)",
        "hi": "✏  कस्टम (संपादक)",         "es": "✏  Personalizado (editor)",
        "pt": "✏  Personalizado (editor)",  "bn": "✏  কাস্টম (সম্পাদক)",
    },
    "lbl_bokeh_editor": {
        "fr": "Éditeur de forme bokeh",   "en": "Bokeh shape editor",
        "nl": "Bokeh-vormeditor",          "de": "Bokeh-Form-Editor",
        "zh": "散景形状编辑器",             "ja": "ボケ形状エディタ",
        "ar": "محرر شكل البوكيه",        "kl": "bIQ mIn chIS",
        "wa": "Èditeur di forme bokeh",   "ru": "Редактор формы боке",
        "hi": "बोके आकार संपादक",        "es": "Editor de forma bokeh",
        "pt": "Editor de forma bokeh",    "bn": "বোকে আকৃতি সম্পাদক",
    },
    "btn_reset_bokeh": {
        "fr": "↺  Réinitialiser",  "en": "↺  Reset",
        "nl": "↺  Terugzetten",    "de": "↺  Zurücksetzen",
        "zh": "↺  重置",           "ja": "↺  リセット",
        "ar": "↺  إعادة تعيين",   "kl": "↺  chu'",
        "wa": "↺  Remète",        "ru": "↺  Сброс",
        "hi": "↺  रीसेट",        "es": "↺  Restablecer",
        "pt": "↺  Repor",         "bn": "↺  রিসেট",
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
        "fr": "Par défaut",         "en": "Default",
        "nl": "Standaard",          "de": "Standard",
        "zh": "默认",               "ja": "デフォルト",
        "ar": "افتراضي",            "kl": "motlh",
        "wa": "Par défåt",          "ru": "По умолчанию",
        "hi": "डिफ़ॉल्ट",          "es": "Por defecto",
        "pt": "Padrão",             "bn": "ডিফল্ট",
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
    "lbl_edge_blur_title": {
        "fr": "Flou de bord  (par sujet)",
        "en": "Edge blur  (by subject)",
        "nl": "Randvervaging  (per onderwerp)",
        "de": "Randunschärfe  (per Subjekt)",
        "zh": "边缘模糊  (按主体)",
        "ja": "周辺ぼかし  (被写体基準)",
        "ar": "ضبابية الحافة  (حسب الموضوع)",
        "kl": "bIQ Hem  (jan)",
        "wa": "Flou di bwès  (pa soumèt)",
        "ru": "Краевое размытие  (по субъекту)",
        "hi": "किनारा धुंध  (विषय अनुसार)",
        "es": "Desenfoque de borde  (por sujeto)",
        "pt": "Desfoque de borda  (por sujeito)",
        "bn": "প্রান্ত ঝাপসা  (বিষয় অনুযায়ী)",
    },
    "btn_pick_subject": {
        "fr": "📍  Cliquer le sujet sur l'image",
        "en": "📍  Click subject on the image",
        "nl": "📍  Klik het onderwerp op de afbeelding",
        "de": "📍  Subjekt auf Bild klicken",
        "zh": "📍  在图像上点击主体",
        "ja": "📍  画像上で被写体をクリック",
        "ar": "📍  انقر على الموضوع في الصورة",
        "kl": "📍  jan nagh'e' Hoch",
        "wa": "📍  Cliker li soumèt su l'imådje",
        "ru": "📍  Кликнуть субъект на изображении",
        "hi": "📍  छवि पर विषय क्लिक करें",
        "es": "📍  Hacer clic en el sujeto de la imagen",
        "pt": "📍  Clicar no sujeito na imagem",
        "bn": "📍  ছবিতে বিষয় ক্লিক করুন",
    },
    "btn_clear_subject": {
        "fr": "✕  Effacer sujet",
        "en": "✕  Clear subject",
        "nl": "✕  Onderwerp wissen",
        "de": "✕  Subjekt löschen",
        "zh": "✕  清除主体",
        "ja": "✕  被写体をクリア",
        "ar": "✕  مسح الموضوع",
        "kl": "✕  jan teH",
        "wa": "✕  Efacer soumèt",
        "ru": "✕  Очистить субъект",
        "hi": "✕  विषय साफ करें",
        "es": "✕  Borrar sujeto",
        "pt": "✕  Limpar sujeito",
        "bn": "✕  বিষয় মুছুন",
    },
    "btn_pick_near": {
        "fr": "🎯  Cliquer limite proche sur image",
        "en": "🎯  Click near limit on image",
        "nl": "🎯  Klik nabij-grens op afbeelding",
        "de": "🎯  Nahgrenze im Bild klicken",
        "zh": "🎯  在图像上点击近端界限",
        "ja": "🎯  画像で手前の境界をクリック",
        "ar": "🎯  انقر الحد القريب في الصورة",
        "kl": "🎯  Qav tuqDaq nagh Hoch",
        "wa": "🎯  Cliker li limita di près su l'imådje",
        "ru": "🎯  Клик ближний предел на изображении",
        "hi": "🎯  छवि पर निकट सीमा क्लिक करें",
        "es": "🎯  Clic límite cercano en imagen",
        "pt": "🎯  Clicar limite próximo na imagem",
        "bn": "🎯  ছবিতে কাছের সীমা ক্লিক করুন",
    },
    "btn_pick_far": {
        "fr": "🎯  Cliquer limite lointaine sur image",
        "en": "🎯  Click far limit on image",
        "nl": "🎯  Klik verre grens op afbeelding",
        "de": "🎯  Ferngrenze im Bild klicken",
        "zh": "🎯  在图像上点击远端界限",
        "ja": "🎯  画像で遠方の境界をクリック",
        "ar": "🎯  انقر الحد البعيد في الصورة",
        "kl": "🎯  Qav HopDaq nagh Hoch",
        "wa": "🎯  Cliker li limita di lon su l'imådje",
        "ru": "🎯  Клик дальний предел на изображении",
        "hi": "🎯  छवि पर दूर सीमा क्लिक करें",
        "es": "🎯  Clic límite lejano en imagen",
        "pt": "🎯  Clicar limite distante na imagem",
        "bn": "🎯  ছবিতে দূরের সীমা ক্লিক করুন",
    },
    "lbl_edge_strength": {
        "fr": "Intensité flou de bord",
        "en": "Edge blur strength",
        "nl": "Randvervaging sterkte",
        "de": "Randunschärfe Stärke",
        "zh": "边缘模糊强度",
        "ja": "周辺ぼかし強度",
        "ar": "شدة ضبابية الحافة",
        "kl": "bIQ Hem HoS",
        "wa": "Force flou di bwès",
        "ru": "Интенс. краевого разм.",
        "hi": "किनारा धुंध तीव्रता",
        "es": "Intensidad desenfoque borde",
        "pt": "Intensidade desfoque borda",
        "bn": "প্রান্ত ঝাপসা তীব্রতা",
    },
    "lbl_edge_radius": {
        "fr": "Rayon horizontal",   "en": "Horizontal radius",
        "nl": "Horizontale straal", "de": "Horizontaler Radius",
        "zh": "水平半径",            "ja": "水平半径",
        "ar": "نصف قطر أفقي",      "kl": "bIQ Daq H",
        "wa": "Rèyon orizontal",   "ru": "Горизонтальный радиус",
        "hi": "क्षैतिज त्रिज्या",  "es": "Radio horizontal",
        "pt": "Raio horizontal",   "bn": "অনুভূমিক ব্যাসার্ধ",
    },
    "lbl_edge_ry": {
        "fr": "Rayon vertical",     "en": "Vertical radius",
        "nl": "Verticale straal",   "de": "Vertikaler Radius",
        "zh": "垂直半径",            "ja": "垂直半径",
        "ar": "نصف قطر رأسي",      "kl": "bIQ Daq V",
        "wa": "Rèyon vèrtical",    "ru": "Вертикальный радиус",
        "hi": "ऊर्ध्वाधर त्रिज्या","es": "Radio vertical",
        "pt": "Raio vertical",     "bn": "উল্লম্ব ব্যাসার্ধ",
    },
    "subject_status_none": {
        "fr": "(aucun sujet défini)",
        "en": "(no subject defined)",
        "nl": "(geen onderwerp)",
        "de": "(kein Subjekt)",
        "zh": "(未定义主体)",
        "ja": "(被写体未設定)",
        "ar": "(لا موضوع محدد)",
        "kl": "(jan Hutlh)",
        "wa": "(nén soumèt)",
        "ru": "(субъект не задан)",
        "hi": "(कोई विषय नहीं)",
        "es": "(ningún sujeto)",
        "pt": "(nenhum sujeito)",
        "bn": "(কোনো বিষয় নেই)",
    },
    "picking_prompt": {
        "fr": "→ Cliquez sur le sujet dans l'image",
        "en": "→ Click the subject in the image",
        "nl": "→ Klik het onderwerp in de afbeelding",
        "de": "→ Klicken Sie das Subjekt im Bild an",
        "zh": "→ 在图像中点击主体",
        "ja": "→ 画像内の被写体をクリック",
        "ar": "→ انقر على الموضوع في الصورة",
        "kl": "→ jan nagh'e' Hoch",
        "wa": "→ Cliker li soumèt dins l'imådje",
        "ru": "→ Кликните субъект на изображении",
        "hi": "→ छवि में विषय क्लिक करें",
        "es": "→ Haga clic en el sujeto de la imagen",
        "pt": "→ Clique no sujeito na imagem",
        "bn": "→ ছবিতে বিষয় ক্লিক করুন",
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
    "overlay_legend_edge": {
        "fr": "■ Flou de bord (sujet)",  "en": "■ Edge blur (subject)",
        "nl": "■ Randvervaging (subj.)", "de": "■ Randunschärfe (Subj.)",
        "zh": "■ 边缘模糊 (主体)",        "ja": "■ 周辺ぼかし (被写体)",
        "ar": "■ ضبابية حافة (موضوع)",  "kl": "■ bIQ Hem (jan)",
        "wa": "■ Flou bwès (soumèt)",   "ru": "■ Краев. разм. (субъект)",
        "hi": "■ किनारा धुंध (विषय)",   "es": "■ Desfoque borde (sujeto)",
        "pt": "■ Desfoque borda (suj.)", "bn": "■ প্রান্ত ঝাপসা (বিষয়)",
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
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    xn = x / radius
    yn = y / radius
    bound = 1.0 / (math.sqrt(3) / 2)
    mask = ((np.abs(xn) <= 1.0) & (np.abs(yn) <= 1.0) &
            (np.abs(xn + yn) <= bound) & (np.abs(xn - yn) <= bound))
    k = mask.astype(np.float32)
    s = k.sum()
    return k / s if s > 0 else make_disk_kernel(radius)


def make_gaussian_kernel(radius: float) -> np.ndarray:
    r = max(int(math.ceil(radius * 2)), 1)
    sigma = radius / 2.0
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    k = np.exp(-(x * x + y * y) / (2 * sigma * sigma)).astype(np.float32)
    k /= k.sum()
    return k


def make_polygon_kernel(radius: float, n_sides: int) -> np.ndarray:
    """Polygone régulier à n côtés (pentagone, octogone, etc.)."""
    r = max(int(math.ceil(radius)), 1)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    inradius = radius * math.cos(math.pi / n_sides)
    offset = -math.pi / 2  # pointe vers le haut
    inside = np.ones((2*r+1, 2*r+1), dtype=bool)
    for i in range(n_sides):
        a = 2*math.pi*i/n_sides + offset
        inside &= (x * math.cos(a) + y * math.sin(a)) <= inradius + 1e-6
    k = inside.astype(np.float32)
    s = k.sum()
    return k / s if s > 0 else make_disk_kernel(radius)


def make_pentagon_kernel(radius: float) -> np.ndarray:
    """Diaphragme pentagone (5 lames)."""
    return make_polygon_kernel(radius, 5)


def make_octagon_kernel(radius: float) -> np.ndarray:
    """Diaphragme octogone (8 lames)."""
    return make_polygon_kernel(radius, 8)


def make_ring_kernel(radius: float) -> np.ndarray:
    """Anneau / donut — objectif catadioptre (miroir)."""
    r = max(int(math.ceil(radius)), 1)
    y, x = np.ogrid[-r:r+1, -r:r+1]
    dist2 = (x*x + y*y).astype(np.float32)
    inner = (radius * 0.45) ** 2
    outer = radius ** 2
    mask = (dist2 <= outer) & (dist2 >= inner)
    k = mask.astype(np.float32)
    s = k.sum()
    return k / s if s > 0 else make_disk_kernel(radius)


def make_anamorphic_kernel(radius: float) -> np.ndarray:
    """Ellipse horizontale — objectif anamorphique (cinéma)."""
    r = max(int(math.ceil(radius)), 1)
    y, x = np.ogrid[-r:r+1, -r:r+1]
    aspect = 0.32          # compression verticale typique des anamorphiques
    mask = (x.astype(np.float32) / radius)**2 + \
           (y.astype(np.float32) / (radius * aspect))**2 <= 1.0
    k = mask.astype(np.float32)
    s = k.sum()
    return k / s if s > 0 else make_disk_kernel(radius)


def make_star_kernel(radius: float) -> np.ndarray:
    """Étoile à 6 branches (effet bokeh star-filter)."""
    r  = max(int(math.ceil(radius)), 1)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    n  = 6                 # branches
    ir = radius * 0.38     # rayon intérieur (vallées)
    dist = np.sqrt(x*x + y*y).astype(np.float32)
    angle = np.arctan2(y.astype(np.float32) * np.ones_like(x, dtype=np.float32),
                       x.astype(np.float32) * np.ones_like(y, dtype=np.float32))
    boundary = ir + (radius - ir) * (np.cos(n * angle) + 1.0) * 0.5
    mask = dist <= boundary
    mask[r, r] = True      # centre toujours plein
    k = mask.astype(np.float32)
    s = k.sum()
    return k / s if s > 0 else make_disk_kernel(radius)


def make_heart_kernel(radius: float) -> np.ndarray:
    """Cœur ♥ — bokeh romantique (courbe implicite)."""
    r = max(int(math.ceil(radius)), 1)
    sc = 1.25 / max(radius, 1)
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    x = (xx * sc).astype(np.float32)
    y = (-yy * sc + 0.15).astype(np.float32)
    # Courbe implicite du cœur : (x²+y²-1)³ ≤ x²·y³
    val = (x*x + y*y - 1.0)**3 - x*x * y**3
    mask = val <= 0.0
    k = mask.astype(np.float32)
    s = k.sum()
    return k / s if s > 0 else make_disk_kernel(radius)


def make_custom_polygon_kernel(radius: float, pts_normalized) -> np.ndarray:
    """
    Génère un noyau bokeh à partir d'un polygone quelconque.
    pts_normalized : liste de (x, y) dans [-1, 1]²
    """
    if not pts_normalized or len(pts_normalized) < 3:
        return make_disk_kernel(radius)
    r  = max(int(math.ceil(radius)), 1)
    sc = np.array([(x * radius, y * radius) for x, y in pts_normalized],
                  dtype=np.float32)
    n  = len(sc)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)

    inside = np.zeros_like(xx, dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = sc[i]
        xj, yj = sc[j]
        cond = ((yi > yy) != (yj > yy)) & (
            xx < (xj - xi) * (yy - yi) / (yj - yi + 1e-9) + xi)
        inside ^= cond
        j = i

    k = inside.astype(np.float32)
    s = k.sum()
    return k / s if s > 0 else make_disk_kernel(radius)


# Points personnalisés (partagés entre UI et render thread)
_CUSTOM_BOKEH_PTS = [(math.cos(2*math.pi*i/16 - math.pi/2),
                      math.sin(2*math.pi*i/16 - math.pi/2))
                     for i in range(16)]


def _make_custom_kernel_global(radius: float) -> np.ndarray:
    return make_custom_polygon_kernel(radius, _CUSTOM_BOKEH_PTS)


KERNEL_FN = {
    "disk":        make_disk_kernel,
    "hex":         make_hex_kernel,
    "gauss":       make_gaussian_kernel,
    "pentagon":    make_pentagon_kernel,
    "octagon":     make_octagon_kernel,
    "ring":        make_ring_kernel,
    "anamorphic":  make_anamorphic_kernel,
    "star":        make_star_kernel,
    "heart":       make_heart_kernel,
    "custom":      _make_custom_kernel_global,
}
KERNEL_KEYS = ["disk", "hex", "gauss", "pentagon", "octagon",
               "ring", "anamorphic", "star", "heart", "custom"]
KERNEL_T    = ["kernel_disk", "kernel_hex", "kernel_gauss",
               "kernel_pentagon", "kernel_octagon",
               "kernel_ring", "kernel_anamorphic",
               "kernel_star", "kernel_heart", "kernel_custom"]


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
            return apply_kernel(img, kernel)   # retombe sur le backend suivant

    if GPU_BACKEND in ("torch_cuda", "torch_mps", "torch_cpu"):
        kH, kW = kernel.shape
        pH, pW = kH // 2, kW // 2
        t = (torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
             .unsqueeze(0).float().to(_TORCH_DEVICE))
        t = _TF.pad(t, (pW, pW, pH, pH), mode="reflect")
        k = (torch.from_numpy(kernel).float()
             .unsqueeze(0).unsqueeze(0)
             .expand(3, 1, kH, kW).contiguous().to(_TORCH_DEVICE))
        out = _TF.conv2d(t, k, groups=3)
        return out.squeeze(0).permute(1, 2, 0).cpu().numpy()

    if HAS_SCIPY:
        return np.stack([
            convolve(img[:, :, c].astype(np.float32), kernel, mode="reflect")
            for c in range(3)
        ], axis=2)

    if HAS_CV2:
        return cv2.filter2D(img.astype(np.float32), -1, kernel,
                            borderType=cv2.BORDER_REFLECT)

    from PIL import ImageFilter
    r = int((kernel.shape[0] - 1) / 4)
    pil = Image.fromarray(img.astype(np.uint8))
    return np.array(pil.filter(ImageFilter.GaussianBlur(radius=max(r, 1))),
                    dtype=np.float32)


# =============================================================================
# Détection automatique NB / couleur
# =============================================================================

def detect_image_type(path: str) -> str:
    """
    Retourne "depth" si l'image est en niveaux de gris (Z-depth),
             "rgb"   si elle est en couleurs.

    Heuristiques (par ordre de priorité) :
    1. Mode PIL natif L/I/F/I;16 → depth
    2. Image RGB dont les 3 canaux sont quasi-identiques → depth
    3. Sinon → rgb
    """
    try:
        with Image.open(path) as img:
            mode = img.mode

            # ── Mode nativement NB / depth ──────────────────────────────────────
            if mode in ("L", "I", "F", "I;16", "I;16B", "I;16S", "I;16BS", "P"):
                return "depth"

            # ── Format dont le nom suggère une depth map ─────────────────────────
            ext = os.path.splitext(path)[1].lower()
            if ext in (".pgm", ".pfm", ".exr", ".hdr"):
                return "depth"

            # ── Image RGB : vérifier la colorimétrie sur un échantillon ─────────
            sample = img.convert("RGB")
            # Réduire à 128×128 max pour la vitesse
            thumb  = sample.resize(
                (min(sample.width, 128), min(sample.height, 128)), _NEAREST)
            arr    = np.array(thumb, dtype=np.float32)
            r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            # Écart-type des différences inter-canaux
            std_rg = float(np.std(r - g))
            std_rb = float(np.std(r - b))
            std_gb = float(np.std(g - b))
            colorfulness = (std_rg + std_rb + std_gb) / 3.0

            # Seuil empirique : < 6 → image sans vraies couleurs
            return "depth" if colorfulness < 6.0 else "rgb"

    except Exception:
        # En cas d'erreur (format inconnu, etc.), supposer RGB
        return "rgb"


def parse_dnd_paths(data: str) -> list:
    """
    Parse la chaîne brute d'un événement DnD tkinterdnd2 :
    gère les espaces dans les noms de fichiers ({...}) et les chemins simples.
    """
    paths = []
    for m in re.finditer(r'\{([^}]+)\}|(\S+)', data):
        p = m.group(1) or m.group(2)
        if p and os.path.isfile(p):
            paths.append(p)
    return paths


# =============================================================================
# Chargement depth
# =============================================================================

def load_depth(path: str, invert: bool = False) -> np.ndarray:
    """
    Chargement robuste d'une depth map — tous formats :
    • 16-bit PNG (mode I;16, I;16B, I) → précision complète 65 535 niveaux
    • 8-bit PNG/JPG/BMP/WebP           → 255 niveaux
    • TIFF (8, 16, 32-bit float)       → précision native
    • PGM / PPM / PBM                  → précision native
    • OpenEXR (.exr) via OpenCV        → HDR float32
    • HDR / PFM                        → float32
    • RGB/RGBA                         → converti en niveaux de gris
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
# v12 — Remap de la courbe de profondeur (contrôle utilisateur)
# =============================================================================

def remap_depth_curve(depth: np.ndarray, curve: float) -> np.ndarray:
    """
    Remappe la courbe tonale de la depth map pour accentuer l'avant ou l'arrière.

    curve ∈ [-1, +1] :
      •  0.0 → identité (aucun effet)
      • > 0  → pousse les valeurs vers le HAUT (zones claires étirées)
               → accentue le côté "proche" (par défaut — dépend de "invert")
      • < 0  → pousse les valeurs vers le BAS (zones sombres étirées)
               → accentue le côté "lointain"

    Implémenté par une loi de puissance : new = depth ** gamma,
    avec gamma = 4 ** (-curve), donc :
      curve = +1  →  gamma = 0.25  (très forte accentuation des valeurs hautes)
      curve =  0  →  gamma = 1.0   (identité)
      curve = -1  →  gamma = 4.0   (très forte accentuation des valeurs basses)
    """
    if abs(curve) < 1e-4:
        return depth
    gamma = 4.0 ** (-float(curve))
    # Clip sécurité avant la puissance (les depth maps sont déjà normalisées
    # dans load_depth, mais on peut recevoir des valeurs légèrement hors [0,1]
    # après un resize bilinéaire).
    d = np.clip(depth, 0.0, 1.0)
    return np.power(d, gamma).astype(np.float32)


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

    # Sécurité : empêcher les transitions de déborder hors du domaine [0, 1]
    # qui produisaient des divisions par ~zéro et des cartes de flou aberrantes.
    falloff_near = max(min(falloff_near, focus_near),       0.0)
    falloff_far  = max(min(falloff_far,  1.0 - focus_far),  0.0)

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

    # ── Normalisation finale ──────────────────────────────────────────────────
    # Les zones smoothstep produisent des valeurs ∈ [0, 1].
    # Les zones « hard » (au-delà de focus ± falloff) produisent ∈ [1, 2].
    # On clippe à [0, 2] et on remappe sur [0, max_blur] de manière linéaire :
    # le résultat est désormais prévisible et indépendant du contenu de la scène
    # (l'ancienne renormalisation par blur.max() pouvait amplifier ou écraser
    #  le flou selon que la scène atteignait ou non les zones « hard »).
    blur = np.clip(blur, 0.0, 2.0) * (max_blur / 2.0)
    return blur


# =============================================================================
# Flou de bord radial (edge blur par sujet)
# =============================================================================

def apply_edge_blur(blur_map: np.ndarray,
                    subject_xy,
                    edge_strength: float,
                    edge_rx: float,
                    edge_ry: float,
                    max_blur: float) -> np.ndarray:
    """
    Ajoute un flou de bord radial elliptique :
    Les pixels nets éloignés du sujet reçoivent un léger flou supplémentaire.

    subject_xy   : (x, y) normalisé [0,1]
    edge_strength: 0.0 = aucun effet · 1.0 = intensité maximale
    edge_rx      : demi-axe horizontal [0,1] (normalisé par la diagonale)
    edge_ry      : demi-axe vertical   [0,1]
    max_blur     : rayon max de flou dans la scène
    """
    if edge_strength <= 1e-4 or subject_xy is None:
        return blur_map

    sx, sy = subject_xy
    H, W   = blur_map.shape
    ys, xs = np.ogrid[:H, :W]

    rx = max(edge_rx, 1e-4)
    ry = max(edge_ry, 1e-4)
    # Distance elliptique normalisée (= 1 à la frontière de l'ellipse)
    dist = np.sqrt(((xs / W - sx) / rx) ** 2
                 + ((ys / H - sy) / ry) ** 2).astype(np.float32)

    t      = np.clip(dist, 0.0, 1.0)
    weight = t * t * (3.0 - 2.0 * t)   # smoothstep

    sharp_factor = np.clip(1.0 - blur_map / max(max_blur, 1e-6),
                           0.0, 1.0).astype(np.float32)
    edge = (edge_strength * max_blur * weight * sharp_factor).astype(np.float32)
    return blur_map + edge


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
            return _dilate(mask, radius)
    if GPU_BACKEND in ("torch_cuda", "torch_mps", "torch_cpu"):
        # Sur CPU, scipy est plus rapide qu'un max_pool torch
        if GPU_BACKEND == "torch_cpu" and HAS_SCIPY:
            return maximum_filter(mask, size=ks)
        t = (torch.from_numpy(mask).float()
             .unsqueeze(0).unsqueeze(0).to(_TORCH_DEVICE))
        t = _TF.pad(t, (r, r, r, r), mode="replicate")
        t = _TF.max_pool2d(t, kernel_size=ks, stride=1, padding=0)
        return t.squeeze().cpu().numpy()
    if HAS_SCIPY:
        return maximum_filter(mask, size=ks)
    if HAS_CV2:
        k = np.ones((ks, ks), dtype=np.uint8)
        return cv2.dilate((mask * 255).astype(np.uint8), k).astype(np.float32) / 255.0
    return mask


def _erode(mask: np.ndarray, radius: float) -> np.ndarray:
    r = max(int(math.ceil(radius)), 1)
    ks = 2 * r + 1
    if GPU_BACKEND == "cupy" and _CUPY_OK:
        try:
            return cp.asnumpy(_cp_minfilt(cp.asarray(mask), size=ks))
        except Exception as _e:
            _cupy_failed(_e)
            return _erode(mask, radius)
    if GPU_BACKEND in ("torch_cuda", "torch_mps", "torch_cpu"):
        if GPU_BACKEND == "torch_cpu" and HAS_SCIPY:
            return minimum_filter(mask, size=ks)
        t = (torch.from_numpy(mask).float()
             .unsqueeze(0).unsqueeze(0).to(_TORCH_DEVICE))
        t = _TF.pad(t, (r, r, r, r), mode="replicate")
        t = -_TF.max_pool2d(-t, kernel_size=ks, stride=1, padding=0)
        return t.squeeze().cpu().numpy()
    if HAS_SCIPY:
        return minimum_filter(mask, size=ks)
    if HAS_CV2:
        k = np.ones((ks, ks), dtype=np.uint8)
        return cv2.erode((mask * 255).astype(np.uint8), k).astype(np.float32) / 255.0
    return mask


def _smooth(mask: np.ndarray, sigma: float) -> np.ndarray:
    if GPU_BACKEND == "cupy" and _CUPY_OK:
        try:
            return cp.asnumpy(_cp_gaussian(cp.asarray(mask), sigma=sigma))
        except Exception as _e:
            _cupy_failed(_e)
            return _smooth(mask, sigma)
    if GPU_BACKEND in ("torch_cuda", "torch_mps", "torch_cpu"):
        # Convolution gaussienne SÉPARABLE (2 conv1d K au lieu d'un conv2d K×K).
        ks  = max(int(math.ceil(sigma * 3)) * 2 + 1, 3)
        dev = _TORCH_DEVICE
        x   = torch.arange(ks, dtype=torch.float32, device=dev) - ks // 2
        gk  = torch.exp(-x * x / (2.0 * sigma * sigma))
        gk  = gk / gk.sum()
        kx  = gk.view(1, 1, 1, ks)
        ky  = gk.view(1, 1, ks, 1)
        t   = (torch.from_numpy(mask).float()
               .unsqueeze(0).unsqueeze(0).to(dev))
        t   = _TF.pad(t, (ks // 2, ks // 2, 0, 0), mode="reflect")
        t   = _TF.conv2d(t, kx)
        t   = _TF.pad(t, (0, 0, ks // 2, ks // 2), mode="reflect")
        t   = _TF.conv2d(t, ky)
        return t.squeeze().cpu().numpy()
    if HAS_SCIPY:
        return gaussian_filter(mask, sigma=sigma)
    if HAS_CV2:
        ks = max(int(math.ceil(sigma * 3)) * 2 + 1, 3)
        return cv2.GaussianBlur(mask, (ks, ks), sigma)
    return mask


# =============================================================================
# Overlay zones DoF
# =============================================================================

def make_zone_overlay(depth, focus_near, focus_far, falloff_near, falloff_far=None):
    if falloff_far is None:
        falloff_far = falloff_near
    falloff_near = max(min(falloff_near, focus_near),      0.0)
    falloff_far  = max(min(falloff_far,  1.0 - focus_far), 0.0)
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


def make_edge_blur_overlay(shape, subject_xy, edge_strength, edge_rx, edge_ry):
    """
    Génère un overlay RGBA (violet) montrant la zone de flou de bord elliptique.
    """
    H, W = shape[:2]
    out  = np.zeros((H, W, 4), dtype=np.uint8)
    if subject_xy is None or edge_strength <= 1e-4:
        return out

    sx, sy = subject_xy
    ys, xs = np.ogrid[:H, :W]
    rx = max(edge_rx, 1e-4)
    ry = max(edge_ry, 1e-4)

    dist   = np.sqrt(((xs / W - sx) / rx) ** 2
                   + ((ys / H - sy) / ry) ** 2).astype(np.float32)
    t      = np.clip(dist, 0.0, 1.0)
    weight = t * t * (3.0 - 2.0 * t)

    alpha = np.clip(weight * edge_strength * 200, 0, 255).astype(np.uint8)
    out[:, :, 0] = 176
    out[:, :, 1] = 78
    out[:, :, 2] = 255
    out[:, :, 3] = alpha

    # Ellipse de contour (frontière visible) — anneau fin à distance ≈ 1
    ring2 = np.abs(dist - 1.0) < 0.03
    out[ring2, 0] = 255
    out[ring2, 1] = 200
    out[ring2, 2] = 255
    out[ring2, 3] = 180
    return out


# =============================================================================
# v12 — Helpers torch « résidents » : opèrent sur des tenseurs déjà sur GPU.
#       Permettent à _render_dof_torch d'éviter les transferts host↔device
#       qui dominaient le temps de rendu sur grosses images.
# =============================================================================

def _t_dilate(mask_2d, radius):
    """Dilation morphologique sur tenseur 2D (H, W) résident."""
    r  = max(int(math.ceil(radius)), 1)
    ks = 2 * r + 1
    x  = mask_2d.unsqueeze(0).unsqueeze(0)
    x  = _TF.pad(x, (r, r, r, r), mode="replicate")
    x  = _TF.max_pool2d(x, kernel_size=ks, stride=1, padding=0)
    return x.squeeze(0).squeeze(0)


def _t_erode(mask_2d, radius):
    """Érosion morphologique sur tenseur 2D résident."""
    return -_t_dilate(-mask_2d, radius)


def _t_smooth(mask_2d, sigma):
    """Flou gaussien séparable sur tenseur 2D résident."""
    ks  = max(int(math.ceil(sigma * 3)) * 2 + 1, 3)
    dev = mask_2d.device
    x   = torch.arange(ks, dtype=torch.float32, device=dev) - ks // 2
    gk  = torch.exp(-x * x / (2.0 * sigma * sigma))
    gk  = gk / gk.sum()
    kx  = gk.view(1, 1, 1, ks)
    ky  = gk.view(1, 1, ks, 1)
    t   = mask_2d.unsqueeze(0).unsqueeze(0)
    t   = _TF.pad(t, (ks // 2, ks // 2, 0, 0), mode="reflect")
    t   = _TF.conv2d(t, kx)
    t   = _TF.pad(t, (0, 0, ks // 2, ks // 2), mode="reflect")
    t   = _TF.conv2d(t, ky)
    return t.squeeze(0).squeeze(0)


def _t_uniform3(mask_2d):
    """uniform_filter size=3 sur tenseur 2D résident."""
    dev = mask_2d.device
    k   = torch.full((1, 1, 3, 3), 1.0 / 9.0, device=dev, dtype=torch.float32)
    x   = mask_2d.unsqueeze(0).unsqueeze(0)
    x   = _TF.pad(x, (1, 1, 1, 1), mode="reflect")
    return _TF.conv2d(x, k).squeeze(0).squeeze(0)


def _t_apply_kernel(img_bchw, kernel_2d):
    """
    Convolution séparée par canal sur tenseur image (1, 3, H, W) résident.
    kernel_2d : tenseur (kH, kW) déjà sur le bon device.
    """
    kH, kW = kernel_2d.shape
    pH, pW = kH // 2, kW // 2
    k      = (kernel_2d.unsqueeze(0).unsqueeze(0)
              .expand(3, 1, kH, kW).contiguous())
    x      = _TF.pad(img_bchw, (pW, pW, pH, pH), mode="reflect")
    return _TF.conv2d(x, k, groups=3)


def _render_dof_torch(rgb, blur_map, depth, kernel_key="disk", steps=16,
                     progress_cb=None, lang="fr",
                     focus_near=0.35, focus_far=0.65):
    """
    Rendu DOF en couches séparées avec **résidence GPU** (v12).

    Toute la pipeline (pre-fill, dilations, convolutions, composition)
    s'exécute sur tenseurs torch maintenus sur le device. Les seuls
    transferts host↔device sont :
      • upload initial : rgb, blur_map, depth (3 tenseurs)
      • upload des kernels (petits, ~K² float32)
      • download final : 1 tenseur résultat (uint8)

    Sur une image 4K et 16 passes, cela évite ~70 allers-retours
    host↔device par rapport au chemin numpy → typiquement ×3 à ×5
    sur GPU CUDA / MPS.
    """
    dev   = _TORCH_DEVICE
    h, w  = rgb.shape[:2]
    maker = KERNEL_FN.get(kernel_key, make_disk_kernel)

    # ── Upload unique vers le GPU ─────────────────────────────────────────────
    rgb_t   = (torch.from_numpy(rgb.astype(np.float32))      # (H, W, 3)
               .permute(2, 0, 1).unsqueeze(0)                # (1, 3, H, W)
               .contiguous().to(dev))
    blur_t  = torch.from_numpy(blur_map.astype(np.float32)).to(dev)   # (H, W)
    depth_t = torch.from_numpy(depth.astype(np.float32)).to(dev)      # (H, W)

    max_r = float(blur_t.max().item())
    if max_r < 0.5:
        return rgb.copy()

    # ── Lissage du depth (uniform 3×3) ────────────────────────────────────────
    depth_s = _t_uniform3(depth_t)

    # ── Séparation focus-aware par sigmoid douce ──────────────────────────────
    depth_mid = (focus_near + focus_far) / 2.0
    d_range   = max(float((depth_s.max() - depth_s.min()).item()), 1e-6)
    softness  = d_range * 0.08
    depth_blend  = torch.sigmoid((depth_s - depth_mid) / softness)
    blurred_mask = (blur_t >= 0.5).float()

    bg_pixels = (1.0 - depth_blend) * blurred_mask
    fg_pixels = depth_blend          * blurred_mask
    sharp_mask_raw = (blur_t < 0.5).float()

    # ── Protection plafonnée ──────────────────────────────────────────────────
    _PROT_CAP  = 8.0
    _SIGMA_CAP = 5.0
    _FILL_CAP  = 12.0

    prot_radius = min(max(max_r * 0.35, 2.0), _PROT_CAP)
    prot_sigma  = min(max(max_r * 0.25, 1.5), _SIGMA_CAP)

    sharp_protection = _t_smooth(_t_dilate(sharp_mask_raw, prot_radius),
                                 sigma=prot_sigma).clamp(0.0, 1.0)

    fg_solid      = (depth_blend >= 0.5).float()
    fg_protection = _t_smooth(
        _t_dilate(fg_solid, min(max(max_r * 0.30, 2.0), _PROT_CAP)),
        sigma=prot_sigma).clamp(0.0, 1.0)
    bg_allowed = (1.0 - sharp_protection - fg_protection).clamp(0.0, 1.0)
    fg_allowed = 1.0 - sharp_protection

    # ── PRE-FILL anti-contamination ───────────────────────────────────────────
    bg_source = (bg_pixels + (1.0 - blurred_mask) * (1.0 - depth_blend)
                ).clamp(0.0, 1.0)
    fg_source = (fg_pixels + (1.0 - blurred_mask) * depth_blend
                ).clamp(0.0, 1.0)

    fill_kernel_np = maker(min(max(max_r * 0.6, 3.0), _FILL_CAP))
    fill_kernel_t  = torch.from_numpy(fill_kernel_np).float().to(dev)

    _PREFILL_ITERS = 2

    rgb_for_bg = rgb_t.clone()
    bg_src_4d  = bg_source.unsqueeze(0).unsqueeze(0)         # (1,1,H,W)
    for _ in range(_PREFILL_ITERS):
        filled     = _t_apply_kernel(rgb_for_bg, fill_kernel_t)
        rgb_for_bg = rgb_for_bg * bg_src_4d + filled * (1.0 - bg_src_4d)

    rgb_for_fg = rgb_t.clone()
    fg_src_4d  = fg_source.unsqueeze(0).unsqueeze(0)
    for _ in range(_PREFILL_ITERS):
        filled     = _t_apply_kernel(rgb_for_fg, fill_kernel_t)
        rgb_for_fg = rgb_for_fg * fg_src_4d + filled * (1.0 - fg_src_4d)

    # ── Pré-calcul des noyaux (CPU) puis upload unique ────────────────────────
    radii_np = np.linspace(1.0, max_r, max(steps, 4))
    kernels_t = [torch.from_numpy(maker(float(r))).float().to(dev)
                 for r in radii_np]
    layer_sigma = max_r / len(radii_np) * 1.2

    total = 1 + len(radii_np) * 2 + 1
    done  = [0]

    def _progress(label):
        done[0] += 1
        if progress_cb:
            progress_cb(done[0], total, label)

    _progress("Pre-fill")

    # ── Couche BG ─────────────────────────────────────────────────────────────
    bg_accum  = torch.zeros_like(rgb_for_bg)
    bg_weight = torch.zeros((h, w), dtype=torch.float32, device=dev)

    for i, r in enumerate(radii_np):
        gw      = torch.exp(-0.5 * ((blur_t - float(r)) / layer_sigma) ** 2)
        layer_w = gw * bg_pixels
        if float(layer_w.sum().item()) < 1e-3:
            _progress(f"{t('progress_bg', lang)} {i + 1}/{len(radii_np)}")
            continue
        dilated   = _t_dilate(layer_w, float(r)) * bg_allowed
        blurred   = _t_apply_kernel(rgb_for_bg, kernels_t[i])
        bg_accum  = bg_accum + blurred * dilated.unsqueeze(0).unsqueeze(0)
        bg_weight = bg_weight + dilated
        _progress(f"{t('progress_bg', lang)} {i + 1}/{len(radii_np)}")

    bg_w_4d  = bg_weight.unsqueeze(0).unsqueeze(0)
    no_bg    = bg_w_4d < 1e-6
    bg_layer = (bg_accum / bg_w_4d.clamp(min=1e-6)).clamp(0.0, 255.0)
    bg_layer = torch.where(no_bg, rgb_t, bg_layer)

    # libérer la mémoire avant la couche FG
    del bg_accum, bg_weight, bg_w_4d, no_bg, rgb_for_bg

    # ── Couche FG ─────────────────────────────────────────────────────────────
    fg_accum  = torch.zeros_like(rgb_t)
    fg_weight = torch.zeros((h, w), dtype=torch.float32, device=dev)
    fg_alpha  = torch.zeros((h, w), dtype=torch.float32, device=dev)

    for i, r in enumerate(radii_np):
        gw      = torch.exp(-0.5 * ((blur_t - float(r)) / layer_sigma) ** 2)
        layer_w = gw * fg_pixels
        if float(layer_w.sum().item()) < 1e-3:
            _progress(f"{t('progress_fg', lang)} {i + 1}/{len(radii_np)}")
            continue
        dilated   = _t_dilate(layer_w, float(r)) * fg_allowed
        blurred   = _t_apply_kernel(rgb_for_fg, kernels_t[i])
        fg_accum  = fg_accum + blurred * dilated.unsqueeze(0).unsqueeze(0)
        fg_weight = fg_weight + dilated
        fg_alpha  = torch.maximum(fg_alpha, dilated)
        _progress(f"{t('progress_fg', lang)} {i + 1}/{len(radii_np)}")

    fg_w_4d  = fg_weight.unsqueeze(0).unsqueeze(0)
    no_fg    = fg_w_4d < 1e-6
    fg_layer = (fg_accum / fg_w_4d.clamp(min=1e-6)).clamp(0.0, 255.0)
    fg_layer = torch.where(no_fg, rgb_t, fg_layer)

    del fg_accum, fg_weight, fg_w_4d, no_fg, rgb_for_fg

    # ── Couche NETTE ──────────────────────────────────────────────────────────
    sharp_eroded = _t_erode(sharp_mask_raw,
                            min(max(max_r * 0.05, 0.5), 1.5))
    sharp_alpha  = _t_smooth(sharp_eroded,
                             sigma=min(max(max_r * 0.04, 0.5), 1.5)
                             ).clamp(0.0, 1.0)

    # ── Composition back-to-front ─────────────────────────────────────────────
    sp_4d = sharp_protection.unsqueeze(0).unsqueeze(0)
    sa_4d = sharp_alpha.unsqueeze(0).unsqueeze(0)

    result = bg_layer * (1.0 - sp_4d) + rgb_t * sp_4d
    result = result   * (1.0 - sa_4d) + rgb_t * sa_4d

    fg_alpha_s = _t_smooth(fg_alpha, sigma=1.0).clamp(0.0, 1.0)
    fas_4d     = fg_alpha_s.unsqueeze(0).unsqueeze(0)
    result     = result * (1.0 - fas_4d) + fg_layer * fas_4d

    _progress(t("progress_comp", lang))

    # ── Download unique vers le host ──────────────────────────────────────────
    # Conversion uint8 côté CPU pour compatibilité MPS (où to(uint8) sur GPU
    # peut être instable sur certaines versions de torch).
    out = (result.clamp(0.0, 255.0)
           .squeeze(0).permute(1, 2, 0)            # (H, W, 3)
           .contiguous().cpu().numpy().astype(np.uint8))
    return out


# =============================================================================
# Moteur DoF — rendu en couches séparées (v9 : séparation focus-aware)
# =============================================================================

def render_dof(rgb, blur_map, depth, kernel_key="disk", steps=16,
               bleed_correction=True, progress_cb=None, lang="fr",
               focus_near=0.35, focus_far=0.65):
    """
    Rendu DOF en couches séparées  —  v9
    (v12 : dispatch automatique vers _render_dof_torch si backend torch_*)

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
    # ── v12 : chemin tensoriel résident pour les backends torch ───────────────
    if GPU_BACKEND in ("torch_cuda", "torch_mps", "torch_cpu"):
        try:
            return _render_dof_torch(
                rgb, blur_map, depth, kernel_key=kernel_key, steps=steps,
                progress_cb=progress_cb, lang=lang,
                focus_near=focus_near, focus_far=focus_far)
        except Exception as _e:
            # En cas d'OOM ou d'autre erreur GPU, retomber sur le chemin numpy
            import traceback as _tb
            _tb.print_exc()
            print(f"[render_dof] chemin torch a échoué ({_e}), "
                  f"retombée sur NumPy/SciPy.")

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
    _PREFILL_ITERS = 2   # v12 : 2 passes au lieu de 3 (≈ 33% gagnés sur cette étape)

    rgb_for_bg = rgb_f.copy()
    bg_src_3d  = bg_source[:, :, np.newaxis]
    for _ in range(_PREFILL_ITERS):
        filled     = apply_kernel(rgb_for_bg, fill_kernel)
        rgb_for_bg = rgb_for_bg * bg_src_3d + filled * (1.0 - bg_src_3d)

    rgb_for_fg = rgb_f.copy()
    fg_src_3d  = fg_source[:, :, np.newaxis]
    for _ in range(_PREFILL_ITERS):
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
# Éditeur interactif de forme de bokeh
# =============================================================================

class BokehEditorCanvas(tk.Canvas):
    """
    Canvas interactif affichant la forme de bokeh avec des points de contrôle
    draggables. L'utilisateur peut déplacer chaque sommet pour remodeler le
    bokeh librement.
    """
    SZ    = 160   # taille en pixels
    PAD   = 14    # marge intérieure
    PT_R  = 5     # rayon des poignées

    # Points initiaux par forme prédéfinie (normalisés dans [-1, 1]²)
    @staticmethod
    def _circle(n=16):
        return [(math.cos(2*math.pi*i/n - math.pi/2),
                 math.sin(2*math.pi*i/n - math.pi/2)) for i in range(n)]

    @staticmethod
    def _polygon(n):
        return [(math.cos(2*math.pi*i/n - math.pi/2),
                 math.sin(2*math.pi*i/n - math.pi/2)) for i in range(n)]

    @staticmethod
    def _star(n=6, inner=0.42):
        pts = []
        for i in range(n * 2):
            a = 2*math.pi*i/(n*2) - math.pi/2
            r = 1.0 if i % 2 == 0 else inner
            pts.append((r*math.cos(a), r*math.sin(a)))
        return pts

    @staticmethod
    def _ellipse(aspect=0.32, n=16):
        return [(math.cos(2*math.pi*i/n - math.pi/2),
                 aspect*math.sin(2*math.pi*i/n - math.pi/2)) for i in range(n)]

    @staticmethod
    def _heart(n=20):
        pts = []
        for i in range(n):
            t = 2*math.pi*i/n - math.pi/2
            x =  16*math.sin(t)**3 / 18
            y = -(13*math.cos(t) - 5*math.cos(2*t)
                  - 2*math.cos(3*t) - math.cos(4*t)) / 18
            pts.append((x, y))
        return pts

    @staticmethod
    def _ring(n=16):
        """Pour l'anneau : 2 anneaux (outer + inner) concaténés."""
        outer = [(math.cos(2*math.pi*i/n - math.pi/2),
                  math.sin(2*math.pi*i/n - math.pi/2)) for i in range(n)]
        return outer  # on dessine juste l'outer pour l'éditeur

    PRESETS = {
        "disk":       lambda s: s._circle(16),
        "gauss":      lambda s: s._circle(16),
        "hex":        lambda s: s._polygon(6),
        "pentagon":   lambda s: s._polygon(5),
        "octagon":    lambda s: s._polygon(8),
        "ring":       lambda s: s._ring(16),
        "anamorphic": lambda s: s._ellipse(0.32, 16),
        "star":       lambda s: s._star(6, 0.42),
        "heart":      lambda s: s._heart(20),
        "custom":     lambda s: s._pts,  # garder les pts courants
    }

    def __init__(self, master, on_change, **kw):
        bg = kw.pop("bg", "#0d0d1a")
        super().__init__(master, width=self.SZ, height=self.SZ,
                         bg=bg, highlightthickness=1,
                         highlightbackground="#2a2a4a", cursor="crosshair", **kw)
        self._pts       = self._circle(16)
        self._drag_idx  = None
        self._on_change = on_change
        self.bind("<ButtonPress-1>",   self._on_press)
        self.bind("<B1-Motion>",       self._on_drag_pt)
        self.bind("<ButtonRelease-1>", self._on_release_pt)
        self.bind("<Configure>",       lambda _: self.redraw())
        self.redraw()

    # ── API publique ──────────────────────────────────────────────────────────

    def load_preset(self, key):
        fn = self.PRESETS.get(key)
        if fn is not None and key != "custom":
            self._pts = fn(self)
        self.redraw()

    def get_points(self):
        return list(self._pts)

    def set_points(self, pts):
        self._pts = list(pts)
        self.redraw()

    # ── Coordonnées ──────────────────────────────────────────────────────────

    def _to_canvas(self, nx, ny):
        r  = (self.SZ - 2*self.PAD) / 2
        cx = cy = self.SZ / 2
        return cx + nx * r, cy + ny * r

    def _from_canvas(self, cx, cy):
        r  = (self.SZ - 2*self.PAD) / 2
        c  = self.SZ / 2
        nx = (cx - c) / r
        ny = (cy - c) / r
        # Clamp dans le disque unité
        mag = math.sqrt(nx*nx + ny*ny)
        if mag > 1.0:
            nx /= mag; ny /= mag
        return float(nx), float(ny)

    # ── Dessin ────────────────────────────────────────────────────────────────

    def redraw(self):
        self.delete("all")
        sz, pad = self.SZ, self.PAD
        r_ref = (sz - 2*pad) / 2
        cx = cy = sz / 2

        # Cercle de référence et réticule
        self.create_oval(cx-r_ref, cy-r_ref, cx+r_ref, cy+r_ref,
                         outline="#1e1e40", width=1)
        self.create_line(cx, pad, cx, sz-pad, fill="#1a1a3a", width=1)
        self.create_line(pad, cy, sz-pad, cy, fill="#1a1a3a", width=1)
        # Cercles de grille 50%
        rh = r_ref * 0.5
        self.create_oval(cx-rh, cy-rh, cx+rh, cy+rh, outline="#161630", width=1)

        if len(self._pts) < 2:
            return

        # Polygone rempli (forme du bokeh)
        canvas_pts = [self._to_canvas(x, y) for x, y in self._pts]
        flat = [c for pt in canvas_pts for c in pt]
        self.create_polygon(flat, fill="#1e2a5e", outline="#5070e0", width=1.5,
                            smooth=True)

        # Lignes de connexion vers le centre (rayons)
        for px, py in canvas_pts:
            self.create_line(cx, cy, px, py, fill="#2a3080", width=1,
                             dash=(2, 3))

        # Points de contrôle
        pr = self.PT_R
        for i, (px, py) in enumerate(canvas_pts):
            col = "#60d0ff" if i % 2 == 0 else "#a0c4ff"
            self.create_oval(px-pr, py-pr, px+pr, py+pr,
                             fill=col, outline="#ffffff", width=1)

    # ── Interaction ──────────────────────────────────────────────────────────

    def _nearest_pt(self, ex, ey, threshold=12):
        best_i, best_d = None, threshold
        for i, (nx, ny) in enumerate(self._pts):
            px, py = self._to_canvas(nx, ny)
            d = math.sqrt((ex-px)**2 + (ey-py)**2)
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    def _on_press(self, e):
        self._drag_idx = self._nearest_pt(e.x, e.y)

    def _on_drag_pt(self, e):
        if self._drag_idx is None:
            return
        nx, ny = self._from_canvas(e.x, e.y)
        pts = list(self._pts)
        pts[self._drag_idx] = (nx, ny)
        self._pts = pts
        self.redraw()
        if self._on_change:
            self._on_change()

    def _on_release_pt(self, e):
        self._drag_idx = None


# =============================================================================
# Noms des langues dans leur propre langue
# =============================================================================

LANG_NAMES = {
    "fr": "Français",
    "en": "English",
    "nl": "Nederlands",
    "de": "Deutsch",
    "zh": "中文",
    "ja": "日本語",
    "ar": "العربية",
    "kl": "tlhIngan Hol",
    "wa": "Walon",
    "ru": "Русский",
    "hi": "हिन्दी",
    "es": "Español",
    "pt": "Português",
    "bn": "বাংলা",
}


# =============================================================================
# Widget sélecteur de langue (drapeau unique + popup)
# =============================================================================

class LangSelector(tk.Frame):
    """
    Affiche un seul drapeau (langue active), toujours bordé en bleu.
    Au clic → popup liste de tous les drapeaux + nom dans leur propre langue.
    Le nom de la langue n'apparaît QUE dans le popup.
    """

    def __init__(self, master, current_lang, switch_cb, **kw):
        BG = kw.pop("bg", "#1a1a2e")
        super().__init__(master, bg=BG, **kw)
        self._switch_cb   = switch_cb
        self._current     = current_lang
        self._popup       = None
        self._flag_widget = None
        self._build(BG)

    def _build(self, BG):
        for w in self.winfo_children():
            w.destroy()
        fb = FlagButton(self, self._current,
                        command=lambda _: self._open_popup(), bg=BG)
        # Aucune bordure sur le drapeau d'en-tête
        fb.config(highlightthickness=0)
        fb.pack(side=tk.LEFT)
        self._flag_widget = fb

    def set_lang(self, code):
        self._current = code
        self._build("#1a1a2e")

    def _open_popup(self):
        if self._popup and self._popup.winfo_exists():
            self._popup.destroy()
            self._popup = None
            return

        BG  = "#0d0d1a"
        BDR = "#2a2a6a"
        pop = tk.Toplevel(self)
        pop.overrideredirect(True)
        pop.configure(bg=BG)
        pop.attributes("-topmost", True)
        self._popup = pop

        # Cadre avec bordure
        outer = tk.Frame(pop, bg=BDR, bd=1)
        outer.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        inner = tk.Frame(outer, bg=BG)
        inner.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        for lc in LANGUAGES:
            row = tk.Frame(inner, bg=BG, cursor="hand2")
            row.pack(fill=tk.X, padx=6, pady=2)
            fb  = FlagButton(row, lc, command=lambda c=lc: self._select(c), bg=BG)
            fb.set_selected(lc == self._current)
            fb.pack(side=tk.LEFT, padx=(0, 8))
            name = LANG_NAMES.get(lc, lc)
            lbl  = tk.Label(row, text=name, bg=BG, fg="#e0e0f0",
                            font=("Courier New", 10), anchor="w", cursor="hand2")
            lbl.pack(side=tk.LEFT, fill=tk.X)
            # Clic sur toute la ligne
            for w in (row, lbl):
                w.bind("<Button-1>", lambda _, c=lc: self._select(c))

        # Positionnement sous le widget
        self.update_idletasks()
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.winfo_height() + 2
        pop.geometry(f"+{x}+{y}")

        # Fermer si clic hors popup
        pop.bind("<FocusOut>", lambda _: self._close_popup())
        pop.focus_set()

    def _select(self, code):
        self._close_popup()
        self._switch_cb(code)

    def _close_popup(self):
        if self._popup and self._popup.winfo_exists():
            self._popup.destroy()
        self._popup = None


# =============================================================================
# Widget drapeaux
# =============================================================================

class FlagButton(tk.Canvas):
    W, H = 26, 17

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
# v12 — Éditeur de courbe Z-depth (style Photoshop / Lightroom)
# =============================================================================

class DepthCurveCanvas(tk.Canvas):
    """
    Éditeur de courbe Bézier cubique style Inkscape (v12).

    Structure : liste d'« ancres », chacune étant un 6-tuple
        (x, y, hx_in, hy_in, hx_out, hy_out)
    où (x, y) est la position de l'ancre dans [0, 1]² et
    (hx_*, hy_*) sont les DÉCALAGES ABSOLUS des deux poignées
    relativement à l'ancre (la poignée IN pointe vers le segment
    précédent, la poignée OUT vers le segment suivant).

    Entre deux ancres consécutives A et B, le segment est une
    Bézier cubique de points de contrôle :
        P0 = A.xy
        P1 = A.xy + A.out
        P2 = B.xy + B.in
        P3 = B.xy

    Interaction :
      • Glisser une ancre   → la déplacer (ses poignées suivent)
      • Glisser une poignée → ajuster la courbure
          (par défaut : l'autre poignée reste alignée, comme un nœud
           « smooth » Inkscape · tenir Alt pour rompre la symétrie)
      • Double-clic ailleurs → ajoute une ancre auto-lissée
      • Clic droit sur une ancre → la supprime
      • Les ancres d'extrémité (x=0 et x=1) sont bloquées en X.

    Le slider « qualité » externe contrôle le nombre total de points
    d'échantillonnage utilisés pour construire la LUT appliquée à la
    depth map et pour tracer la courbe à l'écran (16 à 512).
    """

    PAD = 14
    DEFAULT_QUALITY = 128

    # ── Presets (v13) ────────────────────────────────────────────────────────
    # Chaque preset est décrit par 2 ancres aux coins (0,0) et (1,1) avec
    # des poignées Bézier explicites — un seul segment cubique. Format
    # complet (6-tuples) : (x, y, hx_in, hy_in, hx_out, hy_out).
    @staticmethod
    def _two_pt(p1, p2):
        """Construit deux ancres aux coins avec p1/p2 comme points de contrôle.
        p1 est l'absolu du 1er point de contrôle (poignée sortante de (0,0)),
        p2 est l'absolu du 2nd point de contrôle (poignée entrante de (1,1))."""
        return [
            (0.0, 0.0, 0.0,        0.0,        p1[0],       p1[1]),
            (1.0, 1.0, p2[0] - 1,  p2[1] - 1,  0.0,         0.0),
        ]

    PRESETS = {
        "linear":         _two_pt.__func__((1/3, 1/3), (2/3, 2/3)),

        "concave_soft":   _two_pt.__func__((0.10, 0.40), (0.50, 0.95)),
        "concave":        _two_pt.__func__((0.05, 0.55), (0.40, 1.00)),
        "concave_strong": _two_pt.__func__((0.02, 0.75), (0.25, 1.00)),

        "convex_soft":    _two_pt.__func__((0.50, 0.05), (0.90, 0.60)),
        "convex":         _two_pt.__func__((0.60, 0.00), (0.95, 0.45)),
        "convex_strong":  _two_pt.__func__((0.75, 0.00), (0.98, 0.25)),

        "s_curve":        _two_pt.__func__((0.40, 0.10), (0.60, 0.90)),
        "s_curve_strong": _two_pt.__func__((0.45, 0.02), (0.55, 0.98)),

        "inverse_s":      _two_pt.__func__((0.10, 0.40), (0.90, 0.60)),
    }

    def __init__(self, master, on_change, width=240, height=200, **kw):
        super().__init__(master,
                         width=width, height=height,
                         bg="#0d0d1a",
                         highlightthickness=1,
                         highlightbackground="#2a2a6a",
                         **kw)
        self.on_change = on_change
        self._W       = width
        self._H       = height
        self._quality = self.DEFAULT_QUALITY

        # Ancres : chaque point est (x, y, hx_in, hy_in, hx_out, hy_out)
        self.anchors = []
        self._set_from_anchor_list([(0.0, 0.0), (1.0, 1.0)])

        # État d'interaction
        self._drag_kind = None   # "anchor", "handle_in", "handle_out"
        self._drag_idx  = None
        self._alt_held  = False

        self.bind("<Button-1>",        self._on_press)
        self.bind("<B1-Motion>",       self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Double-Button-1>", self._on_dclick)
        self.bind("<Button-3>",        self._on_rclick)
        self.bind("<Configure>",       self._on_configure)
        self.bind("<KeyPress-Alt_L>",  lambda e: setattr(self, "_alt_held", True))
        self.bind("<KeyRelease-Alt_L>",lambda e: setattr(self, "_alt_held", False))
        self.redraw()

    # ── API publique ─────────────────────────────────────────────────────────

    def set_quality(self, q: int):
        self._quality = max(16, min(512, int(q)))
        self.redraw()
        if self.on_change:
            self.on_change()

    def get_quality(self) -> int:
        return self._quality

    def reset(self):
        self.load_preset("linear")

    def load_preset(self, key: str):
        pts = self.PRESETS.get(key)
        if pts is None:
            return
        self._set_from_anchor_list(pts)
        self.redraw()
        if self.on_change:
            self.on_change()

    def matches_preset(self, key: str, tol: float = 1e-3) -> bool:
        """True si la courbe actuelle correspond exactement au preset `key`."""
        pts = self.PRESETS.get(key)
        if pts is None or len(pts) != len(self.anchors):
            return False
        expected = self._expand_preset(pts)
        for a, b in zip(expected, self.anchors):
            for k in range(6):
                if abs(a[k] - b[k]) > tol:
                    return False
        return True

    @classmethod
    def _expand_preset(cls, pts):
        """Accepte une liste de 2-tuples (auto-lissage) ou de 6-tuples
        (poignées explicites). Retourne toujours une liste de 6-tuples."""
        if not pts:
            return []
        if len(pts[0]) == 6:
            return [tuple(float(v) for v in p) for p in pts]
        return cls._auto_smooth_from_anchors(pts)

    def is_identity(self) -> bool:
        return self.matches_preset("linear")

    def apply(self, depth: np.ndarray) -> np.ndarray:
        """Applique la courbe Bézier échantillonnée à une depth map."""
        if self.is_identity():
            return depth
        xs, ys = self._build_lut()
        d = np.clip(depth, 0.0, 1.0)
        return np.interp(d, xs, ys).astype(np.float32)

    def get_anchors(self):
        """Retourne une copie de la liste d'ancres (6-tuples)."""
        return list(self.anchors)

    # ── Auto-lissage (Catmull-Rom-like) ──────────────────────────────────────

    @classmethod
    def _auto_smooth_from_anchors(cls, anchor_xy_list):
        """
        Retourne une liste de 6-tuples avec poignées calculées pour que
        la courbe Bézier passe par tous les points de manière naturelle,
        à la manière d'une spline Catmull-Rom convertie en Bézier.
        Pour une liste de 2 points, donne une Bézier strictement linéaire.
        """
        n   = len(anchor_xy_list)
        out = []
        for i in range(n):
            x, y = anchor_xy_list[i]
            if i == 0:
                nx, ny = anchor_xy_list[1]
                dx = (nx - x) / 3.0
                dy = (ny - y) / 3.0
                out.append((x, y, 0.0, 0.0, dx, dy))
            elif i == n - 1:
                px, py = anchor_xy_list[i - 1]
                dx = (x - px) / 3.0
                dy = (y - py) / 3.0
                out.append((x, y, -dx, -dy, 0.0, 0.0))
            else:
                px, py = anchor_xy_list[i - 1]
                nx, ny = anchor_xy_list[i + 1]
                tdx = nx - px
                tdy = ny - py
                tlen = math.hypot(tdx, tdy)
                if tlen < 1e-9:
                    out.append((x, y, 0.0, 0.0, 0.0, 0.0))
                    continue
                ux, uy = tdx / tlen, tdy / tlen
                len_prev = math.hypot(x - px, y - py)
                len_next = math.hypot(nx - x, ny - y)
                out.append((x, y,
                            -ux * len_prev / 3.0, -uy * len_prev / 3.0,
                             ux * len_next / 3.0,  uy * len_next / 3.0))
        return out

    def _set_from_anchor_list(self, anchor_list):
        """Accepte des 2-tuples (auto-lissage) ou des 6-tuples (poignées
        explicites — utilisé par les nouveaux presets v13)."""
        self.anchors = list(self._expand_preset(anchor_list))

    # ── Échantillonnage Bézier ───────────────────────────────────────────────

    @staticmethod
    def _sample_bezier(p0, p1, p2, p3, n):
        """Échantillonne une Bézier cubique en n points équirépartis en t."""
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)
        mt = 1.0 - t
        bx = (mt**3 * p0[0] + 3 * mt**2 * t * p1[0]
              + 3 * mt * t**2 * p2[0] + t**3 * p3[0])
        by = (mt**3 * p0[1] + 3 * mt**2 * t * p1[1]
              + 3 * mt * t**2 * p2[1] + t**3 * p3[1])
        return bx, by

    def _build_lut(self):
        """Construit la LUT (xs, ys) à partir des segments Bézier actuels."""
        n_seg = len(self.anchors) - 1
        if n_seg < 1:
            return (np.array([0.0, 1.0], dtype=np.float32),
                    np.array([0.0, 1.0], dtype=np.float32))
        per_seg = max(self._quality // n_seg, 6)
        xs_parts, ys_parts = [], []
        for i in range(n_seg):
            a = self.anchors[i]
            b = self.anchors[i + 1]
            p0 = (a[0],         a[1])
            p1 = (a[0] + a[4],  a[1] + a[5])
            p2 = (b[0] + b[2],  b[1] + b[3])
            p3 = (b[0],         b[1])
            bx, by = self._sample_bezier(p0, p1, p2, p3, per_seg)
            xs_parts.append(bx)
            ys_parts.append(by)
        xs = np.concatenate(xs_parts)
        ys = np.concatenate(ys_parts)
        # Enforce x-monotonie (corrige de petits retours arrière dus à des
        # poignées mal positionnées), sinon np.interp donne des résultats
        # indéfinis.
        xs = np.maximum.accumulate(xs)
        xs = np.clip(xs, 0.0, 1.0)
        ys = np.clip(ys, 0.0, 1.0)
        return xs, ys

    # ── Coordonnées canvas ↔ normalisées ─────────────────────────────────────

    def _to_canvas(self, nx, ny):
        m = self.PAD
        cx = m + nx * (self._W - 2*m)
        cy = m + (1.0 - ny) * (self._H - 2*m)
        return cx, cy

    def _from_canvas(self, cx, cy):
        m = self.PAD
        nx = (cx - m) / max(self._W - 2*m, 1)
        ny = 1.0 - (cy - m) / max(self._H - 2*m, 1)
        return max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))

    # ── Rendu ────────────────────────────────────────────────────────────────

    def redraw(self):
        self.delete("all")
        m   = self.PAD
        W, H = self._W, self._H

        # Grille 4×4
        for i in range(1, 4):
            x = m + i * (W - 2*m) / 4
            y = m + i * (H - 2*m) / 4
            self.create_line(x, m, x, H - m, fill="#1e1e3e")
            self.create_line(m, y, W - m, y, fill="#1e1e3e")

        self.create_rectangle(m, m, W - m, H - m,
                              outline="#2a2a6a", width=1)

        # Diagonale de référence (identité)
        self.create_line(m, H - m, W - m, m, fill="#333355", dash=(2, 2))

        # ── Courbe Bézier échantillonnée ─────────────────────────────────────
        xs, ys = self._build_lut()
        coords = []
        for x, y in zip(xs, ys):
            cx, cy = self._to_canvas(float(x), float(y))
            coords.extend((cx, cy))
        if len(coords) >= 4:
            self.create_line(*coords, fill="#ffcc66", width=2, smooth=False)

        # ── Poignées (dashed lines + petits cercles) ─────────────────────────
        n = len(self.anchors)
        for i, a in enumerate(self.anchors):
            ax, ay, hin_x, hin_y, hout_x, hout_y = a
            acx, acy = self._to_canvas(ax, ay)

            # Poignée IN (sauf 1ère ancre)
            if i > 0:
                hx = max(0.0, min(1.0, ax + hin_x))
                hy = max(0.0, min(1.0, ay + hin_y))
                hcx, hcy = self._to_canvas(hx, hy)
                self.create_line(acx, acy, hcx, hcy,
                                 fill="#7a7aaa", dash=(2, 2))
                self.create_oval(hcx-3, hcy-3, hcx+3, hcy+3,
                                 fill="#4a9eff", outline="#ffffff")

            # Poignée OUT (sauf dernière ancre)
            if i < n - 1:
                hx = max(0.0, min(1.0, ax + hout_x))
                hy = max(0.0, min(1.0, ay + hout_y))
                hcx, hcy = self._to_canvas(hx, hy)
                self.create_line(acx, acy, hcx, hcy,
                                 fill="#7a7aaa", dash=(2, 2))
                self.create_oval(hcx-3, hcy-3, hcx+3, hcy+3,
                                 fill="#4a9eff", outline="#ffffff")

        # ── Ancres (carrés pour les distinguer des poignées) ─────────────────
        for i, a in enumerate(self.anchors):
            ax, ay = a[0], a[1]
            acx, acy = self._to_canvas(ax, ay)
            is_end = (i == 0 or i == n - 1)
            fill = "#ff9f4a" if is_end else "#ffcc66"
            self.create_rectangle(acx-5, acy-5, acx+5, acy+5,
                                  fill=fill, outline="#ffffff", width=1)

        # Graduations
        self.create_text(m + 2,      H - m + 2, anchor="nw", text="0",
                         fill="#666688", font=("Courier New", 7))
        self.create_text(W - m - 2,  H - m + 2, anchor="ne", text="1",
                         fill="#666688", font=("Courier New", 7))
        self.create_text(m - 2,      m,         anchor="ne", text="1",
                         fill="#666688", font=("Courier New", 7))

    # ── Hit-testing ──────────────────────────────────────────────────────────

    def _hit_test(self, cx, cy, threshold=9):
        """Retourne ('anchor'|'handle_in'|'handle_out', index) ou (None, None)."""
        th2 = threshold * threshold
        best_kind, best_idx, best_d = None, None, th2
        n = len(self.anchors)
        # Priorité aux poignées (plus petites, dessinées au-dessus)
        for i, a in enumerate(self.anchors):
            ax, ay, hin_x, hin_y, hout_x, hout_y = a
            if i > 0:
                hcx, hcy = self._to_canvas(ax + hin_x, ay + hin_y)
                d = (hcx - cx) ** 2 + (hcy - cy) ** 2
                if d < best_d:
                    best_d, best_kind, best_idx = d, "handle_in", i
            if i < n - 1:
                hcx, hcy = self._to_canvas(ax + hout_x, ay + hout_y)
                d = (hcx - cx) ** 2 + (hcy - cy) ** 2
                if d < best_d:
                    best_d, best_kind, best_idx = d, "handle_out", i
        # Si rien trouvé sur une poignée, chercher une ancre
        if best_kind is None:
            for i, a in enumerate(self.anchors):
                acx, acy = self._to_canvas(a[0], a[1])
                d = (acx - cx) ** 2 + (acy - cy) ** 2
                if d < best_d:
                    best_d, best_kind, best_idx = d, "anchor", i
        return best_kind, best_idx

    # ── Événements souris ────────────────────────────────────────────────────

    def _on_configure(self, e):
        self._W = e.width
        self._H = e.height
        self.redraw()

    def _on_press(self, e):
        self.focus_set()   # pour recevoir les événements Alt
        self._alt_held = bool(getattr(e, "state", 0) & 0x20000)  # Alt mask
        kind, idx = self._hit_test(e.x, e.y)
        self._drag_kind = kind
        self._drag_idx  = idx

    def _on_drag(self, e):
        if self._drag_kind is None or self._drag_idx is None:
            return
        nx, ny = self._from_canvas(e.x, e.y)
        idx    = self._drag_idx

        if self._drag_kind == "anchor":
            a = list(self.anchors[idx])
            # Contraintes X pour les ancres d'extrémité et les voisinages
            if idx == 0:
                nx = 0.0
            elif idx == len(self.anchors) - 1:
                nx = 1.0
            else:
                left  = self.anchors[idx - 1][0]
                right = self.anchors[idx + 1][0]
                nx = max(left + 0.01, min(right - 0.01, nx))
            a[0], a[1] = nx, ny
            self.anchors[idx] = tuple(a)

        elif self._drag_kind == "handle_out":
            a = list(self.anchors[idx])
            # Décalage absolu
            hx = nx - a[0]
            hy = ny - a[1]
            # Contrainte : poignée OUT ne doit pas dépasser l'ancre suivante
            if idx < len(self.anchors) - 1:
                max_hx = self.anchors[idx + 1][0] - a[0] - 0.005
                hx = max(0.0, min(max_hx, hx))
            a[4], a[5] = hx, hy
            # Par défaut : l'autre poignée reste alignée (nœud smooth)
            if not self._alt_held and idx > 0:
                # longueur de l'ancienne poignée IN conservée
                old_in_len = math.hypot(a[2], a[3])
                out_len    = math.hypot(hx, hy)
                if out_len > 1e-6:
                    # IN pointe à l'opposé
                    a[2] = -hx / out_len * old_in_len
                    a[3] = -hy / out_len * old_in_len
                    # Clipper IN pour ne pas dépasser l'ancre précédente
                    min_hx_in = self.anchors[idx - 1][0] - a[0] + 0.005
                    if a[2] < min_hx_in:
                        a[2] = min_hx_in
            self.anchors[idx] = tuple(a)

        elif self._drag_kind == "handle_in":
            a = list(self.anchors[idx])
            hx = nx - a[0]
            hy = ny - a[1]
            if idx > 0:
                min_hx = self.anchors[idx - 1][0] - a[0] + 0.005
                hx = min(0.0, max(min_hx, hx))
            a[2], a[3] = hx, hy
            if not self._alt_held and idx < len(self.anchors) - 1:
                old_out_len = math.hypot(a[4], a[5])
                in_len      = math.hypot(hx, hy)
                if in_len > 1e-6:
                    a[4] = -hx / in_len * old_out_len
                    a[5] = -hy / in_len * old_out_len
                    max_hx_out = self.anchors[idx + 1][0] - a[0] - 0.005
                    if a[4] > max_hx_out:
                        a[4] = max_hx_out
            self.anchors[idx] = tuple(a)

        self.redraw()

    def _on_release(self, e):
        if self._drag_kind is not None:
            self._drag_kind = None
            self._drag_idx  = None
            if self.on_change:
                self.on_change()

    def _on_dclick(self, e):
        # Ignorer si déjà sur une ancre ou poignée existante
        kind, _ = self._hit_test(e.x, e.y, threshold=10)
        if kind is not None:
            return
        nx, ny = self._from_canvas(e.x, e.y)
        if nx <= 0.01 or nx >= 0.99:
            return
        # Trouver la position d'insertion et ajouter l'ancre en conservant
        # les autres. Ensuite ré-auto-lisser pour garder une courbe propre.
        xy_list = [(a[0], a[1]) for a in self.anchors]
        xy_list.append((nx, ny))
        xy_list.sort(key=lambda p: p[0])
        self._set_from_anchor_list(xy_list)
        self.redraw()
        if self.on_change:
            self.on_change()

    def _on_rclick(self, e):
        kind, idx = self._hit_test(e.x, e.y)
        if kind != "anchor":
            return
        if idx == 0 or idx == len(self.anchors) - 1:
            return
        xy_list = [(a[0], a[1]) for i, a in enumerate(self.anchors)
                   if i != idx]
        self._set_from_anchor_list(xy_list)
        self.redraw()
        if self.on_change:
            self.on_change()


# Ordre d'affichage des presets dans le combobox, avec leur clé de traduction.
CURVE_PRESET_KEYS = [
    "linear",
    "concave_soft",
    "concave",
    "concave_strong",
    "convex_soft",
    "convex",
    "convex_strong",
    "s_curve",
    "s_curve_strong",
    "inverse_s",
]
CURVE_PRESET_T = [
    "curve_linear",
    "curve_concave_soft",
    "curve_concave",
    "curve_concave_strong",
    "curve_convex_soft",
    "curve_convex",
    "curve_convex_strong",
    "curve_s",
    "curve_s_strong",
    "curve_inverse_s",
]


# =============================================================================
# Application principale
# =============================================================================

class DofApp(_DND_BASE):

    PANEL_MIN = 260
    PANEL_MAX = 600
    PANEL_DEF = 320

    # Taille max (en pixels) du plus grand côté de l'aperçu calculé
    # en mode "preview" (sliders). Valeur par défaut, écrasée dans __init__
    # selon la résolution écran : ~560 sur 1080p, ~720 sur 1440p, ~960 sur 4K.
    PREVIEW_MAX_DIM = 560

    def __init__(self):
        super().__init__()
        self.title("Depth of Field Tool  —  HQ  v12")
        self.configure(bg="#1a1a2e")
        self.resizable(True, True)

        # Adapter la taille de l'aperçu à la résolution écran disponible.
        try:
            sh = self.winfo_screenheight()
            self.PREVIEW_MAX_DIM = max(560, min(1200, sh // 3))
        except Exception:
            pass  # garder la valeur de classe par défaut

        self.lang        = "fr"
        self.rgb_img     = None
        self.depth_img   = None
        self.result_img  = None
        self._depth_path = None
        self._render_job = None
        self._computing  = False
        self._kernel_key = "disk"
        self._bokeh_pts  = list(_CUSTOM_BOKEH_PTS)  # points éditeur bokeh

        # Sujet pour le flou de bord radial
        self._subject_xy      = None
        self._picking_subject = False
        self._display_geom    = None    # (x0, y0, nw, nh) de l'image dans le canvas

        # Modes pick near / far
        self._picking_near = False
        self._picking_far  = False

        # Vue comparative (split slider)
        self._split_pos  = 0.5          # position normalisée [0,1]
        self._split_drag = False

        # Largeur courante du panneau gauche
        self._panel_width = self.PANEL_DEF

        # État visuel drag-and-drop
        self._dnd_hover = False

        self._build_ui()
        if HAS_DND:
            self._setup_dnd()

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

        # ── Titre + sélecteur de langue sur la même ligne ────────────────────
        title_row = tk.Frame(ctrl, bg=BG)
        title_row.pack(fill=tk.X, pady=(0, 8))

        self._lang_selector = LangSelector(title_row, self.lang,
                                           switch_cb=self._switch_lang,
                                           bg=BG)
        self._lang_selector.pack(side=tk.LEFT, padx=(0, 8))

        self.lbl_title = ttk.Label(title_row, font=("Courier New", 11, "bold"),
                                   foreground="#a0c4ff")
        self.lbl_title.pack(side=tk.LEFT)

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

        # ── Courbe Z-depth (v12) ──────────────────────────────────────────────
        # Éditeur vectoriel de courbe tonale pour la depth map.
        _curve_header = ttk.Frame(ctrl)
        _curve_header.pack(fill=tk.X, pady=(8, 2))
        self.lbl_depth_curve_k = ttk.Label(_curve_header, foreground="#ffcc66")
        self.lbl_depth_curve_k.pack(side=tk.LEFT)
        self.btn_reset_curve = ttk.Button(
            _curve_header, width=8, command=self._reset_depth_curve)
        self.btn_reset_curve.pack(side=tk.RIGHT)

        # Combobox de présélections
        _preset_row = ttk.Frame(ctrl)
        _preset_row.pack(fill=tk.X, pady=(0, 4))
        self.lbl_curve_preset_k = ttk.Label(_preset_row, foreground="#888899",
                                             font=("Courier New", 9))
        self.lbl_curve_preset_k.pack(side=tk.LEFT, padx=(0, 4))
        self.var_curve_preset = tk.StringVar(value="")
        self.cb_curve_preset = ttk.Combobox(
            _preset_row, textvariable=self.var_curve_preset,
            state="readonly", font=("Courier New", 9))
        self.cb_curve_preset.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.cb_curve_preset.bind("<<ComboboxSelected>>",
                                  self._on_curve_preset_select)

        self.depth_curve_canvas = DepthCurveCanvas(
            ctrl,
            on_change=self._on_depth_curve_changed,
            width=240, height=200)
        self.depth_curve_canvas.pack(fill=tk.X, pady=(0, 2))

        # ── v13 : Aperçu de la Z-depth modifiée par la courbe ────────────────
        self.lbl_depth_preview = tk.Label(ctrl, bg="#0d0d1a",
                                          bd=1, relief="flat")
        self.lbl_depth_preview.pack(fill=tk.X, pady=(0, 4))
        self._depth_preview_imgtk = None  # garder une référence forte

        self.lbl_curve_hint = ttk.Label(ctrl, foreground="#666688",
                                         font=("Courier New", 8),
                                         wraplength=220, justify="left")
        self.lbl_curve_hint.pack(anchor="w", pady=(0, 4))

        # Slider de qualité (nombre de points d'échantillonnage Bézier)
        _quality_row = ttk.Frame(ctrl)
        _quality_row.pack(fill=tk.X, pady=(0, 6))
        self.lbl_curve_quality_k = ttk.Label(_quality_row, foreground="#888899",
                                              font=("Courier New", 9))
        self.lbl_curve_quality_k.pack(side=tk.LEFT, padx=(0, 4))
        self.var_curve_quality = tk.IntVar(
            value=DepthCurveCanvas.DEFAULT_QUALITY)
        ttk.Scale(_quality_row, from_=16, to=512, orient=tk.HORIZONTAL,
                  variable=self.var_curve_quality,
                  command=lambda _: self._on_curve_quality_changed()).pack(
                      side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_curve_quality_v = ttk.Label(_quality_row, foreground="#ffcc66",
                                              width=5,
                                              font=("Courier New", 9))
        self.lbl_curve_quality_v.pack(side=tk.LEFT)
        self.lbl_curve_quality_v.config(text=f"{DepthCurveCanvas.DEFAULT_QUALITY}")

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
            """Slider simple (pour max_blur, steps, etc.)"""
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

        def _precise_slider_row(lbl_attr, var, from_, to_, color, decimals=3):
            """Slider au millième + champ de saisie manuelle."""
            lbl = ttk.Label(ctrl)
            setattr(self, lbl_attr + "_k", lbl)
            lbl.pack(anchor="w")
            row = ttk.Frame(ctrl)
            row.pack(fill=tk.X, pady=(0, 2))

            scale = ttk.Scale(row, from_=from_, to=to_, orient=tk.HORIZONTAL,
                              variable=var)
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

            fmt = f"{{:.{decimals}f}}"
            sv  = tk.StringVar(value=fmt.format(var.get()))

            entry = tk.Entry(row, textvariable=sv, width=7,
                             bg="#16213e", fg=color,
                             insertbackground=color,
                             relief="flat", font=("Courier New", 10),
                             justify="right")
            entry.pack(side=tk.LEFT, padx=(4, 0))
            setattr(self, lbl_attr + "_entry", entry)
            setattr(self, lbl_attr + "_sv",    sv)

            # Slider → Entry
            def _on_scale_move(_event=None):
                sv.set(fmt.format(var.get()))
                self._on_slider()
            scale.config(command=_on_scale_move)

            # Entry → Slider (sur Entrée ou perte de focus)
            def _commit_entry(_event=None):
                try:
                    v = float(sv.get().replace(",", "."))
                    v = max(from_, min(to_, v))
                    var.set(round(v, decimals))
                    sv.set(fmt.format(v))
                    self._on_slider()
                except ValueError:
                    sv.set(fmt.format(var.get()))
            entry.bind("<Return>",   _commit_entry)
            entry.bind("<FocusOut>", _commit_entry)
            # Stocker le callback pour mise à jour externe (pick depuis image)
            setattr(self, lbl_attr + "_commit", _commit_entry)
            setattr(self, lbl_attr + "_v", entry)   # compatibilité _update_labels

        _precise_slider_row("lbl_near",    self.var_near,         0.0,  1.0,  "#60d394")
        # Bouton pick limite proche
        pick_near_row = ttk.Frame(ctrl)
        pick_near_row.pack(fill=tk.X, pady=(0, 6))
        self.btn_pick_near = ttk.Button(pick_near_row, command=self._toggle_pick_near)
        self.btn_pick_near.pack(side=tk.LEFT, fill=tk.X, expand=True)

        _precise_slider_row("lbl_far",     self.var_far,          0.0,  1.0,  "#ff9f6b")
        # Bouton pick limite lointaine
        pick_far_row = ttk.Frame(ctrl)
        pick_far_row.pack(fill=tk.X, pady=(0, 6))
        self.btn_pick_far = ttk.Button(pick_far_row, command=self._toggle_pick_far)
        self.btn_pick_far.pack(side=tk.LEFT, fill=tk.X, expand=True)

        _precise_slider_row("lbl_fo_near", self.var_falloff_near, 0.0,  0.25, "#4a9eff")
        _precise_slider_row("lbl_fo_far",  self.var_falloff_far,  0.0,  0.25, "#ff9f4a")

        ttk.Separator(ctrl, orient="horizontal").pack(fill=tk.X, pady=6)

        self.var_blur = tk.DoubleVar(value=22.0)
        _slider_row("lbl_blur",  self.var_blur,    1.0,   100.0, "#a0c4ff")

        ttk.Separator(ctrl, orient="horizontal").pack(fill=tk.X, pady=6)

        # ── Flou de bord radial (par sujet) ──────────────────────────────────
        self.lbl_edge_blur_title = ttk.Label(ctrl, foreground="#a0c4ff")
        self.lbl_edge_blur_title.pack(anchor="w", pady=(0, 4))

        # Bouton "Définir le sujet"
        pick_row = ttk.Frame(ctrl)
        pick_row.pack(fill=tk.X, pady=(0, 2))
        self.btn_pick_subject = ttk.Button(pick_row, command=self._toggle_pick_mode)
        self.btn_pick_subject.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.btn_clear_subject = ttk.Button(pick_row, command=self._clear_subject, width=3)
        self.btn_clear_subject.pack(side=tk.LEFT, padx=(4, 0))

        self.lbl_subject_status = ttk.Label(ctrl, foreground="#888899",
                                             font=("Courier New", 9))
        self.lbl_subject_status.pack(anchor="w", pady=(0, 4))

        self.var_edge_strength = tk.DoubleVar(value=0.0)
        self.var_edge_rx       = tk.DoubleVar(value=0.4)
        self.var_edge_ry       = tk.DoubleVar(value=0.3)
        _slider_row("lbl_edge_strength", self.var_edge_strength, 0.0, 1.0,   "#d4a0ff")
        _slider_row("lbl_edge_radius",   self.var_edge_rx,       0.05, 1.0,  "#d4a0ff")
        _slider_row("lbl_edge_ry",       self.var_edge_ry,       0.05, 1.0,  "#d4a0ff")

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

        # ── Éditeur interactif de forme bokeh ────────────────────────────────
        self.lbl_bokeh_editor_k = ttk.Label(ctrl, foreground="#888899",
                                             font=("Courier New", 9))
        self.lbl_bokeh_editor_k.pack(anchor="w", pady=(2, 0))

        editor_row = tk.Frame(ctrl, bg="#1a1a2e")
        editor_row.pack(fill=tk.X, pady=(2, 4))

        self.bokeh_editor = BokehEditorCanvas(
            editor_row, on_change=self._on_bokeh_edit, bg="#0d0d1a")
        self.bokeh_editor.pack(side=tk.LEFT)
        self.bokeh_editor.load_preset("disk")

        self.btn_reset_bokeh = ttk.Button(
            editor_row, command=self._reset_bokeh, width=3)
        self.btn_reset_bokeh.pack(side=tk.LEFT, padx=(6, 0), anchor="n", pady=4)

        # NOTE v13 : le slider « Couches (qualité) » est désormais créé
        # juste au-dessus du bouton « Compute (full resolution) ».
        self.var_steps = tk.DoubleVar(value=16.0)

        self.lbl_render_mode_k = ttk.Label(ctrl)
        self.lbl_render_mode_k.pack(anchor="w", pady=(4, 0))

        self._render_mode = "single"   # Par défaut
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

        # ── Overlay DoF — v13 : les contrôles (case + opacité) sont déplacés
        # dans la barre supérieure de l'image. Seule la légende reste ici.
        self.var_overlay = tk.BooleanVar(value=False)
        self.var_overlay_opacity = tk.DoubleVar(value=0.55)
        # Labels conservés (non packés) pour compatibilité avec _apply_lang :
        self.chk_overlay           = ttk.Label(ctrl)   # remplacé plus bas
        self.lbl_overlay_opacity_k = ttk.Label(ctrl)
        self.lbl_overlay_opacity_v = ttk.Label(ctrl)

        leg = ttk.Frame(ctrl)
        leg.pack(anchor="w", pady=(2, 0))
        self.lbl_leg_far   = ttk.Label(leg, foreground="#ff5555", font=("Courier New", 9))
        self.lbl_leg_sharp = ttk.Label(leg, foreground="#5599ff", font=("Courier New", 9))
        self.lbl_leg_near  = ttk.Label(leg, foreground="#44cc66", font=("Courier New", 9))
        self.lbl_leg_edge  = ttk.Label(leg, foreground="#b04eff", font=("Courier New", 9))
        self.lbl_leg_far.pack(anchor="w")
        self.lbl_leg_sharp.pack(anchor="w")
        self.lbl_leg_near.pack(anchor="w")
        self.lbl_leg_edge.pack(anchor="w")

        ttk.Separator(ctrl, orient="horizontal").pack(fill=tk.X, pady=7)

        # ── Couches (qualité) — v13 : juste au-dessus du bouton de calcul ────
        _slider_row("lbl_layers", self.var_steps, 4, 128, "#a0c4ff")

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

        self.var_compare = tk.BooleanVar(value=False)
        self.btn_compare = ttk.Checkbutton(
            zoom_bar, text="⟺  Compare", variable=self.var_compare,
            command=self._refresh_preview, style="TLabel")
        self.btn_compare.pack(side=tk.RIGHT, padx=(0, 6))

        # ── v13 : contrôles d'overlay DoF déplacés dans la barre supérieure ──
        # Packing RIGHT après btn_compare → s'affiche visuellement à sa gauche.
        self.lbl_overlay_opacity_v = ttk.Label(
            zoom_bar, foreground="#c0c0ff", width=5,
            font=("Courier New", 9))
        self.lbl_overlay_opacity_v.pack(side=tk.RIGHT, padx=(2, 6))
        ttk.Scale(zoom_bar, from_=0.05, to=1.0, orient=tk.HORIZONTAL,
                  length=120,
                  variable=self.var_overlay_opacity,
                  command=lambda _: self._refresh_preview()).pack(
                      side=tk.RIGHT, padx=(2, 2))
        self.chk_overlay = ttk.Checkbutton(
            zoom_bar, variable=self.var_overlay,
            command=self._refresh_preview, style="TLabel")
        self.chk_overlay.pack(side=tk.RIGHT, padx=(8, 4))

        self.canvas = tk.Canvas(right_frame, bg="#0d0d1a", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))
        self._tk_img = None
        self.canvas.bind("<Button-1>",        self._on_canvas_click)
        self.canvas.bind("<B1-Motion>",       self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Motion>",          self._on_canvas_motion)

        # ── Bandeau DnD ───────────────────────────────────────────────────────
        self.lbl_dnd_hint = tk.Label(
            right_frame, text="", bg="#0d0d1a", fg="#555577",
            font=("Courier New", 9), pady=2)
        self.lbl_dnd_hint.pack(fill=tk.X, padx=8, pady=(0, 6))

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
        """Conservé pour compatibilité — le slider a été retiré en v13.
        Le redimensionnement se fait désormais uniquement via le sash."""
        return

    def _on_sash_release(self, event):
        """Sash déplacé à la souris → mémoriser la nouvelle largeur."""
        try:
            x, _ = self._paned.sash_coord(0)
            x = max(self.PANEL_MIN, min(self.PANEL_MAX, x))
            self._panel_width = x
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
        self.lang = code
        self._lang_selector.set_lang(code)
        self._apply_lang()

    def _apply_lang(self):
        L  = self.lang
        wl = max(self._panel_width - 30, 60)
        self.title(f"DOF Tool HQ  v13  —  {L.upper()}")
        self.lbl_title.config(text=t("title", L))
        self.btn_load_rgb.config(text=t("btn_load_rgb", L))
        self.btn_load_depth.config(text=t("btn_load_depth", L))
        if not self._has_rgb():
            self.lbl_rgb.config(text=t("no_file", L), wraplength=wl)
        if not self._has_depth():
            self.lbl_depth.config(text=t("no_file", L), wraplength=wl)
        self.chk_invert.config(text=t("chk_invert", L))
        self.lbl_depth_curve_k.config(text=t("lbl_depth_curve", L))
        self.lbl_curve_preset_k.config(text=t("lbl_curve_preset", L))
        preset_labels = [t(k, L) for k in CURVE_PRESET_T]
        self.cb_curve_preset.config(values=preset_labels)
        # Si la courbe correspond exactement à un preset, afficher son nom
        # dans la nouvelle langue ; sinon laisser vide (courbe custom).
        matched = ""
        for key, tk_key in zip(CURVE_PRESET_KEYS, CURVE_PRESET_T):
            if self.depth_curve_canvas.matches_preset(key):
                matched = t(tk_key, L)
                break
        self.var_curve_preset.set(matched)
        self.lbl_curve_hint.config(text=t("curve_hint", L))
        self.lbl_curve_quality_k.config(text=t("lbl_curve_quality", L))
        self.btn_reset_curve.config(text=t("btn_reset_curve", L))
        self.lbl_focus_title.config(text=t("focus_zone_title", L))
        self.lbl_near_k.config(text=t("lbl_near", L))
        self.lbl_far_k.config( text=t("lbl_far",  L))
        self.lbl_fo_near_k.config(text=t("lbl_falloff_near", L))
        self.lbl_fo_far_k.config( text=t("lbl_falloff_far",  L))
        self.lbl_blur_k.config(text=t("lbl_max_blur", L))
        self.lbl_quality_title.config(text=t("quality_title", L))
        self.lbl_bokeh_shape_k.config(text=t("lbl_bokeh_shape", L))
        self.lbl_bokeh_editor_k.config(text=t("lbl_bokeh_editor", L))
        self.btn_reset_bokeh.config(    text=t("btn_reset_bokeh", L))
        self.lbl_layers_k.config(     text=t("lbl_layers", L))
        kl = [t(k, L) for k in KERNEL_T]
        self.cb_kernel.config(values=kl)
        self.var_kernel_display.set(kl[KERNEL_KEYS.index(self._kernel_key)])
        self.lbl_render_mode_k.config(text=t("lbl_render_mode", L))
        rm_vals = [t("render_single", L), t("render_two_layers", L)]
        self.cb_render_mode.config(values=rm_vals)
        self.var_render_mode_display.set(
            rm_vals[0] if self._render_mode == "single" else rm_vals[1])
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
        self.lbl_leg_edge.config(         text=t("overlay_legend_edge",  L))
        # Flou de bord
        self.lbl_edge_blur_title.config(text=t("lbl_edge_blur_title", L))
        self.btn_pick_subject.config(   text=t("btn_pick_subject",    L))
        self.btn_clear_subject.config(  text="✕")
        self.btn_pick_near.config(      text=t("btn_pick_near",       L))
        self.btn_pick_far.config(       text=t("btn_pick_far",        L))
        self.lbl_edge_strength_k.config(text=t("lbl_edge_strength",  L))
        self.lbl_edge_radius_k.config(  text=t("lbl_edge_radius",    L))
        self.lbl_edge_ry_k.config(      text=t("lbl_edge_ry",        L))
        self._update_subject_label()
        self._update_labels()
        self.focus_viz.redraw()
        # Bandeau DnD
        dnd_text = t("dnd_hint", L) if HAS_DND else t("dnd_no_lib", L)
        self.lbl_dnd_hint.config(text=dnd_text,
                                  fg="#4a9eff" if HAS_DND else "#444466")

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
        self._do_load_rgb(path)
        self._status(t("status_rgb_loaded", L))

    def _load_depth(self):
        L = self.lang
        # Tous les formats lisibles par PIL / OpenCV comme depth map
        depth_exts = (
            "*.png *.tif *.tiff *.pgm *.ppm *.pbm *.bmp "
            "*.exr *.hdr *.pfm *.jp2 *.webp *.jpg *.jpeg"
        )
        path = filedialog.askopenfilename(
            title=t("dlg_depth_title", L),
            filetypes=[
                ("Depth maps", depth_exts),
                ("PNG",        "*.png"),
                ("TIFF",       "*.tif *.tiff"),
                ("PGM / PPM",  "*.pgm *.ppm *.pbm"),
                ("OpenEXR",    "*.exr"),
                ("HDR / PFM",  "*.hdr *.pfm"),
                (t("dlg_all_files", L), "*.*"),
            ])
        if not path: return
        self._do_load_depth(path, auto_detect=True)
        self._status(t("status_depth_loaded", L))

    def _reload_depth(self):
        if self._depth_path:
            self._do_load_depth(self._depth_path)

    def _on_depth_curve_changed(self, _=None):
        """Callback du canvas « Courbe Z-depth ». Re-calcule et rafraîchit."""
        # Une édition manuelle casse la correspondance avec un preset :
        # on vide l'affichage du combobox pour signaler "courbe personnalisée".
        if hasattr(self, "cb_curve_preset"):
            self.var_curve_preset.set("")
        self._refresh_depth_preview()
        self._schedule_compute()

    # ── v13 : aperçu de la Z-depth modifiée ──────────────────────────────────
    def _refresh_depth_preview(self):
        """Met à jour la vignette de la depth après application de la courbe."""
        if not hasattr(self, "lbl_depth_preview"):
            return
        if self.depth_img is None:
            self.lbl_depth_preview.config(image="", height=1)
            self._depth_preview_imgtk = None
            return
        try:
            # Vignette adaptée à la largeur du panneau
            target_w = max(self._panel_width - 40, 120)
            h, w = self.depth_img.shape[:2]
            scale = target_w / float(w)
            tw = int(target_w)
            th = max(1, int(h * scale))
            # Downscale rapide via PIL
            small = np.array(
                Image.fromarray((self.depth_img * 255.0).astype(np.uint8))
                     .resize((tw, th), _BILINEAR),
                dtype=np.float32) / 255.0
            # Appliquer la courbe utilisateur
            try:
                modified = self.depth_curve_canvas.apply(small)
            except Exception:
                modified = small
            modified = np.clip(modified, 0.0, 1.0)
            img8 = (modified * 255.0).astype(np.uint8)
            pil  = Image.fromarray(img8, mode="L")
            self._depth_preview_imgtk = ImageTk.PhotoImage(pil)
            self.lbl_depth_preview.config(image=self._depth_preview_imgtk,
                                          height=th)
        except Exception:
            pass

    # ── v13 : auto-détection de l'inversion Z-depth ──────────────────────────
    def _should_auto_invert(self, raw: np.ndarray) -> bool:
        """Heuristique : convention « 0 = loin ». Si les bords (souvent
        l'arrière-plan) sont en moyenne plus brillants que le centre
        (souvent le sujet), c'est que la depth est encodée « loin = clair »
        et il faut donc l'inverser."""
        try:
            h, w = raw.shape[:2]
            if h < 8 or w < 8:
                return False
            bw = max(2, min(h, w) // 12)            # épaisseur de bordure
            border_pixels = np.concatenate([
                raw[:bw, :].ravel(),
                raw[-bw:, :].ravel(),
                raw[:, :bw].ravel(),
                raw[:, -bw:].ravel(),
            ])
            ch0, ch1 = h // 3, 2 * h // 3
            cw0, cw1 = w // 3, 2 * w // 3
            center_pixels = raw[ch0:ch1, cw0:cw1].ravel()
            border_mean = float(border_pixels.mean())
            center_mean = float(center_pixels.mean())
            # Marge minimale pour éviter les faux positifs sur depth ambiguë
            return (border_mean - center_mean) > 0.05
        except Exception:
            return False

    def _reset_depth_curve(self):
        """Réinitialise la courbe Z-depth à l'identité (linéaire)."""
        self.depth_curve_canvas.load_preset("linear")
        if hasattr(self, "cb_curve_preset"):
            L = self.lang
            self.var_curve_preset.set(t("curve_linear", L))

    def _on_curve_preset_select(self, _=None):
        """L'utilisateur a choisi un preset dans le combobox."""
        display = self.var_curve_preset.get()
        if not display:
            return
        L = self.lang
        for key, tk_key in zip(CURVE_PRESET_KEYS, CURVE_PRESET_T):
            if t(tk_key, L) == display:
                self.depth_curve_canvas.load_preset(key)
                # load_preset déclenche on_change qui vide le combobox —
                # on le ré-écrit juste après pour conserver l'affichage.
                self.var_curve_preset.set(display)
                return

    def _on_curve_quality_changed(self, _=None):
        """Slider « qualité de la courbe » → met à jour le canvas."""
        q = int(self.var_curve_quality.get())
        self.lbl_curve_quality_v.config(text=f"{q}")
        self.depth_curve_canvas.set_quality(q)

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
        self.bokeh_editor.load_preset(self._kernel_key)
        self._schedule_compute()

    def _on_bokeh_edit(self):
        """Appelé quand l'utilisateur modifie un point dans l'éditeur."""
        global _CUSTOM_BOKEH_PTS
        self._bokeh_pts     = self.bokeh_editor.get_points()
        _CUSTOM_BOKEH_PTS   = self._bokeh_pts
        self._kernel_key    = "custom"
        # Mettre à jour le combobox
        L  = self.lang
        kl = [t(k, L) for k in KERNEL_T]
        self.var_kernel_display.set(kl[KERNEL_KEYS.index("custom")])
        self._schedule_compute()

    def _reset_bokeh(self):
        """Réinitialise l'éditeur sur la forme du kernel sélectionné."""
        global _CUSTOM_BOKEH_PTS
        # Revenir au preset précédent ou disk
        key = self._kernel_key if self._kernel_key != "custom" else "disk"
        self.bokeh_editor.load_preset(key)
        self._bokeh_pts   = self.bokeh_editor.get_points()
        _CUSTOM_BOKEH_PTS = self._bokeh_pts
        self._schedule_compute()

    def _on_render_mode_select(self, _=None):
        display = self.var_render_mode_display.get()
        L = self.lang
        if display == t("render_two_layers", L):
            self._render_mode = "two_layers"
        else:
            self._render_mode = "single"
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
        # Précision sliders (near/far/falloff) → mettre à jour les StringVars
        for attr, var, dec in [
            ("lbl_near",    self.var_near,         3),
            ("lbl_far",     self.var_far,           3),
            ("lbl_fo_near", self.var_falloff_near,  3),
            ("lbl_fo_far",  self.var_falloff_far,   3),
        ]:
            sv = getattr(self, attr + "_sv", None)
            if sv is not None:
                sv.set(f"{var.get():.3f}")
        self.lbl_blur_v.config(    text=f"{self.var_blur.get():.0f} px")
        self.lbl_layers_v.config(  text=f"{int(self.var_steps.get())}{t('passes_suffix', L)}")
        self.lbl_overlay_opacity_v.config(
            text=f"{int(self.var_overlay_opacity.get() * 100)}%")
        self.lbl_edge_strength_v.config(text=f"{self.var_edge_strength.get():.2f}")
        self.lbl_edge_radius_v.config(  text=f"{self.var_edge_rx.get():.2f}")
        self.lbl_edge_ry_v.config(      text=f"{self.var_edge_ry.get():.2f}")

    def _schedule_compute(self):
        self._update_labels()
        if self._render_job:
            self.after_cancel(self._render_job)
        self._render_job = self.after(420, self._auto_compute)

    # =========================================================================
    # Sujet — flou de bord radial
    # =========================================================================

    def _toggle_pick_mode(self):
        # Désactiver les autres modes
        self._picking_near = self._picking_far = False
        self._picking_subject = not self._picking_subject
        self._update_cursor()
        self._update_subject_label()
        self._refresh_preview()

    def _toggle_pick_near(self):
        self._picking_subject = self._picking_far = False
        self._picking_near = not self._picking_near
        self._update_cursor()
        self._refresh_preview()

    def _toggle_pick_far(self):
        self._picking_subject = self._picking_near = False
        self._picking_far = not self._picking_far
        self._update_cursor()
        self._refresh_preview()

    def _update_cursor(self):
        any_pick = (self._picking_subject or self._picking_near
                    or self._picking_far)
        self.canvas.config(cursor="crosshair" if any_pick else "")

    def _clear_subject(self):
        self._subject_xy      = None
        self._picking_subject = False
        self._update_cursor()
        self._update_subject_label()
        self._refresh_preview()
        self._schedule_compute()

    def _canvas_to_image_rel(self, ex, ey):
        """Convertit coords canvas → (rx, ry) normalisés [0,1] dans l'image."""
        geom = self._display_geom
        if geom is None:
            return None
        x0, y0, nw, nh = geom
        rx = (ex - x0) / max(nw, 1)
        ry = (ey - y0) / max(nh, 1)
        if 0.0 <= rx <= 1.0 and 0.0 <= ry <= 1.0:
            return float(rx), float(ry)
        return None

    def _depth_at(self, rx, ry):
        """Retourne la valeur depth [0,1] au pixel normalisé (rx, ry)."""
        if not hasattr(self, "_depth_preview") or self._depth_preview is None:
            return None
        H, W = self._depth_preview.shape[:2]
        px = int(rx * W)
        py = int(ry * H)
        px = max(0, min(W - 1, px))
        py = max(0, min(H - 1, py))
        return float(self._depth_preview[py, px])

    def _on_canvas_click(self, event):
        # ── Mode pick sujet ──────────────────────────────────────────────────
        if self._picking_subject:
            rel = self._canvas_to_image_rel(event.x, event.y)
            if rel is not None:
                self._subject_xy = rel
            self._picking_subject = False
            self._update_cursor()
            self._update_subject_label()
            self._refresh_preview()
            self._schedule_compute()
            return

        # ── Mode pick limite proche ──────────────────────────────────────────
        if self._picking_near:
            rel = self._canvas_to_image_rel(event.x, event.y)
            if rel is not None:
                d = self._depth_at(*rel)
                if d is not None:
                    d = round(max(0.0, min(d, self.var_far.get() - 0.001)), 3)
                    self.var_near.set(d)
                    sv = getattr(self, "lbl_near_sv", None)
                    if sv: sv.set(f"{d:.3f}")
                    self._on_slider()
            self._picking_near = False
            self._update_cursor()
            self._refresh_preview()
            return

        # ── Mode pick limite lointaine ───────────────────────────────────────
        if self._picking_far:
            rel = self._canvas_to_image_rel(event.x, event.y)
            if rel is not None:
                d = self._depth_at(*rel)
                if d is not None:
                    d = round(min(1.0, max(d, self.var_near.get() + 0.001)), 3)
                    self.var_far.set(d)
                    sv = getattr(self, "lbl_far_sv", None)
                    if sv: sv.set(f"{d:.3f}")
                    self._on_slider()
            self._picking_far = False
            self._update_cursor()
            self._refresh_preview()
            return

        # ── Mode split compare : début de glissement ─────────────────────────
        if self.var_compare.get() and self.result_img is not None:
            self._split_drag = True

    def _on_canvas_drag(self, event):
        if self._split_drag and self.result_img is not None:
            geom = self._display_geom
            if geom is None:
                return
            x0, _y0, nw, _nh = geom
            rx = (event.x - x0) / max(nw, 1)
            self._split_pos = max(0.02, min(0.98, rx))
            self._refresh_preview()

    def _on_canvas_release(self, event):
        self._split_drag = False

    def _on_canvas_motion(self, event):
        """Change le curseur selon la proximité du séparateur split."""
        if not self.var_compare.get() or self.result_img is None:
            return
        if self._picking_subject or self._picking_near or self._picking_far:
            return
        geom = self._display_geom
        if geom is None:
            return
        x0, _y0, nw, _nh = geom
        split_x = x0 + int(self._split_pos * nw)
        if abs(event.x - split_x) < 12:
            self.canvas.config(cursor="sb_h_double_arrow")
        else:
            self.canvas.config(cursor="")

    def _update_subject_label(self):
        L = self.lang
        if self._picking_subject:
            self.lbl_subject_status.config(
                text=t("picking_prompt", L), foreground="#ffdd44")
        elif self._subject_xy is not None:
            sx, sy = self._subject_xy
            self.lbl_subject_status.config(
                text=f"📍  x={sx:.3f}  y={sy:.3f}", foreground="#60d394")
        else:
            self.lbl_subject_status.config(
                text=t("subject_status_none", L), foreground="#888899")

    # =========================================================================
    # Drag-and-Drop
    # =========================================================================

    def _setup_dnd(self):
        """Enregistre les cibles DnD sur le canvas et la fenêtre principale."""
        for widget in (self.canvas, self.lbl_dnd_hint):
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<DropEnter>>",  self._on_dnd_enter)
            widget.dnd_bind("<<DropLeave>>",  self._on_dnd_leave)
            widget.dnd_bind("<<Drop>>",       self._on_dnd_drop)

    def _on_dnd_enter(self, event):
        self._dnd_hover = True
        self.canvas.config(highlightthickness=3,
                           highlightbackground="#4a9eff")
        self.lbl_dnd_hint.config(fg="#4a9eff",
                                  font=("Courier New", 10, "bold"))
        return event.action

    def _on_dnd_leave(self, event):
        self._dnd_hover = False
        self.canvas.config(highlightthickness=0)
        self.lbl_dnd_hint.config(fg="#4a9eff",
                                  font=("Courier New", 9))

    def _on_dnd_drop(self, event):
        self._dnd_hover = False
        self.canvas.config(highlightthickness=0)
        self.lbl_dnd_hint.config(fg="#4a9eff",
                                  font=("Courier New", 9))

        paths = parse_dnd_paths(event.data)
        if not paths:
            return

        if len(paths) == 1:
            self._load_single_auto(paths[0])
        else:
            # Plusieurs fichiers : on n'en garde que 2 max
            self._load_pair_auto(paths[:2])

    def _load_single_auto(self, path):
        """Charge un seul fichier en détectant automatiquement son type."""
        kind = detect_image_type(path)
        L    = self.lang
        if kind == "depth":
            self._do_load_depth(path, auto_detect=True)
            self._status(t("dnd_loaded_depth", L) + "  —  " + os.path.basename(path))
        else:
            self._do_load_rgb(path)
            self._status(t("dnd_loaded_rgb", L) + "  —  " + os.path.basename(path))

    def _load_pair_auto(self, paths):
        """
        Charge deux fichiers en assignant automatiquement RGB et depth.
        Stratégie :
          • Si un seul des deux est clairement NB → depth, l'autre → rgb
          • Si les deux semblent identiques (ambiguïté) → charger dans l'ordre
            et avertir l'utilisateur.
        """
        L      = self.lang
        kinds  = [detect_image_type(p) for p in paths]

        if kinds[0] == "depth" and kinds[1] == "rgb":
            depth_path, rgb_path = paths[0], paths[1]
        elif kinds[0] == "rgb" and kinds[1] == "depth":
            rgb_path, depth_path = paths[0], paths[1]
        elif kinds[0] == "rgb" and kinds[1] == "rgb":
            # Les deux semblent en couleurs : on prend le plus petit comme depth
            # (les depth maps en RGB 8-bit sont souvent plus légères)
            sizes = [os.path.getsize(p) for p in paths]
            if sizes[0] <= sizes[1]:
                rgb_path, depth_path = paths[1], paths[0]
            else:
                rgb_path, depth_path = paths[0], paths[1]
            self.after(100, lambda: self._status(t("dnd_ambiguous", L)))
        else:
            # Les deux semblent NB : prendre le plus grand comme depth 16-bit
            sizes = [os.path.getsize(p) for p in paths]
            depth_path = paths[0] if sizes[0] >= sizes[1] else paths[1]
            rgb_path   = paths[1] if sizes[0] >= sizes[1] else paths[0]
            self.after(100, lambda: self._status(t("dnd_ambiguous", L)))

        self._do_load_rgb(rgb_path)
        self._do_load_depth(depth_path, auto_detect=True)
        self._status(
            t("dnd_loaded_rgb", L)   + ": " + os.path.basename(rgb_path)   + "  |  " +
            t("dnd_loaded_depth", L) + ": " + os.path.basename(depth_path))

    def _do_load_rgb(self, path):
        """Charge l'image RGB depuis un chemin (sans dialogue)."""
        L = self.lang
        self.rgb_img = np.array(Image.open(path).convert("RGB"))
        wl = max(self._panel_width - 30, 60)
        self.lbl_rgb.config(text=os.path.basename(path)[-40:],
                            foreground="#60d394", wraplength=wl)
        self._auto_compute()

    def _do_load_depth(self, path, auto_detect=False):
        """Charge la depth map depuis un chemin (sans dialogue).
        Si auto_detect=True (v13), tente de déterminer automatiquement
        s'il faut inverser la Z-depth pour respecter la convention 0=loin."""
        self._depth_path = path
        if auto_detect:
            raw = load_depth(path, invert=False)
            if self._should_auto_invert(raw):
                self.depth_img = 1.0 - raw
                # var_invert.set() ne déclenche pas la commande du Checkbutton
                self.var_invert.set(True)
            else:
                self.depth_img = raw
                self.var_invert.set(False)
        else:
            self.depth_img = load_depth(path, self.var_invert.get())
        if self._has_rgb() and self.depth_img.shape[:2] != self.rgb_img.shape[:2]:
            h, w = self.rgb_img.shape[:2]
            if HAS_CV2:
                self.depth_img = cv2.resize(self.depth_img, (w, h),
                                            interpolation=cv2.INTER_LINEAR)
            else:
                self.depth_img = np.array(
                    Image.fromarray(self.depth_img, mode="F")
                        .resize((w, h), _BILINEAR),
                    dtype=np.float32)
        wl = max(self._panel_width - 30, 60)
        self.lbl_depth.config(text=os.path.basename(path)[-40:],
                              foreground="#60d394", wraplength=wl)
        self._refresh_depth_preview()
        self._auto_compute()

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
        if self._computing:
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
                scale = min(1.0, self.PREVIEW_MAX_DIM / max(rgb.shape[:2]))
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
            else:
                scale = 1.0

            near         = self.var_near.get()
            far          = self.var_far.get()
            falloff_near = self.var_falloff_near.get()
            falloff_far  = self.var_falloff_far.get()
            max_b        = self.var_blur.get()
            if preview_mode and scale < 1.0:
                # Redimensionner le rayon de flou proportionnellement.
                # On NE force PLUS de minimum : si l'utilisateur a choisi 0
                # ou très peu, l'aperçu doit refléter ce choix.
                max_b = max_b * scale

            steps    = int(self.var_steps.get())
            # ── Kernel personnalisé ──────────────────────────────────────────
            if self._kernel_key == "custom":
                global _CUSTOM_BOKEH_PTS
                _CUSTOM_BOKEH_PTS = list(self._bokeh_pts)

            # ── v12 : courbe Z-depth (remap tonal via canvas éditeur) ───────
            # Appliquée AVANT tout le reste pour que compute_blur_radius_map,
            # render_dof et la prévisualisation « Z-Depth » voient tous
            # la même carte remappée.
            depth = self.depth_curve_canvas.apply(depth)

            blur_map = compute_blur_radius_map(depth, near, far, max_b,
                                               falloff_near, falloff_far)

            # ── Flou de bord radial elliptique ───────────────────────────────
            edge_strength = self.var_edge_strength.get()
            edge_rx       = self.var_edge_rx.get()
            edge_ry       = self.var_edge_ry.get()
            blur_map = apply_edge_blur(blur_map, self._subject_xy,
                                       edge_strength, edge_rx, edge_ry, max_b)

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

            # ── Overlay flou de bord radial (violet) ─────────────────────────
            if self._subject_xy is not None and self.var_edge_strength.get() > 1e-4:
                h_a, w_a = arr.shape[:2]
                edge_ov = make_edge_blur_overlay(
                    (h_a, w_a), self._subject_xy,
                    self.var_edge_strength.get(),
                    self.var_edge_rx.get(),
                    self.var_edge_ry.get())
                arr = composite_overlay(arr, edge_ov,
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
        x0    = cw // 2 - nw // 2
        y0    = ch // 2 - nh // 2
        self._display_geom = (x0, y0, nw, nh)

        # ── Vue comparative split ─────────────────────────────────────────────
        if (self.var_compare.get()
                and self.result_img is not None
                and self._has_rgb()):
            orig_r = Image.fromarray(self.rgb_img.astype(np.uint8)).resize(
                (nw, nh), _BILINEAR)
            res_r  = Image.fromarray(self.result_img.astype(np.uint8)).resize(
                (nw, nh), _BILINEAR)
            split_px = int(self._split_pos * nw)
            # Composer : gauche = original, droite = résultat
            composite = Image.new("RGB", (nw, nh))
            composite.paste(orig_r.crop((0, 0, split_px, nh)), (0, 0))
            composite.paste(res_r.crop((split_px, 0, nw, nh)), (split_px, 0))
            self._tk_img = ImageTk.PhotoImage(composite)
            self.canvas.delete("all")
            self.canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER,
                                      image=self._tk_img)
            # Ligne de séparation
            lx = x0 + split_px
            self.canvas.create_line(lx, y0, lx, y0 + nh,
                                    fill="#ffffff", width=2, dash=(4, 3))
            # Étiquettes ORIGINAL / RÉSULTAT
            self.canvas.create_text(
                lx - 6, y0 + 18, text=t("view_original", self.lang).upper(),
                anchor="e",
                fill="#ffffff", font=("Courier New", 9, "bold"))
            self.canvas.create_text(
                lx + 6, y0 + 18, text=t("view_result", self.lang).upper(),
                anchor="w",
                fill="#60d394", font=("Courier New", 9, "bold"))
            # Poignée centrale
            hx, hy = lx, y0 + nh // 2
            self.canvas.create_oval(hx-8, hy-8, hx+8, hy+8,
                                    fill="#1a1a2e", outline="#ffffff", width=2)
            self.canvas.create_line(hx-5, hy, hx+5, hy, fill="#ffffff", width=2)
            self.canvas.create_line(hx-3, hy-3, hx, hy, fill="#ffffff", width=1)
            self.canvas.create_line(hx+3, hy-3, hx, hy, fill="#ffffff", width=1)
            self.lbl_zoom_v.config(text=f"×{zoom:.2f}")
            return

        # ── Affichage normal ─────────────────────────────────────────────────
        pil = Image.fromarray(arr.astype(np.uint8)).resize((nw, nh), _BILINEAR)
        self._tk_img = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER,
                                  image=self._tk_img)

        # ── Marqueur sujet ───────────────────────────────────────────────────
        if self._subject_xy is not None:
            sx, sy = self._subject_xy
            px = x0 + int(sx * nw)
            py = y0 + int(sy * nh)
            R  = 10
            self.canvas.create_oval(px-R, py-R, px+R, py+R,
                                    outline="#ff4444", width=2)
            self.canvas.create_line(px-R-4, py, px+R+4, py,
                                    fill="#ff4444", width=2)
            self.canvas.create_line(px, py-R-4, px, py+R+4,
                                    fill="#ff4444", width=2)

        # ── Bandeaux de mode pick ────────────────────────────────────────────
        if self._picking_subject:
            self.canvas.create_text(cw // 2, 20,
                text="📍  " + t("picking_prompt", self.lang),
                fill="#ffdd44", font=("Courier New", 11, "bold"))
        elif self._picking_near:
            self.canvas.create_text(cw // 2, 20,
                text="🎯  " + t("btn_pick_near", self.lang),
                fill="#60d394", font=("Courier New", 11, "bold"))
        elif self._picking_far:
            self.canvas.create_text(cw // 2, 20,
                text="🎯  " + t("btn_pick_far", self.lang),
                fill="#ff9f6b", font=("Courier New", 11, "bold"))

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
