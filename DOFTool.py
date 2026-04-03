"""
Depth of Field Tool  –  High Quality
=====================================
Image RGB + Z-depth 16-bit PNG → flou de profondeur photoréaliste.

Langues : 🇫🇷 Français   🇬🇧 English    🇧🇪 Nederlands  🇩🇪 Deutsch
          🇨🇳 中文       🇯🇵 日本語     🇸🇦 العربية     🏴 Klingon
          🏴 Wallon li.  🇷🇺 Русский    🇮🇳 हिन्दी       🇪🇸 Español
          🇧🇷 Português  🇧🇩 বাংলা

Dépendances : pip install numpy pillow scipy
              (optionnel)  pip install opencv-python
"""

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import math

try:
    from scipy.ndimage import convolve
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# =============================================================================
# Langues disponibles  (2 rangées de 7 drapeaux)
# =============================================================================

LANGUAGES = ["fr", "en", "nl", "de", "zh", "ja", "ar",
             "kl", "wa", "ru", "hi", "es", "pt", "bn"]

FLAG_ROWS = [LANGUAGES[:7], LANGUAGES[7:]]


# =============================================================================
# Traductions
# =============================================================================

T = {
    "title": {
        "fr": "── DOF TOOL HQ ──",      "en": "── DOF TOOL HQ ──",
        "nl": "── DOF TOOL HQ ──",      "de": "── DOF TOOL HQ ──",
        "zh": "── DOF 工具 HQ ──",      "ja": "── DOF ツール HQ ──",
        "ar": "── أداة DOF HQ ──",      "kl": "── DOF jan HQ ──",
        "wa": "── OUTIL DOF HQ ──",     "ru": "── ИНСТРУМЕНТ DOF ──",
        "hi": "── DOF टूल HQ ──",       "es": "── HERRAMIENTA DOF ──",
        "pt": "── FERRAMENTA DOF ──",   "bn": "── DOF টুল HQ ──",
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
        "fr": "Nombre de couches (qualité)",
        "en": "Number of layers (quality)",
        "nl": "Aantal lagen (kwaliteit)",
        "de": "Anzahl Ebenen (Qualität)",
        "zh": "层数（品质）",
        "ja": "レイヤー数（品質）",
        "ar": "عدد الطبقات (الجودة)",
        "kl": "mI' tep (quv)",
        "wa": "Nombe di coûtches (qualité)",
        "ru": "Кол-во слоёв (качество)",
        "hi": "परतों की संख्या (गुणवत्ता)",
        "es": "Número de capas (calidad)",
        "pt": "Número de camadas (qualidade)",
        "bn": "স্তরের সংখ্যা (মান)",
    },
    "chk_bleed": {
        "fr": "✦  Anti-bleeding (bords nets)",
        "en": "✦  Anti-bleeding (sharp edges)",
        "nl": "✦  Anti-doorloop (scherpe randen)",
        "de": "✦  Anti-Bleeding (scharfe Kanten)",
        "zh": "✦  抗渗色（清晰边缘）",
        "ja": "✦  アンチブリード（鮮明なエッジ）",
        "ar": "✦  مانع النزيف (حواف حادة)",
        "kl": "✦  Anti-bIQ (nIH mIw)",
        "wa": "✦  Anti-saignance (bwès netes)",
        "ru": "✦  Анти-растечка (чёткие края)",
        "hi": "✦  एंटी-ब्लीड (स्पष्ट किनारे)",
        "es": "✦  Anti-sangrado (bordes nítidos)",
        "pt": "✦  Anti-sangramento (bordas nítidas)",
        "bn": "✦  অ্যান্টি-ব্লিড (তীক্ষ্ণ প্রান্ত)",
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
    "progress_conv": {
        "fr": "Convolution",  "en": "Convolution",  "nl": "Convolutie",
        "de": "Faltung",      "zh": "卷积",          "ja": "畳み込み",
        "ar": "الطيّ",        "kl": "mIwHom",       "wa": "Convolution",
        "ru": "Свёртка",      "hi": "कनवोल्यूशन",  "es": "Convolución",
        "pt": "Convolução",   "bn": "কনভোলিউশন",
    },
    "progress_comp": {
        "fr": "Composition",  "en": "Composition",  "nl": "Compositie",
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
}


def t(key: str, lang: str) -> str:
    """Retourne la traduction de key dans lang, fallback anglais."""
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


def apply_kernel(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if HAS_SCIPY:
        return np.stack([
            convolve(img[:, :, c].astype(np.float32), kernel, mode='reflect')
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
# Moteur DoF
# =============================================================================

def load_depth(path: str, invert: bool = False) -> np.ndarray:
    img = Image.open(path)
    arr = (np.array(img, dtype=np.float32)
           if img.mode in ("I", "I;16")
           else np.array(img.convert("L"), dtype=np.float32))
    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()
    return 1.0 - arr if invert else arr


def compute_blur_radius_map(depth, focus_near, focus_far, max_blur, falloff):
    d, blur = depth.copy(), np.zeros_like(depth)
    # Zone de transition proche (soft near)
    sn = (d < focus_near) & (d >= focus_near - falloff)
    if sn.any() and falloff > 1e-6:
        tv = np.clip((focus_near - d[sn]) / falloff, 0.0, 1.0)
        blur[sn] = tv * tv * (3.0 - 2.0 * tv)
    # Zone très proche (hard near)
    hn = d < (focus_near - falloff)
    if hn.any() and (focus_near - falloff) > 1e-6:
        tv = np.clip((focus_near - falloff - d[hn]) / max(focus_near - falloff, 1e-6), 0.0, 1.0)
        blur[hn] = 1.0 + tv
    # Zone de transition lointaine (soft far)
    sf = (d > focus_far) & (d <= focus_far + falloff)
    if sf.any() and falloff > 1e-6:
        tv = np.clip((d[sf] - focus_far) / falloff, 0.0, 1.0)
        blur[sf] = tv * tv * (3.0 - 2.0 * tv)
    # Zone très lointaine (hard far)
    hf = d > (focus_far + falloff)
    rf = 1.0 - (focus_far + falloff)
    if hf.any() and rf > 1e-6:
        tv = np.clip((d[hf] - focus_far - falloff) / rf, 0.0, 1.0)
        blur[hf] = 1.0 + tv
    blur = np.clip(blur, 0.0, None)
    if blur.max() > 0:
        blur = blur / blur.max() * max_blur
    return blur


def render_dof(rgb, blur_map, depth, kernel_key="disk",
               steps=16, bleed_correction=True,
               progress_cb=None, lang="fr") -> np.ndarray:
    h, w  = rgb.shape[:2]
    max_r = blur_map.max()
    if max_r < 0.5:
        return rgb.copy()

    maker     = KERNEL_FN.get(kernel_key, make_disk_kernel)
    radii     = np.linspace(0.0, max_r, steps + 1)
    step_size = radii[1] - radii[0] if steps > 0 else 1.0
    total     = len(radii) * 2
    done      = 0

    blurred = {}
    for r in radii:
        blurred[r] = (rgb.astype(np.float32) if r < 0.5
                      else apply_kernel(rgb, maker(r)))
        done += 1
        if progress_cb:
            progress_cb(done, total, f"{t('progress_conv', lang)}  {done}/{len(radii)}")

    accum  = np.zeros((h, w, 3), dtype=np.float32)
    weight = np.zeros((h, w),    dtype=np.float32)

    for r in reversed(radii):
        wp = np.maximum(0.0, 1.0 - np.abs(blur_map - r) / (step_size + 1e-6))
        if bleed_correction and r > 0.5:
            sm = (blur_map < step_size * 0.5).astype(np.float32)
            wp = wp * (1.0 - sm * np.clip(r / max_r, 0, 1) * 0.85)
        accum  += blurred[r] * wp[:, :, np.newaxis]
        weight += wp
        done += 1
        if progress_cb:
            progress_cb(done, total, f"{t('progress_comp', lang)}  {done - len(radii)}/{len(radii)}")

    return np.clip(accum / np.maximum(weight, 1e-6)[:, :, np.newaxis],
                   0, 255).astype(np.uint8)


# =============================================================================
# Widget drapeaux  (pur Canvas Tkinter — zéro image externe, zéro couleur alpha)
# =============================================================================

class FlagButton(tk.Canvas):
    W, H = 36, 24

    def __init__(self, master, lang_code: str, command, **kw):
        super().__init__(master, width=self.W, height=self.H,
                         highlightthickness=2, cursor="hand2", **kw)
        self.lang_code = lang_code
        self.command   = command
        self._selected = False
        self._draw()
        self.bind("<Button-1>", lambda _: self.command(self.lang_code))

    def set_selected(self, v: bool):
        self._selected = v
        self.config(highlightbackground="#a0c4ff" if v else "#2a2a4a")

    # ── helpers ───────────────────────────────────────────────────────────
    def _star(self, cx, cy, r, color):
        pts = []
        for i in range(10):
            a = math.radians(i * 36 - 90)
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

    # ── dessin selon le code langue ───────────────────────────────────────
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
            # Arabie Saoudite stylisée : fond vert + ornements blancs
            self.create_rectangle(0, 0, W, H, fill="#006C35", outline="")
            self.create_line(4, H//2+2, W-4, H//2+2, fill="#FFFFFF", width=2)
            self.create_arc(4, H//2-2, 14, H//2+8, start=0, extent=180,
                            outline="#FFFFFF", width=2, style="arc")
            self.create_line(6, H//2-4, W-6, H//2-4, fill="#FFFFFF", width=1)
            self.create_line(9, H//2-7, W-9, H//2-7, fill="#FFFFFF", width=1)

        elif lc == "kl":
            # Klingon (inventé) : fond noir, losange bordeaux, symbole trefoil doré
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
            # Wallon de Liège : fond jaune, coq wallon rouge simplifié
            self.create_rectangle(0, 0, W, H, fill="#FBDB00", outline="")
            # Corps
            self.create_oval(8, 7, 21, 19, fill="#CC0000", outline="")
            # Tête
            self.create_oval(19, 4, 27, 12, fill="#CC0000", outline="")
            # Crête
            self.create_polygon([20,4, 24,1, 27,5], fill="#CC0000", outline="")
            # Bec
            self.create_polygon([26,7, 31,9, 26,10], fill="#FBDB00", outline="")
            # Queue
            self.create_polygon([8,10, 2,5, 2,18], fill="#CC0000", outline="")
            # Patte
            self.create_line(14, 19, 12, 23, fill="#CC0000", width=2)
            self.create_line(14, 19, 16, 23, fill="#CC0000", width=2)

        elif lc == "ru":
            self._hband("#FFFFFF", "#0039A6", "#D52B1E")

        elif lc == "hi":
            # Inde : safran | blanc + roue | vert
            self._hband("#FF9933", "#FFFFFF", "#138808")
            cx, cy, r = W//2, H//2, 4
            self.create_oval(cx-r, cy-r, cx+r, cy+r, outline="#000080", width=1, fill="")
            for i in range(8):
                a = math.radians(i * 45)
                self.create_line(cx, cy, cx+math.cos(a)*r, cy+math.sin(a)*r,
                                 fill="#000080", width=1)

        elif lc == "es":
            # Espagne : rouge | jaune (large) | rouge
            self.create_rectangle(0,       0, W, H//4,   fill="#AA151B", outline="")
            self.create_rectangle(0,   H//4, W, 3*H//4,  fill="#F1BF00", outline="")
            self.create_rectangle(0, 3*H//4, W, H,       fill="#AA151B", outline="")

        elif lc == "pt":
            # Portugal : vert (1/3) | rouge (2/3) + petit écu jaune
            self.create_rectangle(0,    0, W//3, H, fill="#006600", outline="")
            self.create_rectangle(W//3, 0, W,    H, fill="#FF0000", outline="")
            cx = W//3
            self.create_oval(cx-4, H//2-4, cx+4, H//2+4,
                             fill="#FFD700", outline="#003399", width=1)

        elif lc == "bn":
            # Bangladesh : fond vert + cercle rouge légèrement décalé à gauche
            self.create_rectangle(0, 0, W, H, fill="#006A4E", outline="")
            cx, cy = W//2 - 2, H//2
            r = min(W, H) // 3
            self.create_oval(cx-r, cy-r, cx+r, cy+r, fill="#F42A41", outline="")

        # Cadre fin opaque (pas de canal alpha)
        self.create_rectangle(0, 0, W-1, H-1, outline="#333333", fill="")


# =============================================================================
# Widget visualiseur zone de netteté
# =============================================================================

class FocusZoneCanvas(tk.Canvas):
    H   = 82
    PAD = 16

    def __init__(self, master, var_near, var_far, var_falloff,
                 callback, lang_getter, **kw):
        super().__init__(master, height=self.H, bg="#0d0d1a",
                         highlightthickness=1, highlightbackground="#2a2a4a", **kw)
        self.var_near    = var_near
        self.var_far     = var_far
        self.var_falloff = var_falloff
        self.callback    = callback
        self.lang_getter = lang_getter
        self._drag       = None
        self.bind("<Configure>",       lambda _: self.redraw())
        self.bind("<ButtonPress-1>",   self._on_press)
        self.bind("<B1-Motion>",       self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)

    def redraw(self):
        self.delete("all")
        W = self.winfo_width()
        if W < 30:
            return
        H, P = self.H, self.PAD
        IW   = W - 2 * P
        near = self.var_near.get()
        far  = self.var_far.get()
        fo   = self.var_falloff.get()
        lang = self.lang_getter()

        def xp(d):  return P + d * IW
        def yp(tv): return H - P - tv * (H - 2*P)

        self.create_rectangle(xp(near), P+2, xp(far), H-P, fill="#132613", outline="")

        N = max(int(IW * 2), 120)
        for side in ("left", "right"):
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
            self.create_line(pts, fill="#4a9eff", width=2, smooth=True)

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

    def __init__(self):
        super().__init__()
        self.title("Depth of Field Tool  —  HQ")
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

        # ── Panneau gauche ────────────────────────────────────────────────
        ctrl = ttk.Frame(self, padding=14)
        ctrl.pack(side=tk.LEFT, fill=tk.Y)

        # ── Drapeaux : 2 rangées de 7 ─────────────────────────────────────
        flag_frame = ttk.Frame(ctrl)
        flag_frame.pack(fill=tk.X, pady=(0, 8))

        self._flag_btns = {}
        for row_langs in FLAG_ROWS:
            row = ttk.Frame(flag_frame)
            row.pack(pady=2)
            for lc in row_langs:
                fb = FlagButton(row, lc, self._switch_lang, bg="#1a1a2e")
                fb.pack(side=tk.LEFT, padx=2)
                self._flag_btns[lc] = fb
        self._flag_btns[self.lang].set_selected(True)

        # ── Titre ─────────────────────────────────────────────────────────
        self.lbl_title = ttk.Label(ctrl, font=("Courier New", 12, "bold"),
                                   foreground="#a0c4ff")
        self.lbl_title.pack(pady=(0, 8))

        # ── Fichiers ──────────────────────────────────────────────────────
        self.btn_load_rgb = ttk.Button(ctrl, command=self._load_rgb)
        self.btn_load_rgb.pack(fill=tk.X, pady=3)
        self.lbl_rgb = ttk.Label(ctrl, foreground="#555577")
        self.lbl_rgb.pack()

        self.btn_load_depth = ttk.Button(ctrl, command=self._load_depth)
        self.btn_load_depth.pack(fill=tk.X, pady=(8, 3))
        self.lbl_depth = ttk.Label(ctrl, foreground="#555577")
        self.lbl_depth.pack()

        self.var_invert = tk.BooleanVar(value=False)
        self.chk_invert = ttk.Checkbutton(ctrl, variable=self.var_invert,
                                           command=self._reload_depth)
        self.chk_invert.pack(anchor="w", pady=(5, 0))

        ttk.Separator(ctrl, orient="horizontal").pack(fill=tk.X, pady=9)

        # ── Zone de netteté ───────────────────────────────────────────────
        self.lbl_focus_title = ttk.Label(ctrl, foreground="#a0c4ff")
        self.lbl_focus_title.pack(anchor="w")

        self.var_near    = tk.DoubleVar(value=0.35)
        self.var_far     = tk.DoubleVar(value=0.65)
        self.var_falloff = tk.DoubleVar(value=0.06)

        self.focus_viz = FocusZoneCanvas(
            ctrl, self.var_near, self.var_far, self.var_falloff,
            callback=self._on_focus_changed,
            lang_getter=lambda: self.lang)
        self.focus_viz.pack(fill=tk.X, pady=(4, 8))

        def _slider_row(lbl_attr, var, from_, to_, color):
            lbl = ttk.Label(ctrl)
            setattr(self, lbl_attr + "_k", lbl)
            lbl.pack(anchor="w")
            row = ttk.Frame(ctrl); row.pack(fill=tk.X, pady=(0, 4))
            ttk.Scale(row, from_=from_, to=to_, orient=tk.HORIZONTAL,
                      variable=var,
                      command=lambda _: self._on_slider()).pack(
                          side=tk.LEFT, fill=tk.X, expand=True)
            val_lbl = ttk.Label(row, foreground=color, width=8,
                                font=("Courier New", 10))
            val_lbl.pack(side=tk.LEFT)
            setattr(self, lbl_attr + "_v", val_lbl)

        _slider_row("lbl_near",    self.var_near,    0.0,   1.0,   "#60d394")
        _slider_row("lbl_far",     self.var_far,     0.0,   1.0,   "#ff9f6b")
        _slider_row("lbl_fo",      self.var_falloff, 0.0,   0.25,  "#c0c0ff")

        ttk.Separator(ctrl, orient="horizontal").pack(fill=tk.X, pady=6)

        self.var_blur = tk.DoubleVar(value=22.0)
        _slider_row("lbl_blur",    self.var_blur,    1.0,   100.0, "#a0c4ff")

        ttk.Separator(ctrl, orient="horizontal").pack(fill=tk.X, pady=6)

        # ── Qualité ───────────────────────────────────────────────────────
        self.lbl_quality_title = ttk.Label(ctrl, foreground="#a0c4ff")
        self.lbl_quality_title.pack(anchor="w")

        self.lbl_bokeh_shape_k = ttk.Label(ctrl)
        self.lbl_bokeh_shape_k.pack(anchor="w", pady=(4, 0))

        self.var_kernel_display = tk.StringVar()
        self.cb_kernel = ttk.Combobox(ctrl, textvariable=self.var_kernel_display,
                                       state="readonly", width=24)
        self.cb_kernel.pack(fill=tk.X, pady=(2, 5))
        self.cb_kernel.bind("<<ComboboxSelected>>", self._on_kernel_select)

        self.var_steps = tk.IntVar(value=16)
        _slider_row("lbl_layers",  self.var_steps,   4,     32,    "#a0c4ff")

        self.var_bleed = tk.BooleanVar(value=True)
        self.chk_bleed = ttk.Checkbutton(ctrl, variable=self.var_bleed,
                                          command=self._schedule_compute)
        self.chk_bleed.pack(anchor="w", pady=(3, 0))

        ttk.Separator(ctrl, orient="horizontal").pack(fill=tk.X, pady=7)

        # ── Aperçu ────────────────────────────────────────────────────────
        self.lbl_preview_title = ttk.Label(ctrl)
        self.lbl_preview_title.pack(anchor="w")

        self.var_view = tk.StringVar(value="result")
        for val, attr in [("original", "rb_original"), ("depth", "rb_depth"),
                          ("blur_map", "rb_blur_map"), ("result", "rb_result")]:
            rb = ttk.Radiobutton(ctrl, variable=self.var_view, value=val,
                                 command=self._refresh_preview, style="TLabel")
            rb.pack(anchor="w")
            setattr(self, attr, rb)

        ttk.Separator(ctrl, orient="horizontal").pack(fill=tk.X, pady=7)

        self.btn_compute = ttk.Button(ctrl, command=self._compute_full)
        self.btn_compute.pack(fill=tk.X, pady=3)
        self.btn_export  = ttk.Button(ctrl, command=self._export)
        self.btn_export.pack(fill=tk.X, pady=3)

        self.var_progress = tk.DoubleVar(value=0.0)
        self.progressbar  = ttk.Progressbar(ctrl, variable=self.var_progress,
                                             maximum=100.0, length=260,
                                             mode="determinate")
        self.progressbar.pack(fill=tk.X, pady=(8, 2))
        self.lbl_progress = ttk.Label(ctrl, text="", foreground="#8888aa",
                                       font=("Courier New", 9))
        self.lbl_progress.pack(anchor="w")

        self.lbl_status = ttk.Label(ctrl, foreground="#60d394", wraplength=270)
        self.lbl_status.pack(pady=(8, 0))

        # ── Canvas image ──────────────────────────────────────────────────
        cf = ttk.Frame(self)
        cf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.canvas = tk.Canvas(cf, bg="#0d0d1a", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self._tk_img = None

        self._apply_lang()

    # =========================================================================
    # Langue
    # =========================================================================

    def _switch_lang(self, code: str):
        if code == self.lang:
            return
        for lc, fb in self._flag_btns.items():
            fb.set_selected(lc == code)
        self.lang = code
        self._apply_lang()

    def _apply_lang(self):
        L = self.lang
        self.title(f"DOF Tool HQ  —  {L.upper()}")
        self.lbl_title.config(text=t("title", L))

        self.btn_load_rgb.config(text=t("btn_load_rgb", L))
        self.btn_load_depth.config(text=t("btn_load_depth", L))
        if not self._has_rgb():
            self.lbl_rgb.config(text=t("no_file", L))
        if not self._has_depth():
            self.lbl_depth.config(text=t("no_file", L))

        self.chk_invert.config(text=t("chk_invert", L))
        self.lbl_focus_title.config(text=t("focus_zone_title", L))
        self.lbl_near_k.config(text=t("lbl_near", L))
        self.lbl_far_k.config( text=t("lbl_far",  L))
        self.lbl_fo_k.config(  text=t("lbl_falloff", L))
        self.lbl_blur_k.config(text=t("lbl_max_blur", L))

        self.lbl_quality_title.config(text=t("quality_title", L))
        self.lbl_bokeh_shape_k.config(text=t("lbl_bokeh_shape", L))
        self.lbl_layers_k.config(     text=t("lbl_layers", L))
        self.chk_bleed.config(        text=t("chk_bleed", L))

        kl = [t(k, L) for k in KERNEL_T]
        self.cb_kernel.config(values=kl)
        self.var_kernel_display.set(kl[KERNEL_KEYS.index(self._kernel_key)])

        self.lbl_preview_title.config(text=t("preview_title", L))
        self.rb_original.config(text=t("view_original", L))
        self.rb_depth.config(   text=t("view_depth",    L))
        self.rb_blur_map.config(text=t("view_blur_map", L))
        self.rb_result.config(  text=t("view_result",   L))

        self.btn_compute.config(text=t("btn_compute", L))
        self.btn_export.config( text=t("btn_export",  L))
        self.lbl_status.config( text=t("status_ready", L))

        self._update_labels()
        self.focus_viz.redraw()

    def _has_rgb(self):   return self.rgb_img   is not None
    def _has_depth(self): return self.depth_img  is not None

    # =========================================================================
    # Chargement
    # =========================================================================

    def _load_rgb(self):
        L = self.lang
        path = filedialog.askopenfilename(
            title=t("dlg_rgb_title", L),
            filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff"),
                       (t("dlg_all_files", L), "*.*")])
        if not path: return
        self.rgb_img = np.array(Image.open(path).convert("RGB"))
        self.lbl_rgb.config(text=path.split("/")[-1][-34:], foreground="#60d394")
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
        self.lbl_depth.config(text=path.split("/")[-1][-34:], foreground="#60d394")
        self._status(t("status_depth_loaded", L))
        self._auto_compute()

    def _reload_depth(self):
        if self._depth_path:
            self.depth_img = load_depth(self._depth_path, self.var_invert.get())
            self._auto_compute()

    # =========================================================================
    # Combobox noyau
    # =========================================================================

    def _on_kernel_select(self, _=None):
        display = self.var_kernel_display.get()
        L = self.lang
        for key, tk_key in zip(KERNEL_KEYS, KERNEL_T):
            if t(tk_key, L) == display:
                self._kernel_key = key
                break
        self._schedule_compute()

    # =========================================================================
    # Sliders / labels
    # =========================================================================

    def _on_slider(self):
        n, f = self.var_near.get(), self.var_far.get()
        if n >= f:
            self.var_far.set(min(n + 0.02, 1.0))
        self._update_labels()
        self.focus_viz.redraw()
        self._schedule_compute()

    def _on_focus_changed(self):
        self._update_labels()
        self._schedule_compute()

    def _update_labels(self):
        L = self.lang
        self.lbl_near_v.config(   text=f"{self.var_near.get():.2f}")
        self.lbl_far_v.config(    text=f"{self.var_far.get():.2f}")
        self.lbl_fo_v.config(     text=f"{self.var_falloff.get():.2f}")
        self.lbl_blur_v.config(   text=f"{self.var_blur.get():.0f} px")
        self.lbl_layers_v.config( text=f"{int(self.var_steps.get())}{t('passes_suffix', L)}")

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
    # Calcul
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

    def _run_compute(self, preview_mode: bool):
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
                        Image.fromarray(rgb).resize((w, h), Image.BILINEAR))
                    depth = np.array(
                        Image.fromarray((depth * 65535).astype(np.uint16))
                            .resize((w, h), Image.BILINEAR),
                        dtype=np.float32) / 65535.0

            near    = self.var_near.get()
            far     = self.var_far.get()
            falloff = self.var_falloff.get()
            max_b   = self.var_blur.get()
            if preview_mode:
                s     = min(1.0, 560 / max(self.rgb_img.shape[:2]))
                max_b = max(max_b * s, 2.0)

            steps    = int(self.var_steps.get())
            bleed    = self.var_bleed.get()
            blur_map = compute_blur_radius_map(depth, near, far, max_b, falloff)

            def progress_cb(done, total, label):
                self.after(0, self._set_progress, done, total, label)

            result = render_dof(rgb, blur_map, depth,
                                kernel_key=self._kernel_key,
                                steps=steps, bleed_correction=bleed,
                                progress_cb=progress_cb, lang=L)

            self.result_img        = result
            self._depth_preview    = depth
            self._blur_map_preview = blur_map

            done_msg = (t("status_done_full", L) if not preview_mode
                        else t("status_done_preview", L))
            self.after(0, self._refresh_preview)
            self.after(0, self._status, done_msg)
            self.after(0, self._reset_progress, "")

        except Exception as e:
            self.after(0, self._status, f"Erreur : {e}")
            self.after(0, self._reset_progress, "")
        finally:
            self._computing = False

    # =========================================================================
    # Affichage
    # =========================================================================

    def _refresh_preview(self, _=None):
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
        self._show_array(arr)

    def _show_array(self, arr: np.ndarray):
        h, w  = arr.shape[:2]
        cw    = self.canvas.winfo_width()  or 900
        ch    = self.canvas.winfo_height() or 600
        scale = min(cw / w, ch / h, 1.0)
        nw    = max(int(w * scale), 1)
        nh    = max(int(h * scale), 1)
        pil   = Image.fromarray(arr.astype(np.uint8)).resize((nw, nh), Image.BILINEAR)
        self._tk_img = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER,
                                  image=self._tk_img)

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
        self._status(t("status_exported", L) + path.split("/")[-1])

    def _status(self, msg: str):
        self.lbl_status.config(text=msg)


# =============================================================================

if __name__ == "__main__":
    app = DofApp()
    app.geometry("1400x840")
    app.mainloop()
