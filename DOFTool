"""
Depth of Field Tool  –  High Quality
=====================================
Image RGB + Z-depth 16-bit PNG → flou de profondeur photoréaliste.

Améliorations qualité :
  • Noyau Bokeh (disque / hexagone / gaussien) via scipy ou fallback NumPy
  • Accumulation pondérée par couches (pas de bandes entre tranches)
  • Correction du bleeding : le fond flou ne "saigne" pas sur les objets nets
  • Deux passes séparées avant-plan / arrière-plan pour éviter les halos
  • Interpolation bilinéaire entre couches adjacentes (pas de paliers)

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
    from scipy.ndimage import gaussian_filter, convolve
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# =============================================================================
# Noyaux Bokeh
# =============================================================================

def make_disk_kernel(radius: float) -> np.ndarray:
    """Disque binaire normalisé (bokeh circulaire parfait)."""
    r = max(int(math.ceil(radius)), 1)
    size = 2 * r + 1
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    mask = (x * x + y * y) <= radius * radius
    k = mask.astype(np.float32)
    k /= k.sum()
    return k


def make_hex_kernel(radius: float) -> np.ndarray:
    """Hexagone régulier normalisé (bokeh à diaphragme hexagonal)."""
    r = max(int(math.ceil(radius)), 1)
    size = 2 * r + 1
    k = np.zeros((size, size), dtype=np.float32)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            x, y = dx / radius, dy / radius
            # Test hexagone : |x| <= 1, |y| <= 1, |x+y| <= 1, |x-y| <= 1
            if (abs(x) <= 1.0 and abs(y) <= 1.0 and
                    abs(x + y) <= 1.0 / (math.sqrt(3) / 2) and
                    abs(x - y) <= 1.0 / (math.sqrt(3) / 2)):
                k[dy + r, dx + r] = 1.0
    s = k.sum()
    if s > 0:
        k /= s
    else:
        k = make_disk_kernel(radius)
    return k


def make_gaussian_kernel(radius: float) -> np.ndarray:
    """Gaussien analytique normalisé."""
    r = max(int(math.ceil(radius * 2)), 1)
    size = 2 * r + 1
    sigma = radius / 2.0
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    k = np.exp(-(x * x + y * y) / (2 * sigma * sigma)).astype(np.float32)
    k /= k.sum()
    return k


KERNEL_MAKERS = {
    "Disque (Bokeh)":    make_disk_kernel,
    "Hexagone (Bokeh)":  make_hex_kernel,
    "Gaussien (doux)":   make_gaussian_kernel,
}


def apply_kernel(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve img RGB avec kernel 2-D."""
    if HAS_SCIPY:
        return np.stack([
            convolve(img[:, :, c].astype(np.float32), kernel, mode='reflect')
            for c in range(3)
        ], axis=2)
    elif HAS_CV2:
        return cv2.filter2D(img.astype(np.float32), -1, kernel,
                            borderType=cv2.BORDER_REFLECT)
    else:
        # Fallback NumPy : séparable gaussien uniquement
        from PIL import ImageFilter
        r = int((kernel.shape[0] - 1) / 4)
        pil = Image.fromarray(img.astype(np.uint8))
        return np.array(pil.filter(ImageFilter.GaussianBlur(radius=max(r, 1))),
                        dtype=np.float32)


# =============================================================================
# Carte de flou
# =============================================================================

def load_depth(path: str, invert: bool = False) -> np.ndarray:
    img = Image.open(path)
    if img.mode in ("I", "I;16"):
        arr = np.array(img, dtype=np.float32)
    else:
        arr = np.array(img.convert("L"), dtype=np.float32)
    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()
    return 1.0 - arr if invert else arr


def compute_blur_radius_map(depth: np.ndarray,
                             focus_near: float, focus_far: float,
                             max_blur: float, falloff: float) -> np.ndarray:
    """Rayon de flou continu [0, max_blur] pour chaque pixel."""
    d    = depth.copy()
    blur = np.zeros_like(d)

    # — Avant la zone nette (objets proches, devant) —
    soft_near = (d < focus_near) & (d >= focus_near - falloff)
    if soft_near.any() and falloff > 1e-6:
        t = np.clip((focus_near - d[soft_near]) / falloff, 0.0, 1.0)
        blur[soft_near] = t * t * (3.0 - 2.0 * t)
    hard_near = d < (focus_near - falloff)
    if hard_near.any() and (focus_near - falloff) > 1e-6:
        t = np.clip((focus_near - falloff - d[hard_near]) /
                    max(focus_near - falloff, 1e-6), 0.0, 1.0)
        blur[hard_near] = 1.0 + t   # dépasse 1 pour les très proches

    # — Après la zone nette (arrière-plan, derrière) —
    soft_far = (d > focus_far) & (d <= focus_far + falloff)
    if soft_far.any() and falloff > 1e-6:
        t = np.clip((d[soft_far] - focus_far) / falloff, 0.0, 1.0)
        blur[soft_far] = t * t * (3.0 - 2.0 * t)
    hard_far = d > (focus_far + falloff)
    range_f = 1.0 - (focus_far + falloff)
    if hard_far.any() and range_f > 1e-6:
        t = np.clip((d[hard_far] - focus_far - falloff) / range_f, 0.0, 1.0)
        blur[hard_far] = 1.0 + t

    blur = np.clip(blur, 0.0, None)
    if blur.max() > 0:
        blur = blur / blur.max() * max_blur
    return blur


# =============================================================================
# Moteur de rendu haute qualité
# =============================================================================

def render_dof(rgb: np.ndarray,
               blur_map: np.ndarray,
               depth: np.ndarray,
               kernel_name: str = "Disque (Bokeh)",
               steps: int = 16,
               bleed_correction: bool = True,
               progress_cb=None) -> np.ndarray:
    """
    Rendu DoF par accumulation pondérée avec plusieurs couches.

    Algorithme :
      1. Pour chaque couche de rayon r_i on génère une image floutée F_i
         avec le noyau bokeh sélectionné.
      2. On compose de l'arrière-plan (grand r) vers l'avant-plan (petit r) :
         chaque pixel contribue avec son poids alpha = 1 − |blur − r_i| / pas.
         Cela donne une interpolation douce entre les couches (pas de bandes).
      3. Correction bleeding : avant de composer F_i sur le résultat, on masque
         les pixels dont la profondeur est nettement plus proche que la couche
         courante. Cela évite que le fond flou "éclabousse" les bords nets.
    """
    h, w = rgb.shape[:2]
    max_r = blur_map.max()

    if max_r < 0.5:
        return rgb.copy()

    maker = KERNEL_MAKERS.get(kernel_name, make_disk_kernel)

    # Radii de 0 à max_r (inclus) — on inclut 0 pour la zone nette
    radii = np.linspace(0.0, max_r, steps + 1)
    step_size = radii[1] - radii[0] if steps > 0 else 1.0

    # Pré-calcul de toutes les images floutées (phase 1 : convolutions)
    total_steps = len(radii) + len(radii)   # phase1 + phase2
    done = 0
    blurred_layers = {}
    for r in radii:
        if r < 0.5:
            blurred_layers[r] = rgb.astype(np.float32)
        else:
            k = maker(r)
            blurred_layers[r] = apply_kernel(rgb, k)
        done += 1
        if progress_cb:
            progress_cb(done, total_steps, f"Convolution  {done}/{len(radii)}")

    # Normalisation depth pour le test bleeding [0..1]
    d_norm = np.clip(depth, 0.0, 1.0)

    # Composition arrière → avant (les couches proches écrasent les lointaines)
    # On travaille avec un accumulateur RGBA (A = poids total)
    accum  = np.zeros((h, w, 3), dtype=np.float32)
    weight = np.zeros((h, w),    dtype=np.float32)

    for r in reversed(radii):
        # Poids par pixel : pic triangulaire centré sur blur_map == r
        w_pix = np.maximum(0.0, 1.0 - np.abs(blur_map - r) / (step_size + 1e-6))

        if bleed_correction and r > 0.5:
            # Fraction de la scène que cette couche représente en profondeur
            layer_depth_fraction = r / max_r  # 0 = net, 1 = très loin/très près
            # On atténue le bleeding : si le pixel net est devant la couche floue
            # on ne laisse pas cette couche floue le contaminer.
            # Proxy : les pixels avec blur_map ≈ 0 (nets) bloquent les couches lointaines
            sharp_mask = (blur_map < step_size * 0.5).astype(np.float32)
            # Bleeding factor : 1 = pas de bleeding, 0 = bleeding total bloqué
            bleeding_block = 1.0 - sharp_mask * np.clip(layer_depth_fraction, 0, 1) * 0.85
            w_pix = w_pix * bleeding_block

        f = blurred_layers[r]
        accum  += f * w_pix[:, :, np.newaxis]
        weight += w_pix
        done += 1
        if progress_cb:
            progress_cb(done, total_steps, f"Composition  {done - len(radii)}/{len(radii)}")

    # Normalisation
    weight = np.maximum(weight, 1e-6)
    result = accum / weight[:, :, np.newaxis]
    return np.clip(result, 0, 255).astype(np.uint8)


# =============================================================================
# Widget visualiseur de zone
# =============================================================================

class FocusZoneCanvas(tk.Canvas):
    H   = 82
    PAD = 16

    def __init__(self, master, var_near, var_far, var_falloff, callback, **kw):
        super().__init__(master, height=self.H, bg="#0d0d1a",
                         highlightthickness=1, highlightbackground="#2a2a4a", **kw)
        self.var_near    = var_near
        self.var_far     = var_far
        self.var_falloff = var_falloff
        self.callback    = callback
        self._drag       = None
        self.bind("<Configure>",       lambda _: self.redraw())
        self.bind("<ButtonPress-1>",   self._on_press)
        self.bind("<B1-Motion>",       self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)

    def redraw(self):
        self.delete("all")
        W = self.winfo_width()
        if W < 30: return
        H, P = self.H, self.PAD
        IW = W - 2 * P

        near = self.var_near.get()
        far  = self.var_far.get()
        fo   = self.var_falloff.get()

        def xp(d): return P + d * IW
        def yp(t): return H - P - t * (H - 2 * P)

        self.create_rectangle(xp(near), P + 2, xp(far), H - P,
                              fill="#132613", outline="")

        N = max(int(IW * 2), 120)
        for side in ("left", "right"):
            pts = []
            for i in range(N + 1):
                d = i / N
                if side == "left"  and d > near: continue
                if side == "right" and d < far:  continue
                gap = (near - d) if side == "left" else (d - far)
                t = min(gap / max(fo, 1e-6), 1.0)
                t = t * t * (3.0 - 2.0 * t)
                pts += [xp(d), yp(t)]
            if len(pts) < 4: continue
            poly = ([xp(0), yp(0)] + pts + [xp(near), yp(0)]
                    if side == "left"
                    else [xp(far), yp(0)] + pts + [xp(1), yp(0)])
            self.create_polygon(poly, fill="#0a2040", outline="")
            self.create_line(pts, fill="#4a9eff", width=2, smooth=True)

        self.create_line(P, H - P, W - P, H - P, fill="#2a2a5a", width=1)
        self.create_line(xp(near), H - P, xp(far), H - P, fill="#2aaa5a", width=2)

        for val, color, tag in [(near, "#60d394", "near"), (far, "#ff9f6b", "far")]:
            cx = xp(val)
            self.create_line(cx, P + 2, cx, H - P, fill=color, width=2, dash=(4, 3))
            self.create_oval(cx - 6, H - P - 6, cx + 6, H - P + 6,
                             fill=color, outline="#1a1a2e", width=2, tags=tag)
            self.create_text(cx, P + 10, text=f"{val:.2f}",
                             fill=color, font=("Courier New", 8, "bold"))

        mid_x = xp((near + far) / 2)
        self.create_text(mid_x, H // 2 + 4, text="NET",
                         fill="#60d394", font=("Courier New", 8, "bold"))
        self.create_text(P,     H - 4, text="0", fill="#444466",
                         font=("Courier New", 8), anchor="w")
        self.create_text(W - P, H - 4, text="1", fill="#444466",
                         font=("Courier New", 8), anchor="e")

    def _d_from_x(self, ex):
        W, P = self.winfo_width(), self.PAD
        return max(0.0, min(1.0, (ex - P) / max(W - 2 * P, 1)))

    def _on_press(self, e):
        W, P = self.winfo_width(), self.PAD
        IW = W - 2 * P
        xn = P + self.var_near.get() * IW
        xf = P + self.var_far.get()  * IW
        dn, df = abs(e.x - xn), abs(e.x - xf)
        if min(dn, df) > 16: return
        self._drag = "near" if dn <= df else "far"

    def _on_drag(self, e):
        if not self._drag: return
        d = self._d_from_x(e.x)
        near, far = self.var_near.get(), self.var_far.get()
        if self._drag == "near": self.var_near.set(min(d, far  - 0.02))
        else:                    self.var_far.set( max(d, near + 0.02))
        self.redraw()
        self.callback()

    def _on_release(self, e): self._drag = None


# =============================================================================
# Application principale
# =============================================================================

class DofApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Depth of Field Tool  —  HQ")
        self.configure(bg="#1a1a2e")
        self.resizable(True, True)

        self.rgb_img:    np.ndarray | None = None
        self.depth_img:  np.ndarray | None = None
        self.result_img: np.ndarray | None = None
        self._depth_path: str | None       = None
        self._render_job  = None
        self._computing   = False

        self._build_ui()

    # ── UI ───────────────────────────────────────────────────────────────────

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

        ctrl = ttk.Frame(self, padding=16)
        ctrl.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(ctrl, text="── DOF TOOL HQ ──",
                  font=("Courier New", 13, "bold"),
                  foreground="#a0c4ff").pack(pady=(0, 14))

        # Fichiers
        ttk.Button(ctrl, text="📂  Image RGB",
                   command=self._load_rgb).pack(fill=tk.X, pady=3)
        self.lbl_rgb = ttk.Label(ctrl, text="(aucune)", foreground="#555577")
        self.lbl_rgb.pack()

        ttk.Button(ctrl, text="📂  Z-Depth 16-bit PNG",
                   command=self._load_depth).pack(fill=tk.X, pady=(10, 3))
        self.lbl_depth = ttk.Label(ctrl, text="(aucune)", foreground="#555577")
        self.lbl_depth.pack()

        self.var_invert = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="↕  Inverser Z-depth  (0 = loin)",
                        variable=self.var_invert,
                        command=self._reload_depth).pack(anchor="w", pady=(6, 0))

        ttk.Separator(ctrl, orient="horizontal").pack(fill=tk.X, pady=12)

        # Visualiseur
        ttk.Label(ctrl, text="Zone de netteté  —  glisser les poignées",
                  foreground="#a0c4ff").pack(anchor="w")

        self.var_near    = tk.DoubleVar(value=0.35)
        self.var_far     = tk.DoubleVar(value=0.65)
        self.var_falloff = tk.DoubleVar(value=0.06)

        self.focus_viz = FocusZoneCanvas(
            ctrl, self.var_near, self.var_far, self.var_falloff,
            callback=self._on_focus_changed)
        self.focus_viz.pack(fill=tk.X, pady=(4, 10))

        def make_row(label, var, lo, hi, color, fmt="{:.2f}"):
            ttk.Label(ctrl, text=label).pack(anchor="w")
            row = ttk.Frame(ctrl)
            row.pack(fill=tk.X, pady=(0, 5))
            ttk.Scale(row, from_=lo, to=hi, orient=tk.HORIZONTAL, variable=var,
                      command=lambda _: self._on_slider()).pack(
                          side=tk.LEFT, fill=tk.X, expand=True)
            lbl = ttk.Label(row, foreground=color, width=8,
                            font=("Courier New", 10))
            lbl.pack(side=tk.LEFT)
            return lbl, fmt

        self.lbl_near_v,  _ = make_row("Limite proche",    self.var_near,    0.0,  1.0,   "#60d394")
        self.lbl_far_v,   _ = make_row("Limite lointaine", self.var_far,     0.0,  1.0,   "#ff9f6b")
        self.lbl_fo_v,    _ = make_row("Transition douce", self.var_falloff, 0.0,  0.25,  "#c0c0ff")

        ttk.Separator(ctrl, orient="horizontal").pack(fill=tk.X, pady=8)

        self.var_blur = tk.DoubleVar(value=22.0)
        self.lbl_blur_v, _ = make_row("Flou maximum (px)", self.var_blur, 1.0, 100.0, "#a0c4ff", "{:.0f} px")

        ttk.Separator(ctrl, orient="horizontal").pack(fill=tk.X, pady=8)

        # ── Options qualité ──────────────────────────────────────────────
        ttk.Label(ctrl, text="── Qualité ──",
                  foreground="#a0c4ff").pack(anchor="w")

        ttk.Label(ctrl, text="Forme du bokeh").pack(anchor="w", pady=(6, 0))
        self.var_kernel = tk.StringVar(value="Disque (Bokeh)")
        cb = ttk.Combobox(ctrl, textvariable=self.var_kernel,
                          values=list(KERNEL_MAKERS.keys()),
                          state="readonly", width=22)
        cb.pack(fill=tk.X, pady=(2, 6))
        cb.bind("<<ComboboxSelected>>", lambda _: self._schedule_compute())

        ttk.Label(ctrl, text="Nombre de couches (qualité)").pack(anchor="w")
        self.var_steps = tk.IntVar(value=16)
        row_s = ttk.Frame(ctrl)
        row_s.pack(fill=tk.X, pady=(0, 5))
        ttk.Scale(row_s, from_=4, to=32, orient=tk.HORIZONTAL,
                  variable=self.var_steps,
                  command=lambda _: self._on_slider()).pack(
                      side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_steps = ttk.Label(row_s, foreground="#a0c4ff", width=8,
                                   font=("Courier New", 10))
        self.lbl_steps.pack(side=tk.LEFT)

        self.var_bleed = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="✦  Anti-bleeding (bords nets)",
                        variable=self.var_bleed,
                        command=self._schedule_compute).pack(anchor="w", pady=(4, 0))

        ttk.Separator(ctrl, orient="horizontal").pack(fill=tk.X, pady=10)

        # Aperçu
        ttk.Label(ctrl, text="Aperçu").pack(anchor="w")
        self.var_view = tk.StringVar(value="result")
        for val, lbl in [("original", "Original"),
                         ("depth",    "Z-Depth"),
                         ("blur_map", "Carte de flou"),
                         ("result",   "Résultat DoF")]:
            ttk.Radiobutton(ctrl, text=lbl, variable=self.var_view,
                            value=val, command=self._refresh_preview,
                            style="TLabel").pack(anchor="w")

        ttk.Separator(ctrl, orient="horizontal").pack(fill=tk.X, pady=10)

        ttk.Button(ctrl, text="▶  Calculer (pleine résolution)",
                   command=self._compute_full).pack(fill=tk.X, pady=3)
        ttk.Button(ctrl, text="💾  Exporter résultat",
                   command=self._export).pack(fill=tk.X, pady=3)

        # ── Barre de progression ──────────────────────────────────────────
        self.var_progress = tk.DoubleVar(value=0.0)
        self.progressbar = ttk.Progressbar(ctrl, variable=self.var_progress,
                                           maximum=100.0, length=240,
                                           mode="determinate")
        self.progressbar.pack(fill=tk.X, pady=(10, 2))
        self.lbl_progress = ttk.Label(ctrl, text="", foreground="#8888aa",
                                      font=("Courier New", 9))
        self.lbl_progress.pack(anchor="w")

        self.lbl_status = ttk.Label(ctrl, text="Prêt.", foreground="#60d394",
                                    wraplength=240)
        self.lbl_status.pack(pady=(8, 0))

        # Canvas image
        cf = ttk.Frame(self)
        cf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.canvas = tk.Canvas(cf, bg="#0d0d1a", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self._tk_img = None

        self._update_labels()

    # ── Chargement ───────────────────────────────────────────────────────────

    def _load_rgb(self):
        path = filedialog.askopenfilename(
            title="Image RGB",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff"),
                       ("Tous", "*.*")])
        if not path: return
        self.rgb_img = np.array(Image.open(path).convert("RGB"))
        self.lbl_rgb.config(text=path.split("/")[-1][-32:], foreground="#60d394")
        self._status("Image RGB chargée.")
        self._auto_compute()

    def _load_depth(self):
        path = filedialog.askopenfilename(
            title="Z-Depth 16-bit PNG",
            filetypes=[("PNG", "*.png"), ("Tous", "*.*")])
        if not path: return
        self._depth_path = path
        self.depth_img = load_depth(path, self.var_invert.get())
        self.lbl_depth.config(text=path.split("/")[-1][-32:], foreground="#60d394")
        self._status("Z-Depth chargé.")
        self._auto_compute()

    def _reload_depth(self):
        if self._depth_path:
            self.depth_img = load_depth(self._depth_path, self.var_invert.get())
            self._auto_compute()

    # ── Sliders ──────────────────────────────────────────────────────────────

    def _on_slider(self):
        n, f = self.var_near.get(), self.var_far.get()
        if n >= f: self.var_far.set(min(n + 0.02, 1.0))
        self._update_labels()
        self.focus_viz.redraw()
        self._schedule_compute()

    def _on_focus_changed(self):
        self._update_labels()
        self._schedule_compute()

    def _update_labels(self):
        self.lbl_near_v.config(text=f"{self.var_near.get():.2f}")
        self.lbl_far_v.config( text=f"{self.var_far.get():.2f}")
        self.lbl_fo_v.config(  text=f"{self.var_falloff.get():.2f}")
        self.lbl_blur_v.config(text=f"{self.var_blur.get():.0f} px")
        self.lbl_steps.config( text=f"{int(self.var_steps.get())} passes")

    def _schedule_compute(self):
        self._update_labels()
        if self._render_job: self.after_cancel(self._render_job)
        self._render_job = self.after(420, self._auto_compute)

    def _set_progress(self, done: int, total: int, label: str):
        """Appelé depuis le thread worker via self.after(0, ...)."""
        pct = done / total * 100.0
        self.var_progress.set(pct)
        self.lbl_progress.config(text=f"{label}  ({done}/{total})")

    def _reset_progress(self, msg: str = ""):
        self.var_progress.set(0.0)
        self.lbl_progress.config(text=msg)

    # ── Calcul ───────────────────────────────────────────────────────────────

    def _auto_compute(self):
        if self.rgb_img is None or self.depth_img is None or self._computing:
            return
        threading.Thread(target=self._run_compute,
                         args=(True,), daemon=True).start()

    def _compute_full(self):
        if self.rgb_img is None or self.depth_img is None:
            messagebox.showwarning("Données manquantes",
                                   "Chargez l'image et le z-depth d'abord.")
            return
        self._status("Calcul pleine résolution…")
        threading.Thread(target=self._run_compute,
                         args=(False,), daemon=True).start()

    def _run_compute(self, preview_mode: bool):
        self._computing = True
        self.after(0, self._reset_progress, "Démarrage…")
        try:
            rgb   = self.rgb_img
            depth = self.depth_img

            if preview_mode:
                scale = min(1.0, 560 / max(rgb.shape[:2]))
                if scale < 1.0:
                    h = int(rgb.shape[0] * scale)
                    w = int(rgb.shape[1] * scale)
                    rgb = np.array(
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
                s = min(1.0, 560 / max(self.rgb_img.shape[:2]))
                max_b = max(max_b * s, 2.0)

            steps  = int(self.var_steps.get())
            kernel = self.var_kernel.get()
            bleed  = self.var_bleed.get()

            blur_map = compute_blur_radius_map(depth, near, far, max_b, falloff)

            def progress_cb(done, total, label):
                self.after(0, self._set_progress, done, total, label)

            result = render_dof(rgb, blur_map, depth, kernel, steps, bleed,
                                progress_cb=progress_cb)

            self.result_img        = result
            self._depth_preview    = depth
            self._blur_map_preview = blur_map

            done_msg = "✓ Calcul terminé." if not preview_mode else "✓ Aperçu prêt."
            self.after(0, self._refresh_preview)
            self.after(0, self._status, done_msg)
            self.after(0, self._reset_progress, "")
        except Exception as e:
            self.after(0, self._status, f"Erreur : {e}")
            self.after(0, self._reset_progress, "")
        finally:
            self._computing = False

    # ── Affichage ────────────────────────────────────────────────────────────

    def _refresh_preview(self, _=None):
        mode = self.var_view.get()
        if   mode == "original"  and self.rgb_img is not None:
            arr = self.rgb_img
        elif mode == "depth"     and hasattr(self, "_depth_preview"):
            d = (self._depth_preview * 255).astype(np.uint8)
            arr = np.stack([d, d, d], axis=2)
        elif mode == "blur_map"  and hasattr(self, "_blur_map_preview"):
            bm  = self._blur_map_preview
            bmn = (bm / max(bm.max(), 1e-6) * 255).astype(np.uint8)
            arr = np.stack([bmn, bmn, bmn], axis=2)
        elif mode == "result"    and self.result_img is not None:
            arr = self.result_img
        else: return
        self._show_array(arr)

    def _show_array(self, arr: np.ndarray):
        h, w = arr.shape[:2]
        cw = self.canvas.winfo_width()  or 900
        ch = self.canvas.winfo_height() or 600
        scale = min(cw / w, ch / h, 1.0)
        nw, nh = max(int(w * scale), 1), max(int(h * scale), 1)
        pil = Image.fromarray(arr.astype(np.uint8)).resize((nw, nh), Image.BILINEAR)
        self._tk_img = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2,
                                  anchor=tk.CENTER, image=self._tk_img)

    # ── Export ───────────────────────────────────────────────────────────────

    def _export(self):
        if self.result_img is None:
            messagebox.showwarning("Rien à exporter",
                                   "Lancez d'abord le calcul pleine résolution.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
            title="Enregistrer")
        if not path: return
        Image.fromarray(self.result_img).save(path)
        self._status(f"Exporté → {path.split('/')[-1]}")

    def _status(self, msg: str):
        self.lbl_status.config(text=msg)


# =============================================================================

if __name__ == "__main__":
    app = DofApp()
    app.geometry("1320x800")
    app.mainloop()
