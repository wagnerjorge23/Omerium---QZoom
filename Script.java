"""
Contador de Peças em Feixe de Aço
Suporta: tubo retangular, tubo quadrado, ferro chato, cantoneira
Requer: pip install opencv-python pillow numpy
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import csv
import os
from datetime import datetime

# ──────────────────────────────────────────────
# CONFIGURAÇÕES
# ──────────────────────────────────────────────
CSV_PATH = "historico_contagens.csv"
CANVAS_MAX_W = 900
CANVAS_MAX_H = 620

CORES = {
    "auto":   (0, 200, 255),   # amarelo-laranja = detectado automaticamente
    "manual": (0, 255, 80),    # verde = adicionado manualmente
    "remove": (0, 0, 220),     # vermelho = marcado para remoção
}


# ──────────────────────────────────────────────
# DETECÇÃO DE CONTORNOS
# ──────────────────────────────────────────────
def detectar_pecas(img_bgr, min_area, max_area, sensibilidade):
    """
    Detecta contornos poligonais (retângulos, quadrados, L's) na imagem.
    Retorna lista de (cx, cy, contorno, area).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Equalização adaptativa para lidar com variações de iluminação
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Blur para reduzir ruído
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny com threshold ajustável pela sensibilidade (0-100)
    t_low  = max(5,  int(30  - sensibilidade * 0.25))
    t_high = max(20, int(120 - sensibilidade * 0.8))
    edges = cv2.Canny(blur, t_low, t_high)

    # Dilatação para fechar bordas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges,  kernel, iterations=1)

    contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    resultados = []
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if not (min_area <= area <= max_area):
            continue

        # Aproximar contorno para polígono
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

        # Aceitar formas com 3-8 lados (quadrado, retângulo, L, trapézio)
        if len(approx) < 3 or len(approx) > 8:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        resultados.append((cx, cy, cnt, area))

    # Remover duplicatas muito próximas (distância mínima = 15px)
    filtrados = []
    for r in resultados:
        muito_proximo = False
        for f in filtrados:
            dx, dy = r[0] - f[0], r[1] - f[1]
            if (dx*dx + dy*dy) < 15**2:
                muito_proximo = True
                break
        if not muito_proximo:
            filtrados.append(r)

    return filtrados


# ──────────────────────────────────────────────
# APLICAÇÃO PRINCIPAL
# ──────────────────────────────────────────────
class AppContador:
    def __init__(self, root):
        self.root = root
        self.root.title("Contador de Peças em Feixe de Aço")
        self.root.configure(bg="#1e1e2e")
        self.root.resizable(True, True)

        # Estado
        self.img_orig   = None   # BGR original
        self.img_path   = ""
        self.scale      = 1.0
        self.pecas_auto = []     # [(cx, cy, cnt, area)]
        self.pecas_man  = []     # [(cx, cy)] adicionadas manualmente
        self.removidos  = set()  # índices de pecas_auto removidas
        self.zoom       = 1.0

        self._build_ui()
        self._garantir_csv()

    # ── UI ──────────────────────────────────────
    def _build_ui(self):
        # Toolbar superior
        bar = tk.Frame(self.root, bg="#2a2a3e", pady=6)
        bar.pack(fill="x")

        btn_style = {"bg": "#4e9af1", "fg": "white", "font": ("Segoe UI", 10, "bold"),
                     "relief": "flat", "padx": 12, "pady": 4, "cursor": "hand2"}

        tk.Button(bar, text="📂  Abrir Imagem",  command=self.abrir_imagem,  **btn_style).pack(side="left", padx=6)
        tk.Button(bar, text="🔍  Detectar",      command=self.detectar,      **btn_style).pack(side="left", padx=2)
        tk.Button(bar, text="🔄  Limpar Manual", command=self.limpar_manual, **btn_style).pack(side="left", padx=2)
        tk.Button(bar, text="💾  Salvar CSV",    command=self.salvar_csv,    **btn_style).pack(side="left", padx=2)
        tk.Button(bar, text="📋  Ver Histórico", command=self.ver_historico, **btn_style).pack(side="left", padx=2)

        # Painel de controles
        ctrl = tk.Frame(self.root, bg="#2a2a3e", pady=4)
        ctrl.pack(fill="x")

        lbl = {"bg": "#2a2a3e", "fg": "#aaa", "font": ("Segoe UI", 9)}

        tk.Label(ctrl, text="Sensibilidade:", **lbl).pack(side="left", padx=(10,2))
        self.sl_sens = tk.Scale(ctrl, from_=0, to=100, orient="horizontal",
                                length=130, bg="#2a2a3e", fg="white",
                                highlightthickness=0, troughcolor="#444",
                                command=lambda e: self.detectar())
        self.sl_sens.set(50)
        self.sl_sens.pack(side="left")

        tk.Label(ctrl, text="  Área mín (px²):", **lbl).pack(side="left", padx=(10,2))
        self.sv_min = tk.StringVar(value="400")
        tk.Entry(ctrl, textvariable=self.sv_min, width=6,
                 bg="#3a3a5e", fg="white", insertbackground="white").pack(side="left")

        tk.Label(ctrl, text="  Área máx (px²):", **lbl).pack(side="left", padx=(6,2))
        self.sv_max = tk.StringVar(value="80000")
        tk.Entry(ctrl, textvariable=self.sv_max, width=8,
                 bg="#3a3a5e", fg="white", insertbackground="white").pack(side="left")

        tk.Label(ctrl, text="  Modo clique:", **lbl).pack(side="left", padx=(14,2))
        self.modo_clique = tk.StringVar(value="adicionar")
        tk.Radiobutton(ctrl, text="➕ Adicionar", variable=self.modo_clique,
                       value="adicionar", bg="#2a2a3e", fg="#7ef090",
                       selectcolor="#2a2a3e", activebackground="#2a2a3e").pack(side="left")
        tk.Radiobutton(ctrl, text="➖ Remover", variable=self.modo_clique,
                       value="remover", bg="#2a2a3e", fg="#f07e7e",
                       selectcolor="#2a2a3e", activebackground="#2a2a3e").pack(side="left")

        # Canvas + scrollbars
        frame_canvas = tk.Frame(self.root, bg="#1e1e2e")
        frame_canvas.pack(fill="both", expand=True, padx=6, pady=6)

        self.canvas = tk.Canvas(frame_canvas, bg="#111122",
                                cursor="crosshair", highlightthickness=0)
        vsb = tk.Scrollbar(frame_canvas, orient="vertical",   command=self.canvas.yview)
        hsb = tk.Scrollbar(frame_canvas, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<Button-1>", self.click_canvas)
        self.canvas.bind("<MouseWheel>",       self._zoom_wheel)
        self.canvas.bind("<Button-4>",         self._zoom_wheel)
        self.canvas.bind("<Button-5>",         self._zoom_wheel)

        # Status bar
        sb = tk.Frame(self.root, bg="#2a2a3e", pady=4)
        sb.pack(fill="x")

        self.lbl_total = tk.Label(sb, text="Total: 0 peças",
                                  bg="#2a2a3e", fg="#4ef1a0",
                                  font=("Segoe UI", 13, "bold"))
        self.lbl_total.pack(side="left", padx=12)

        self.lbl_info = tk.Label(sb, text="Abra uma imagem para começar.",
                                 bg="#2a2a3e", fg="#888",
                                 font=("Segoe UI", 9))
        self.lbl_info.pack(side="left", padx=10)

        tk.Label(sb, text="Scroll = zoom  |  Clique = adicionar/remover marcação",
                 bg="#2a2a3e", fg="#555", font=("Segoe UI", 8)).pack(side="right", padx=10)

    # ── Abrir imagem ─────────────────────────────
    def abrir_imagem(self):
        path = filedialog.askopenfilename(
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")])
        if not path:
            return
        self.img_path  = path
        self.img_orig  = cv2.imread(path)
        self.pecas_auto = []
        self.pecas_man  = []
        self.removidos  = set()
        self.zoom       = 1.0
        self.lbl_info.config(text=os.path.basename(path))
        self.detectar()

    # ── Detecção ─────────────────────────────────
    def detectar(self):
        if self.img_orig is None:
            return
        try:
            min_a = int(self.sv_min.get())
            max_a = int(self.sv_max.get())
        except ValueError:
            return

        sens = self.sl_sens.get()
        self.pecas_auto = detectar_pecas(self.img_orig, min_a, max_a, sens)
        self.removidos  = set()
        self._renderizar()

    # ── Renderizar imagem no canvas ───────────────
    def _renderizar(self):
        if self.img_orig is None:
            return

        img = self.img_orig.copy()
        h, w = img.shape[:2]

        # Desenhar contornos automáticos
        for i, (cx, cy, cnt, area) in enumerate(self.pecas_auto):
            if i in self.removidos:
                continue
            cor = CORES["auto"]
            cv2.drawContours(img, [cnt], -1, cor, 2)
            cv2.circle(img, (cx, cy), 6, cor, -1)
            cv2.putText(img, str(i+1), (cx-8, cy+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 3)
            cv2.putText(img, str(i+1), (cx-8, cy+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, cor, 1)

        # Desenhar pontos manuais
        base = len(self.pecas_auto) - len(self.removidos)
        for j, (mx, my) in enumerate(self.pecas_man):
            cor = CORES["manual"]
            cv2.circle(img, (mx, my), 14, cor, 2)
            cv2.circle(img, (mx, my),  5, cor, -1)
            num = base + j + 1
            cv2.putText(img, str(num), (mx-8, my+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 3)
            cv2.putText(img, str(num), (mx-8, my+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, cor, 1)

        # Escalar para o zoom atual
        nw = max(1, int(w * self.zoom))
        nh = max(1, int(h * self.zoom))
        img_res = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        self.scale = self.zoom

        # Converter para Tkinter
        img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        self._tk_img = ImageTk.PhotoImage(pil_img)

        self.canvas.config(scrollregion=(0, 0, nw, nh))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_img)

        total = (len(self.pecas_auto) - len(self.removidos)) + len(self.pecas_man)
        self.lbl_total.config(text=f"Total: {total} peças")

    # ── Clique no canvas ──────────────────────────
    def click_canvas(self, event):
        if self.img_orig is None:
            return

        # Coordenadas reais considerando scroll e zoom
        cx = int(self.canvas.canvasx(event.x) / self.scale)
        cy = int(self.canvas.canvasy(event.y) / self.scale)

        if self.modo_clique.get() == "adicionar":
            self.pecas_man.append((cx, cy))
        else:
            # Tentar remover peca_auto mais próxima
            raio = 25
            removeu = False
            for i, (px, py, cnt, area) in enumerate(self.pecas_auto):
                if i in self.removidos:
                    continue
                if abs(px-cx) < raio and abs(py-cy) < raio:
                    self.removidos.add(i)
                    removeu = True
                    break
            if not removeu:
                # Tentar remover peca manual
                for j, (px, py) in enumerate(self.pecas_man):
                    if abs(px-cx) < raio and abs(py-cy) < raio:
                        self.pecas_man.pop(j)
                        break

        self._renderizar()

    # ── Zoom ─────────────────────────────────────
    def _zoom_wheel(self, event):
        fator = 1.1 if (event.delta > 0 or event.num == 4) else 0.9
        self.zoom = max(0.2, min(5.0, self.zoom * fator))
        self._renderizar()

    # ── Limpar marcações manuais ──────────────────
    def limpar_manual(self):
        self.pecas_man = []
        self.removidos = set()
        self._renderizar()

    # ── CSV ──────────────────────────────────────
    def _garantir_csv(self):
        if not os.path.exists(CSV_PATH):
            with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["Data/Hora", "Arquivo", "Auto Detectadas",
                     "Removidas Manual", "Adicionadas Manual", "Total Final"])

    def salvar_csv(self):
        if self.img_orig is None:
            messagebox.showwarning("Aviso", "Nenhuma imagem carregada.")
            return

        auto  = len(self.pecas_auto)
        rem   = len(self.removidos)
        man   = len(self.pecas_man)
        total = (auto - rem) + man

        agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        nome  = os.path.basename(self.img_path)

        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([agora, nome, auto, rem, man, total])

        messagebox.showinfo("Salvo!", f"✅ Registro salvo!\n\nTotal: {total} peças\nArquivo: {CSV_PATH}")

    # ── Ver histórico ─────────────────────────────
    def ver_historico(self):
        if not os.path.exists(CSV_PATH):
            messagebox.showinfo("Histórico", "Nenhum registro ainda.")
            return

        win = tk.Toplevel(self.root)
        win.title("Histórico de Contagens")
        win.configure(bg="#1e1e2e")
        win.geometry("820x400")

        cols = ["Data/Hora", "Arquivo", "Auto", "Removidas", "Manuais", "Total"]
        tree = ttk.Treeview(win, columns=cols, show="headings")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview",
                        background="#2a2a3e", foreground="white",
                        fieldbackground="#2a2a3e", rowheight=24)
        style.configure("Treeview.Heading", background="#3a3a5e", foreground="#4ef1a0")

        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=120 if c != "Arquivo" else 220, anchor="center")

        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # pular cabeçalho
            for row in reader:
                tree.insert("", "end", values=row)

        sb = tk.Scrollbar(win, command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        tree.pack(fill="both", expand=True, padx=6, pady=6)

        tk.Button(win, text="Fechar", command=win.destroy,
                  bg="#4e9af1", fg="white", relief="flat",
                  font=("Segoe UI", 10), padx=10, pady=4).pack(pady=6)


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1100x780")
    app = AppContador(root)
    root.mainloop()
