"""informe/generar_informe_final.py — Informe final (formato WAIMLAp, plantilla del grupo 4).

Uso (desde la raíz del repo):
    python informe/figuras_paper.py          # figuras en inglés (informe/figuras/)
    python informe/generar_informe_final.py  # escribe informe/E5_Grupo4_DINOv2_EuroSAT_LoRA.docx

Toma la plantilla informe/plantilla_E5_WAIMLAp.docx (encabezado, logos, título y autores) y
agrega el contenido bajo cada sección. Todas las cifras salen de results/ (ver DESIGN_LOG.md).
"""
import copy
import re

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

PLANTILLA = "informe/plantilla_E5_WAIMLAp.docx"
SALIDA = "informe/E5_Grupo4_DINOv2_EuroSAT_LoRA.docx"
FIG = "informe/figuras/"
FONT = "Times New Roman"
TINTA = RGBColor(0x0F, 0x17, 0x2A)
ANCHO = 16.9  # cm de texto (A4, márgenes 2.03 cm)

doc = Document(PLANTILLA)
cursor = None


def _mover(el):
    global cursor
    cursor.addnext(el)
    cursor = el
    return el


def _run(p, texto, size=10, bold=False, italic=False, sub=False, sup=False, color=None):
    r = p.add_run(texto)
    r.font.name = FONT
    r._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    r.font.size = Pt(size)
    r.bold, r.italic = bold or None, italic or None
    if sub:
        r.font.subscript = True
    if sup:
        r.font.superscript = True
    if color:
        r.font.color.rgb = color
    return r


TOKENS = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|~[^~]+~|\^[^^]+\^)")


def _texto(p, texto, size=10, **kw):
    for parte in TOKENS.split(texto):
        if not parte:
            continue
        if parte.startswith("**"):
            _run(p, parte[2:-2], size, bold=True, **kw)
        elif parte.startswith("*"):
            _run(p, parte[1:-1], size, italic=True, **kw)
        elif parte.startswith("~"):
            _run(p, parte[1:-1], size, sub=True, **kw)
        elif parte.startswith("^"):
            _run(p, parte[1:-1], size, sup=True, **kw)
        else:
            _run(p, parte, size, **kw)


def P(texto, size=10, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=4, before=0, keep=False, indent=None):
    p = doc.add_paragraph()
    _mover(p._p)
    pf = p.paragraph_format
    pf.alignment, pf.space_after, pf.space_before, pf.line_spacing = align, Pt(after), Pt(before), 1.0
    pf.keep_with_next = keep
    if indent is not None:
        pf.left_indent, pf.first_line_indent = Cm(indent), Cm(-indent)
    _texto(p, texto, size)
    return p


def H2(texto):
    p = doc.add_paragraph()
    _mover(p._p)
    pf = p.paragraph_format
    pf.space_before, pf.space_after, pf.keep_with_next, pf.line_spacing = Pt(6), Pt(2), True, 1.0
    _run(p, texto, 10, bold=True, italic=True, color=TINTA)


nfig = ntab = 0


def FIGURA(archivo, pie, ancho_cm):
    global nfig
    nfig += 1
    p = doc.add_paragraph()
    _mover(p._p)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(6), Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(FIG + archivo, width=Cm(ancho_cm))
    c = doc.add_paragraph()
    _mover(c._p)
    c.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    c.paragraph_format.space_after, c.paragraph_format.line_spacing = Pt(8), 1.0
    _run(c, f"Fig. {nfig}. ", 8.5, bold=True)
    _texto(c, pie, 8.5)


def _bordes(celda, arriba=None, abajo=None):
    tcPr = celda._tc.get_or_add_tcPr()
    b = OxmlElement("w:tcBorders")
    for lado, val in (("top", arriba), ("bottom", abajo), ("left", None), ("right", None)):
        e = OxmlElement(f"w:{lado}")
        if val:
            e.set(qn("w:val"), "single"); e.set(qn("w:sz"), str(val)); e.set(qn("w:color"), "0F172A")
        else:
            e.set(qn("w:val"), "nil")
        b.append(e)
    tcPr.append(b)


def TABLA(titulo, filas, anchos_cm, size=8.5, alinear_izq=(0,)):
    """filas[0] = encabezado. Bordes estilo booktabs."""
    global ntab
    ntab += 1
    c = doc.add_paragraph()
    _mover(c._p)
    c.paragraph_format.space_before, c.paragraph_format.space_after = Pt(6), Pt(3)
    c.paragraph_format.keep_with_next, c.paragraph_format.line_spacing = True, 1.0
    _run(c, f"Table {ntab}. ", 8.5, bold=True)
    _texto(c, titulo, 8.5)
    t = doc.add_table(rows=len(filas), cols=len(filas[0]))
    _mover(t._tbl)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    n = len(filas)
    for i, fila in enumerate(filas):
        tr = t.rows[i]._tr
        trPr = tr.get_or_add_trPr(); cs = OxmlElement("w:cantSplit"); trPr.append(cs)
        for j, txt in enumerate(fila):
            cel = t.cell(i, j)
            cel.width = Cm(anchos_cm[j])
            _bordes(cel, arriba=12 if i == 0 else (6 if i == 1 else None), abajo=12 if i == n - 1 else None)
            p = cel.paragraphs[0]
            p.paragraph_format.space_after, p.paragraph_format.space_before = Pt(1), Pt(1)
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.keep_with_next = i < n - 1
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT if j in alinear_izq else WD_ALIGN_PARAGRAPH.CENTER
            _texto(p, txt, size, ) if i else _run(p, txt, size, bold=True)
    esp = doc.add_paragraph()
    _mover(esp._p)
    esp.paragraph_format.space_after = Pt(4)


def seccion(titulo):
    """Posiciona el cursor en el título de la plantilla y borra el párrafo vacío que le sigue."""
    global cursor
    for p in doc.paragraphs:
        if p.text.strip() == titulo:
            sig = p._p.getnext()
            if sig is not None and sig.tag == qn("w:p") and not "".join(sig.itertext()).strip():
                sig.getparent().remove(sig)
            cursor = p._p
            return
    raise KeyError(titulo)


# =========================================================== 1. INTRODUCTION
seccion("1. INTRODUCTION")
P("Self-supervised vision foundation models such as DINOv2 [1], a Vision Transformer (ViT) [2] pre-trained on 142 million curated natural images, produce general-purpose features that are usually transferred with a frozen backbone and a linear classifier (*linear probing*). Sentinel-2 imagery [3] is a different domain: tiles are seen from above at about 10 m per pixel, are dominated by repetitive textures and contain no central object. Low-Rank Adaptation (LoRA) [4] offers a middle ground between freezing and full fine-tuning: it trains only a low-rank update of a few weight matrices and, once merged, adds no inference cost.")
P("Land-cover classifiers are trained on some regions and deployed on others, where crops, soils and vegetation change. A random split of EuroSAT [5] overestimates this ability, because neighbouring and nearly identical tiles fall in both training and test sets; spatially blocked validation is the recommended remedy for such autocorrelated data [6]. We therefore ask: **does adding LoRA to DINOv2 improve land-cover classification in geographic regions not seen during training, compared with a frozen DINOv2?** We test two hypotheses: **H1**, LoRA improves macro-F1 on a random (internal) split; and **H2**, on held-out regions both models degrade and LoRA’s advantage shrinks, because part of it would come from fitting the training regions.")
P("Our contributions are: (i) an architectural teardown of DINOv2 ViT-S/14 (cost, latency, representations and token-norm artefacts [7]) that motivates where to intervene; (ii) a geographic split of EuroSAT built from the georeferenced EuroSAT MS tiles, with an *in-region* and an *out-of-region* test set for the same model; (iii) a controlled comparison with three seeds, exact McNemar tests and a decision rule fixed before running; and (iv) ablations of rank, depth and projections on both splits, each with a numerical prediction registered beforehand. Code, splits, all runs and the decision log are public at github.com/jhntan2075/dinov2-eurosat-lora.")

# =========================================================== 2. METHODOLOGY
seccion("2. METHODOLOGY")
H2("2.1 Data and splits")
P("We use EuroSAT RGB [5]: 27,000 Sentinel-2 tiles of 64×64 px in 10 land-use and land-cover classes (2,000–3,000 per class). Tiles are resized to 224×224 px with bicubic interpolation, without the default center crop (it would remove 23 % of the scene), and normalized with ImageNet statistics as in DINOv2 pre-training. The **internal split** is a stratified 70/15/15 partition with seed 42 (18,900 / 4,050 / 4,050 images).")
P("The RGB tiles carry no coordinates, but their counterparts in EuroSAT MS (same file name) are georeferenced GeoTIFFs. We verified the 1:1 pairing of all 27,000 tiles, took each tile center in WGS84, projected it to ETRS89-LAEA (EPSG:3035) and grouped tiles with K-means. Whole regions, disjoint from each other, form the validation and the **out-of-region test** set (each 12–18 % of the data, with at least 5 % of every class); among 20,000 random assignments the one closest to 15/15 is kept. A stratified **in-region test** (10 % of the data) is drawn from the remaining regions, and the rest is used for training. A fallback order (K = 12 → 15 → 20, then a 3 % class floor) was fixed before running; K = 12 had no valid assignment, so K = 15 was used. The result is 15,383 training, 4,754 validation, 2,700 in-region and 4,163 out-of-region images (Fig. 1). The held-out regions lie in central Europe (centers near 18.5°E 52.2°N and 16.9°E 48.0°N) and in the eastern Mediterranean (33.0°E 34.9°N). The median distance from an out-of-region image to the nearest training image is 492 km (minimum 12.4 km), against 0.9 km for the in-region test.", keep=False)
FIGURA("fig_particion_geo.png", "Geographic split by spatial blocks (K = 15 K-means regions in EPSG:3035). Grey: training regions, from which the in-region test is sampled; blue: validation regions; orange: held-out regions used as out-of-region test.", 10.0)

H2("2.2 Model and architectural teardown (Phase A)")
P("We use DINOv2 ViT-S/14 without registers (Hugging Face *facebook/dinov2-small*): 12 identical Transformer blocks with d = 384, 6 heads and an MLP of 1,536 units. Table 1 summarizes its measured cost on a Tesla T4; measured FLOPs match the analytical count. A PCA of the patch tokens shows that block 4 mainly encodes patch position, whereas block-12 tokens group by scene class; the effective receptive field [8] is already global at block 1. This motivates adapting the last blocks.")
TABLA("Architectural teardown of DINOv2 ViT-S/14 (Phase A), NVIDIA Tesla T4.", [
    ["Measurement", "Value"],
    ["Total parameters", "22,056,576 (88.2 MB in fp32)"],
    ["Parameters per block (×12)", "1,775,232 (MLP 67 %, attention 33 %)"],
    ["GFLOPs per image at 112 / 224 / 448 px", "2.87 / 12.25 / 63.36"],
    ["Share of quadratic attention in total FLOPs", "2.7 / 9.9 / 30.6 %"],
    ["Latency, fp32, batch 1", "8.89 ms (112 img/s)"],
    ["Throughput, fp32, batch 32", "263 img/s (3.80 ms/img)"],
    ["Throughput, fp16, batch 128", "854 img/s (1.17 ms/img)"],
    ["Peak memory, fp32, batch 1 / 128", "187 / 833 MB"],
    ["Effective receptive field r50, blocks 1 / 4 / 8 / 12", "69 / 57 / 67 / 85 px"],
], [8.0, 6.5])

H2("2.3 Compared conditions and training budget")
P("**Baseline (linear probing).** The backbone is frozen and each image is represented by the concatenation of the [CLS] token and the mean of the patch tokens after the final LayerNorm (768 dimensions), following the DINOv2 linear evaluation protocol [1]; a linear head with 7,690 parameters is trained on cached embeddings. **LoRA.** The query (Q) and value (V) projections of blocks 11 and 12 become W′ = W + (α/r)·B·A, with W frozen, A ∈ ℝ^8×384^ (Kaiming-uniform) and B ∈ ℝ^384×8^ initialized to zero (r = α = 8), so training starts exactly from the frozen model. This adds 24,576 weights (0.11 % of the backbone). After training, B·A is merged into W, so the deployed model has the original size and latency. **Identical budget.** Both conditions use AdamW [9] (learning rate 10^−3^, weight decay 10^−4^), batch 64, 8 epochs, mixed precision on GPU and model selection by the best validation macro-F1; the test sets are never used for any decision.")

H2("2.4 Evaluation protocol and pre-registered decision rule")
P("The main metric is macro-F1, since classes are imbalanced. Every baseline-vs-LoRA comparison is repeated with seeds 42, 0 and 1 (splits do not change with the seed), and significance against test-set noise is assessed with exact McNemar tests [10] on paired predictions. For H2 we define, per seed and condition, the **drop** = macro-F1(in-region) − macro-F1(out-of-region) and LoRA’s **advantage** = macro-F1(LoRA) − macro-F1(baseline). The rule, recorded in the decision log on 24/11/2026 before running Phase C, uses the mean over the three seeds: the H2 prediction *holds* if both drops lie between 3 and 10 percentage points (pp) and the out-of-region advantage is ≤ +0.5 pp; it is *rejected* if any drop is < 1 pp or the out-of-region advantage is ≥ +1.3 pp; any other outcome is reported as intermediate. A direction is claimed only if all three seeds agree in sign.")

H2("2.5 Ablations")
P("With seed 42 and on both splits we vary the rank (r ∈ {4, 16}, α = r), the depth (LoRA on the last 4 blocks, 9–12) and the projections (Q only). Each ablation has a numerical prediction and a rejection criterion registered before running. Because a single seed cannot separate differences below ≈ 0.6 pp from noise, the depth ablation, the only one with a clear effect, was replicated with seeds 0 and 1 after registering a new prediction.")

H2("2.6 Implementation and reproducibility")
P("All runs were trained on Google Colab (NVIDIA Tesla T4, PyTorch 2.11, Transformers 5.20). The code is a tested Python package (43 unit tests on CPU) driven by thin notebooks; every run is idempotent and stores its metrics, per-image predictions and weights, so all tables are regenerated from these files. Seeds, splits and environment are documented in the repository.")

# =========================================================== 3. RESULTS AND DISCUSSION
seccion("3. RESULTS AND DISCUSSION")
H2("3.1 In-distribution performance (H1)")
P("Table 2 and Fig. 2 summarize the three seeds. On the internal split LoRA improves macro-F1 in every seed (+1.19, +1.32 and +2.06 pp; mean +1.52 ± 0.47) with exact McNemar p ≤ 2×10^−4^ in each seed (Table 3). **H1 is confirmed.**")
TABLA("Macro-F1 (%) per test set, mean ± standard deviation over seeds 42, 0 and 1. Δ = LoRA − baseline.", [
    ["Test set", "Linear probing", "LoRA r = 8", "Δ (pp)"],
    ["Internal split (random test)", "95.66 ± 0.19", "97.18 ± 0.39", "+1.52 ± 0.47"],
    ["Geographic split, in-region test", "96.40 ± 0.22", "97.32 ± 0.19", "+0.92 ± 0.36"],
    ["Geographic split, out-of-region test", "89.25 ± 0.88", "91.45 ± 0.57", "**+2.20 ± 1.00**"],
    ["Drop (in-region − out-of-region)", "7.15 ± 0.68", "5.87 ± 0.40", "−1.28"],
], [6.2, 3.2, 3.2, 3.2])
FIGURA("fig_macro_f1_conjuntos.png", "Macro-F1 per test set. Dots: seeds; bars: mean. LoRA’s advantage grows on held-out regions.", 9.5)

H2("3.2 Generalization to unseen regions (H2)")
P("Both models lose performance on held-out regions: the mean drop is 7.15 ± 0.68 pp for the baseline and 5.87 ± 0.40 pp for LoRA, so the distribution shift is real and of the predicted size. LoRA’s advantage, however, does not shrink: it rises from +0.92 pp in-region to **+2.20 pp out-of-region** (+1.70, +3.35 and +1.56 pp per seed), is positive in all three seeds and is significant in each one (McNemar p < 10^−6^, Table 3). Since the out-of-region advantage exceeds the +1.3 pp threshold, **the H2 prediction is rejected**. As a secondary comparison, the models trained on the random split score 6.41 pp (baseline) and 5.73 pp (LoRA) higher on their random test than the geographic models on unseen regions: a random split overestimates real out-of-region performance by about 6 pp.")
TABLA("Exact McNemar test, baseline vs. LoRA: images correctly classified only by the baseline / only by LoRA, and two-sided p-value.", [
    ["Seed", "Internal test", "In-region test", "Out-of-region test"],
    ["42", "51 / 97 (p = 1.9×10^−4^)", "32 / 48 (p = 0.093)", "67 / 161 (p = 4.2×10^−10^)"],
    ["0", "54 / 104 (p = 8.5×10^−5^)", "29 / 63 (p = 5.1×10^−4^)", "95 / 211 (p = 2.9×10^−11^)"],
    ["1", "24 / 103 (p = 7.5×10^−13^)", "29 / 49 (p = 0.031)", "77 / 153 (p = 6.1×10^−7^)"],
], [1.4, 4.6, 4.4, 4.6])
P("**Where does LoRA gain outside the training regions?** Per-class F1 (Fig. 3) shows the largest out-of-region gains in herbaceous vegetation (+5.2 pp), permanent crops (+4.4), highways (+4.1) and rivers (+4.0); highways and rivers were already the classes that gained most on the internal split, where LoRA reduced river→highway confusions from 21 to 2. The drop concentrates in agricultural classes: permanent crops lose 34 pp (baseline) and 31 pp (LoRA) and pastures about 12 pp in both, while forest, water and industrial areas barely change. LoRA reduces the drop for herbaceous vegetation (from 11.9 to 6.8 pp) and annual crops (from 8.1 to 5.8 pp). The permanent-crop collapse is driven mainly by false positives (Fig. 4): on unseen regions herbaceous vegetation and annual crops are predicted as permanent crops (89 and 68 images per seed with the baseline; 56 and 49 with LoRA). Overall, LoRA cuts out-of-region errors from 352 to 257 per seed (−27 %). Per held-out region, LoRA improves macro-F1 in all three (+1.1 to +1.8 pp); the eastern-Mediterranean region is the hardest for both models (47.5 vs. 49.3), although it contains only 5 classes and 424 images.")
FIGURA("fig_f1_por_clase.png", "Per-class F1 (mean of three seeds) on the in-region and out-of-region tests. The drop concentrates in agricultural classes; LoRA gains mostly in herbaceous vegetation, permanent crops, highways and rivers.", 13.5)
FIGURA("fig_confusion_fuera.png", "Confusion matrices on the out-of-region test (three seeds summed). Colour: share of the row (recall); numbers: images. Herbaceous vegetation and annual crops predicted as permanent crops are the main errors of both models.", 13.5)

H2("3.3 Ablations")
P("Table 4 compares each variant with LoRA r = 8, Q+V, blocks 11–12. Rank barely matters (r = 4: +0.49 pp; r = 16: +0.08 pp on the internal split), consistent with the low intrinsic rank reported for LoRA [4], and adapting only Q is within single-seed noise (+0.45 pp internal, −0.19 pp out-of-region). Adapting the last **four** blocks is the only variant with a clear effect: with seed 42 it gained +1.19 pp (internal) and +1.48 pp (out-of-region) and *reduced* the drop by 0.69 pp, contrary to our prediction that adapting earlier, more generic blocks would hurt transfer [11]. The replication with seeds 0 and 1 confirms the direction in all three seeds but with a smaller mean effect, +0.71 pp internal and +0.83 pp out-of-region: four blocks lower the mean drop from 5.87 to 5.49 pp and raise the out-of-region advantage over the frozen model to +3.04 pp.")
TABLA("Ablations, macro-F1 (%). Δ: difference with the reference LoRA (r = 8, Q+V, blocks 11–12) of the same seed. Rows 2–5: seed 42; last row: mean over seeds 42, 0 and 1.", [
    ["Variant", "Internal F1 (Δ)", "Out-of-region F1 (Δ)", "Δ drop"],
    ["Reference: r = 8, Q+V, blocks 11–12 (seed 42)", "96.78", "90.79", "—"],
    ["r = 4", "97.27 (+0.49)", "90.54 (−0.25)", "−0.09"],
    ["r = 16", "96.86 (+0.08)", "90.97 (+0.18)", "+0.46"],
    ["Blocks 9–12", "97.97 (+1.19)", "92.27 (+1.48)", "−0.69"],
    ["Q only", "97.24 (+0.45)", "90.60 (−0.19)", "+0.30"],
    ["Blocks 9–12, three seeds (mean)", "97.89 (+0.71)", "92.29 (+0.83)", "−0.38"],
], [6.8, 3.2, 3.6, 2.2])

H2("3.4 Pre-registered predictions vs. outcomes")
P("Table 5 contrasts every prediction registered before running with its outcome. Four of nine predictions were rejected; the key one, H2 (A4), was rejected because LoRA generalized *better*, not worse, than expected.")
TABLA("Predictions registered in the decision log before running, and outcomes (pp of macro-F1; Δ with respect to LoRA r = 8, Q+V, blocks 11–12).", [
    ["ID", "Experiment", "Prediction (rejection criterion)", "Outcome", "Verdict"],
    ["A2", "Baseline vs. LoRA, 3 seeds, internal", "Δ +1.0 to +1.6 in all seeds; SD ≤ 0.3 (reject: any Δ ≤ 0 or SD > 0.5)", "Δ = +1.19, +1.32, +2.06; SD 0.47", "Not rejected; seed 1 and SD outside range"],
    ["A3", "Rank 4 and 16, internal", "Both within ±0.3 of r = 8 (reject: r16 > +0.5 or r4 < −0.5)", "r4 +0.49; r16 +0.08", "Not rejected; r4 outside range"],
    ["A4", "Phase C, 3 seeds", "Drop 3–10 in both; out-of-region advantage ≤ +0.5 (reject: drop < 1 or advantage ≥ +1.3)", "Drop 7.15 / 5.87; advantage +2.20 (3/3 positive)", "**Rejected (H2)**"],
    ["A5", "Blocks 9–12, internal", "+0.1 to +0.5 (reject: < −0.3 or > +0.8)", "+1.19", "Rejected (larger gain)"],
    ["A6", "Q only, internal", "Drop of 0.2–0.6 (reject: Q only beats Q+V or drops > 1.0)", "+0.45", "Rejected (within noise)"],
    ["A3g", "Rank, geographic", "In-region ±0.3; similar drop (reject: r16 reduces drop > 1.5)", "In-region −0.35 / +0.65; Δ drop −0.09 / +0.46", "Not rejected"],
    ["A5g", "Blocks 9–12, geographic", "In-region +0.1 to +0.4; drop 0.5–1.5 larger (reject: drop > 0.5 smaller)", "In-region +0.79; drop 0.69 smaller", "Rejected (better transfer)"],
    ["A5s", "Blocks 9–12, seeds 0 and 1, both splits", "Δ > 0 in all seeds; mean internal +0.6 to +1.4, out-of-region +0.5 to +2.0 (reject: any Δ ≤ 0 or mean < +0.3)", "Internal +0.71; out-of-region +0.83; all Δ > 0", "Not rejected"],
    ["A6g", "Q only, geographic", "In-region −0.2 to −0.5; drop 0.2–0.5 smaller (reject: out-of-region > 1.5 below baseline)", "In-region +0.10; drop 0.30 larger", "Not rejected; opposite direction"],
], [1.0, 3.0, 5.6, 4.0, 3.2], size=7.5)

H2("3.5 Diagnostics and cost")
P("**Token-norm artefacts.** On 64 out-of-region images, patch tokens with a norm above twice their image median appear from block 7 on (0.5–0.8 % of patches in blocks 8–11, about 1–2 per image, up to 4.5× the median in block 11) and vanish at the output of block 12, matching the high-norm artefacts reported for ViTs without registers [7]. **Latency.** The anomaly observed in Phase A (39.5 ms for fp16 at batch 1) is not reproduced with synchronized timing and the median of 50 runs: batch 1 takes 7.2–8.7 ms in fp32, 9.8–12.2 ms with fp16 autocast and 7.3–9.2 ms in pure fp16, so at batch 1 the GPU is under-utilized and autocast casts add 2–4 ms. **Cost.** A LoRA run takes 500–600 s on the T4, against about 5 s for the baseline once embeddings are cached (≈ 70 s the first time). After merging, inference cost is unchanged (4.43 vs. 4.41 ms/image at batch 32).")

H2("3.6 Discussion")
P("We expected part of LoRA’s advantage to be overfitting to the training regions, since its training loss becomes very small. The data say otherwise: the advantage is larger outside than inside the training regions, and LoRA’s drop is smaller than the baseline’s. A reading consistent with the teardown is that the main mismatch between DINOv2 and EuroSAT is not regional but a domain gap shared by all regions (nadir view, upsampled low-resolution tiles, textures instead of objects), which adapting the attention of the deepest blocks corrects everywhere; that adapting four blocks transfers even better points in the same direction. What is regional, the appearance of crops, still limits both models (Fig. 5): permanent crops lose more than 30 pp on unseen regions, and dark forest confused with water (≈ 11 images per seed) persists in both.")
FIGURA("fig_ejemplos.png", "Out-of-region examples (seed 42) where only LoRA, only the frozen baseline, or neither model is correct, prioritizing vegetation classes.", 11.5)

H2("3.7 Threats to validity")
P("(i) The out-of-region test covers only three regions, two of them in central Europe; another split could change the size of the drop. (ii) Ablations use one seed, except the depth ablation (three seeds). (iii) No hyper-parameter search was performed; both conditions share the same budget. (iv) A post-hoc code review found two deviations from the Phase A/B budget in the LoRA runs: fp16 training without gradient scaling and an initialization of A that did not depend on the run seed. The baseline runs are unaffected, and LoRA’s internal validation score (97.46) reproduces the earlier one (97.52). A control run with the corrected code (geographic split, seed 42) changes macro-F1 by −0.15 (validation), +0.46 (in-region) and −0.08 pp (out-of-region), and the out-of-region advantage from +1.70 to +1.63 pp, so the deviations do not alter the results. (v) The geographic split was generated in the same session as training rather than committed beforehand; it is deterministic and was regenerated identically, image by image.")

# =========================================================== 4. CONCLUSIONS
seccion("4. CONCLUSIONS")
P("Adapting DINOv2 ViT-S/14 with LoRA on the query and value projections of its last two blocks (0.11 % extra parameters and no inference cost after merging) improves EuroSAT macro-F1 on a random split (+1.5 pp) and even more on geographic regions never seen in training (+2.2 pp, consistent across three seeds). Both models lose 6–7 pp on unseen regions, mostly in agricultural classes, and a random split overestimates out-of-region performance by about 6 pp. The hypothesis that LoRA would lose its advantage outside the training regions is rejected; adapting four blocks instead of two adds a further +0.7 pp in-distribution and +0.8 pp out-of-region. Future work includes spatial cross-validation with several held-out folds, more seeds for the ablations, deeper adaptation and the register variant of DINOv2, which should remove the token-norm artefacts.")
P("**Reproducibility and use of AI.** Splits, all runs with per-image predictions, code, tests and the decision log with every pre-registered prediction are available at github.com/jhntan2075/dinov2-eurosat-lora. AI assistants (Claude, Antigravity) were used for protocol design, code, code review and drafting; the decision log details each use and what the team reviewed.", after=2)

# =========================================================== REFERENCES
seccion("REFERENCES")
REFS = [
    "M. Oquab, T. Darcet, T. Moutakanni *et al.*, “DINOv2: Learning robust visual features without supervision,” *Trans. Mach. Learn. Res.*, 2024.",
    "A. Dosovitskiy *et al.*, “An image is worth 16x16 words: Transformers for image recognition at scale,” in *Proc. ICLR*, 2021.",
    "M. Drusch *et al.*, “Sentinel-2: ESA’s optical high-resolution mission for GMES operational services,” *Remote Sens. Environ.*, vol. 120, pp. 25–36, 2012.",
    "E. J. Hu *et al.*, “LoRA: Low-rank adaptation of large language models,” in *Proc. ICLR*, 2022.",
    "P. Helber, B. Bischke, A. Dengel, and D. Borth, “EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification,” *IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens.*, vol. 12, no. 7, pp. 2217–2226, 2019.",
    "D. R. Roberts *et al.*, “Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure,” *Ecography*, vol. 40, no. 8, pp. 913–929, 2017.",
    "T. Darcet, M. Oquab, J. Mairal, and P. Bojanowski, “Vision transformers need registers,” in *Proc. ICLR*, 2024.",
    "W. Luo, Y. Li, R. Urtasun, and R. Zemel, “Understanding the effective receptive field in deep convolutional neural networks,” in *Proc. NeurIPS*, 2016.",
    "I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in *Proc. ICLR*, 2019.",
    "Q. McNemar, “Note on the sampling error of the difference between correlated proportions or percentages,” *Psychometrika*, vol. 12, no. 2, pp. 153–157, 1947.",
    "J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, “How transferable are features in deep neural networks?” in *Proc. NeurIPS*, 2014.",
]
for i, r in enumerate(REFS, 1):
    P(f"[{i}] {r}", size=8.5, after=1.5, indent=0.6)

doc.core_properties.title = "Dissecting and Adapting DINOv2 for Land Cover Classification"
doc.save(SALIDA)
print("Escrito", SALIDA, "| figuras:", nfig, "| tablas:", ntab)
