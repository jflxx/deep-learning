# DINOv2 + LoRA para clasificación de cobertura del suelo (EuroSAT RGB)

Proyecto de la tarea académica final de **1INF52 Deep Learning (PUCP, 2026-2)**.

**Pregunta de investigación:** ¿añadir adaptadores LoRA mejora la clasificación en regiones geográficas nuevas frente a usar DINOv2 congelado?

| Fase | Contenido | Estado |
|---|---|---|
| A · Teardown | Parámetros y FLOPs por bloque, latencia y memoria, PCA de tokens, atención CLS, campo receptivo efectivo, norma de tokens | E2 + E3 (diagnósticos) ✅ |
| B · Intervención | LoRA (r = 8) en Q y V de los bloques 11–12 vs linear probing, mismo presupuesto, 3 semillas | E2 + E3 ✅ |
| C · Estrés | Partición geográfica por bloques espaciales, evaluación en regiones reservadas, análisis de fallos, ablaciones | E3 ✅ |

## Resultados del E3 (3 semillas, Colab Tesla T4)

| Conjunto de prueba (macro-F1 %) | Linear probing | LoRA r=8 | Δ |
|---|---|---|---|
| Partición interna (aleatoria) | 95.66 ± 0.19 | 97.18 ± 0.39 | +1.52 |
| Geográfica, test dentro (regiones de entrenamiento) | 96.40 ± 0.22 | 97.32 ± 0.19 | +0.92 |
| Geográfica, test fuera (regiones reservadas) | 89.25 ± 0.88 | 91.45 ± 0.57 | **+2.20** |

- **Respuesta a la pregunta de investigación:** sí. Ambos modelos caen al pasar a regiones nuevas (7.2 pts el congelado, 5.9 pts LoRA), pero la ventaja de LoRA **crece** fuera de las regiones de entrenamiento (+2.20 pts, positiva en las 3 semillas, McNemar exacto p < 10⁻⁶). La hipótesis H2 (que LoRA perdería su ventaja) **se rechaza** según la regla fijada antes de ejecutar.
- La partición aleatoria sobreestima en ~6 pts el rendimiento en regiones nuevas. La caída se concentra en clases agrícolas (cultivos permanentes −31 a −34 pts).
- Ablaciones: el rango casi no importa; adaptar los 4 últimos bloques mejora +0.7 pts (interna) y +0.8 pts (fuera) en 3 semillas.
- Informe final: `informe/Informe_final_DINOv2_EuroSAT_LoRA.docx` (IEEE, 5 págs.). Detalle de decisiones, predicciones y resultados en `DESIGN_LOG.md`.

## Resultados del E2 (Colab, Tesla T4, semilla 42)

| Condición | Parámetros entrenables | Accuracy test | Macro-F1 test |
|---|---|---|---|
| Linear probing (DINOv2 congelado) | 7,690 | 0.9600 | 0.9580 |
| LoRA r=8 en Q, V (bloques 11–12) | 32,266 | **0.9728** | **0.9714** |

- LoRA mejora +1.33 pts de macro-F1 en la **partición interna** (aleatoria). Las mayores mejoras están en Highway y River; aparecen nuevas confusiones entre clases de vegetación.
- Teardown: 22.06 M parámetros, 12.25 GFLOPs por imagen a 224 px, 8.9 ms (batch 1, fp32) y 854 img/s (batch 128, fp16) en una T4.
- Detalle, interpretación y limitaciones en `DESIGN_LOG.md`; tablas en `results/` y figuras en `figures/`.

## Estructura

```
├── E2_teardown_lora.ipynb   # notebook principal del E2
├── E3_regiones_ablaciones.ipynb  # notebook del E3 tal como se ejecutó en Colab (20 corridas)
├── src/eurosat_lora/        # paquete: datos, LoRA, entrenamiento, partición geográfica, experimentos, análisis
├── app/                     # demo Streamlit (base vs LoRA): streamlit run app/streamlit_app.py
├── informe/                 # informe final (Word, IEEE) y su generador
├── tests/                   # pruebas en CPU (pytest)
├── DESIGN_LOG.md            # cuaderno de decisiones (decisiones, hipótesis, resultados, uso de IA)
├── requirements.txt
├── scripts/
│   └── C1_ejemplos_prediccion.py   # celda extra: predicciones individuales + McNemar exacto pareado
├── poster/                  # póster WAIMLAp 2026 (A0): PDF para imprimir y PPTX editable
├── abstract/                # abstract WAIMLAp 2026 en LaTeX (.tex + figura) y PDF compilado
├── docs/                    # informe de avance (Word) y diagrama de Gantt
├── splits/                  # coordenadas de EuroSAT (desde EuroSAT MS) y partición geográfica
├── results/                 # tablas CSV/JSON; E3_corridas/ = las 20 corridas del E3; E3_corridas_A5s/ = réplica de 4 bloques (semillas 0 y 1)
├── figures/                 # figuras PNG (se generan al ejecutar)
└── checkpoints/geo_s42/     # cabeza lineal y LoRA (partición geográfica, semilla 42) para la demo
```

## Cómo ejecutar

### Opción recomendada: Google Colab (GPU gratuita)
1. Abrir el notebook en Colab: `https://colab.research.google.com/github/jflxx/deep-learning/blob/main/E2_teardown_lora.ipynb`
2. *Entorno de ejecución → Ejecutar todo*. El notebook descarga EuroSAT RGB (~90 MB, Zenodo) y los pesos de DINOv2 (Hugging Face). Tarda ~30–60 min.
3. Descargar las carpetas `results/`, `figures/`, `splits/` y `checkpoints/` y subirlas al repositorio.

### Opción local
```bash
python -m venv .venv && source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook E2_teardown_lora.ipynb
```
Sin GPU NVIDIA, poner `MAX_POR_CLASE = 200` en la celda de configuración (subconjunto de 2,000 imágenes). Las GPU AMD de la serie RX 500 no son compatibles con PyTorch.

## Cómo reproducir el E3
```bash
pip install -e .                                   # paquete eurosat_lora y dependencias
pytest -q                                          # pruebas en CPU
python scripts/E3_analisis.py                      # tablas results/C*.csv y figuras figures/C*.png desde results/E3_corridas/
streamlit run app/streamlit_app.py                 # demo
```
- Corridas en GPU: `E3_regiones_ablaciones.ipynb` (20 corridas), `scripts/E3_control_correccion.py` y `scripts/E3_bloques_semillas.py` (Colab T4).
- Partición geográfica: `scripts/E3_coordenadas_eurosat_ms.py` (descarga EuroSAT MS, ~2 GB) y `scripts/E3_particion_geo.py`; ya están en `splits/`.
- Desviaciones declaradas (D21 en `DESIGN_LOG.md`): GradScaler e inicialización de A en las corridas originales; una corrida de control muestra que no cambian los resultados.

## Reproducibilidad
- Semilla fija (`SEMILLA = 42`) en Python, NumPy y PyTorch; cuDNN determinista.
- La partición se guarda en `splits/split_seed42.json` y se reutiliza en todas las ejecuciones.
- Versiones y hardware de cada ejecución en `results/entorno.json`.
- Todos los hiperparámetros están en la celda de configuración del notebook y en `DESIGN_LOG.md`.

## Póster y abstract (WAIMLAp 2026)

- **Póster:** `poster/Poster_DINOv2_EuroSAT_LoRA.pdf` (A0, para imprimir) y `.pptx` (editable). Incluye el pipeline completo, el diagrama de LoRA, los resultados de las fases A y B y predicciones individuales reales del conjunto de test.
- **Abstract:** `abstract/Abstract_DINOv2_EuroSAT_LoRA.tex` (compilar con `pdflatex` dos veces; la figura `fig_predicciones.png` va en la misma carpeta) y el PDF compilado.
- **Predicciones individuales:** `scripts/C1_ejemplos_prediccion.py` se pega como última celda del notebook después de *Ejecutar todo*. Genera `figures/C1_prediction_examples.png`, `results/C1_predicciones_ejemplo.csv` y `results/C1_mcnemar_exacto.json`.
- **Nota:** el McNemar del póster (χ² = 9.56, p ≈ 0.002) es una cota conservadora (peor caso). El McNemar exacto con predicciones pareadas está en `results/C3_mcnemar_exacto.csv` (E3).

## Datos y modelos (licencias)
- **EuroSAT** — Helber, P., Bischke, B., Dengel, A., Borth, D. (2019). *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.* IEEE JSTARS 12(7), 2217–2226. [GitHub](https://github.com/phelber/EuroSAT) · [Zenodo](https://zenodo.org/records/7711810). Licencia MIT; imágenes Copernicus Sentinel-2 sujetas a sus términos de uso.
- **DINOv2** — Oquab, M. et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision.* [arXiv:2304.07193](https://arxiv.org/abs/2304.07193). Código y pesos: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2), licencia Apache-2.0. Se usa la versión de Hugging Face `facebook/dinov2-small`.
- **LoRA** — Hu, E. J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685). Implementación propia en el notebook.
- **Campo receptivo efectivo** — Luo, W. et al. (2016). *Understanding the Effective Receptive Field in Deep Convolutional Neural Networks.* NeurIPS.

## Uso de asistencia de IA
Se usaron asistentes de IA como herramienta de trabajo. El detalle (qué, dónde y qué revisó el equipo) está en la sección 4 de `DESIGN_LOG.md`.

## Equipo
- Alfredo Jimenez
- Jhonatan Davila
- Italo Mijail
