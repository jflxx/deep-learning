# E3 · Regiones, semillas y ablaciones — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar todo lo necesario para que el equipo ejecute el E3 en Colab (partición geográfica, 3 semillas, ablaciones, Fase C, diagnósticos) y cerrarlo con demo y documentación.

**Architecture:** El código reutilizable del E2 pasa a un paquete `src/eurosat_lora/` probado en CPU con un DINOv2 diminuto de configuración aleatoria. Scripts de una sola vez generan coordenadas y partición geográfica; un notebook delgado de Colab recorre la matriz de experimentos de forma idempotente, guardando cada corrida en Drive; un módulo de análisis produce las tablas `results/C*.csv`, las figuras y el veredicto de A4.

**Tech Stack:** Python ≥3.10, PyTorch, torchvision, transformers (`Dinov2Model`), scikit-learn, scipy, pandas, matplotlib, rasterio, pyproj, streamlit, pytest.

**Spec:** `docs/superpowers/specs/2026-11-24-e3-design.md`

## Global Constraints

- Presupuesto idéntico al E2 en todas las corridas: AdamW, lr 1e-3, weight decay 1e-4, batch 64, 8 épocas, selección por mejor macro-F1 de validación, imagen 224 px bicúbico sin recorte, normalización ImageNet, representación [CLS ; media de parches] tras la norma final.
- LoRA por defecto: r=8, α=r, dropout 0, Q y V, bloques 11–12 (índices 0-based `(10, 11)`), B inicializada en 0, A Kaiming uniforme `a=√5`.
- Modelo: `facebook/dinov2-small`. Semillas de entrenamiento: 42, 0, 1. Particiones fijas (no cambian con la semilla).
- Partición interna: estratificada 70/15/15, semilla 42 → 18,900 / 4,050 / 4,050.
- Partición geográfica: K-means K=12 sobre EPSG:3035, semilla 42; test fuera y validación por regiones completas distintas, cada uno entre 12 % y 18 % del total; cada clase ≥5 % de sus imágenes en test fuera y en validación; 20,000 intentos; respaldo en orden fijo (K=15, K=20, mínimo 3 %); test dentro = 10 % del total, aleatorio estratificado de las regiones de train.
- El test (interno, dentro o fuera) nunca se usa para decisiones.
- No modificar `E2_teardown_lora.ipynb`.
- Commits sin `Co-Authored-By` ni atribución a IA. Trabajo en la rama `e3-regiones-ablaciones`.
- Todo número en documentos sale de archivos generados por el código.
- La GPU local (RX 580) no sirve: las pruebas locales son en CPU; las corridas completas, en Colab T4 por el equipo.

## Review Focus

1. **Nombres RGB↔MS que no coinciden o faltan** → el script de coordenadas debe detenerse listando los faltantes, nunca emparejar por aproximación. Test en Task 5.
2. **Colab se desconecta a mitad de una corrida** → una carpeta sin `metricas.json` se vuelve a ejecutar desde cero; `metricas.json` se escribe al final y de forma atómica (archivo temporal + `os.replace`). Test en Task 7.
3. **Lanzar una ablación sin su predicción registrada** → el notebook debe negarse con un mensaje claro que nombre el ID faltante. Test en Task 7.
4. **Demo sin checkpoints (antes de Colab)** → la app muestra cómo obtenerlos en lugar de un traceback. Test en Task 10.
5. **Diferencias de versión de transformers** (capas `query`/`q_proj`, bloques que devuelven tupla o tensor) → `aplicar_lora` y `adelante_lora` funcionan con ambos casos. Test en Task 1.

---

## Estructura de archivos

| Archivo | Responsabilidad |
|---|---|
| `pyproject.toml` | Paquete instalable `eurosat_lora` (src layout) + config de pytest |
| `src/eurosat_lora/config.py` | `Config` (dataclass) con los valores del E2 |
| `src/eurosat_lora/modelo.py` | DINOv2, `LoRALinear`, `aplicar_lora`, `adelante_lora`, `fusionar_lora`, `representacion` |
| `src/eurosat_lora/datos.py` | Descarga EuroSAT RGB, dataset, partición estratificada, DataLoaders, archivos↔índices |
| `src/eurosat_lora/metricas.py` | Métricas, F1 por clase, McNemar exacto |
| `src/eurosat_lora/entrenamiento.py` | Extracción de features, entrenamiento base y LoRA, predicción |
| `src/eurosat_lora/geo.py` | Coordenadas, proyección, regiones, asignación, test dentro, distancias |
| `src/eurosat_lora/experimentos.py` | `Corrida`, `MATRIZ_E3`, `ejecutar_corrida`, chequeo de predicciones |
| `src/eurosat_lora/analisis.py` | Tablas C2–C7, `veredicto_a4`, figuras |
| `src/eurosat_lora/diagnosticos.py` | Norma de tokens, latencia (re-medición fp16 batch 1) |
| `scripts/E3_coordenadas_eurosat_ms.py` | MS → `splits/eurosat_coords.csv` |
| `scripts/E3_particion_geo.py` | → `splits/split_geo.json`, `results/C0_particion_geo.csv`, `figures/C0_mapa_regiones.png` |
| `E3_regiones_ablaciones.ipynb` | Notebook de Colab |
| `app/demo_nucleo.py`, `app/streamlit_app.py` | Demo |
| `tests/` | pytest en CPU |

Tests usan un modelo diminuto: `Dinov2Config(hidden_size=32, num_hidden_layers=4, num_attention_heads=2, intermediate_size=64, image_size=28, patch_size=14)` → `Dinov2Model(config)`, sin descargas (fixture `modelo_chico` en `tests/conftest.py`).

---

### Task 1: Paquete, configuración y modelo con LoRA

**Files:**
- Create: `pyproject.toml`, `src/eurosat_lora/__init__.py`, `src/eurosat_lora/config.py`, `src/eurosat_lora/modelo.py`, `tests/conftest.py`, `tests/test_modelo.py`
- Modify: `.gitignore` (añadir `.venv/`, `*.egg-info/`, `.pytest_cache/`)

**Interfaces:**
- Produces:
  - `Config` dataclass: `modelo_id='facebook/dinov2-small'`, `img_size=224`, `batch=64`, `epocas=8`, `lr=1e-3`, `weight_decay=1e-4`, `lora_rango=8`, `lora_alpha=8`, `lora_dropout=0.0`, `lora_bloques=(10, 11)`, `lora_proyecciones=('q','v')`, `max_por_clase=None`, `num_workers=2`.
  - `fijar_semilla(seed: int) -> None`
  - `cargar_dinov2(modelo_id: str, device: str) -> Dinov2Model` (congelado, `eval()`)
  - `class LoRALinear(nn.Module)`: `__init__(base: nn.Linear, r: int, alpha: float, dropout: float)`, `forward`, `fusionar() -> nn.Linear` — copia exacta del E2.
  - `aplicar_lora(m, bloques: tuple[int,...], proyecciones: tuple[str,...], r: int, alpha: float, dropout: float) -> list[str]`; `'q'` ↔ `('query','q_proj')`, `'v'` ↔ `('value','v_proj')`; `assert len(reemplazados) == len(proyecciones) * len(bloques)`.
  - `representacion(ultima: Tensor) -> Tensor` (B, 2D)
  - `adelante_lora(m, x: Tensor, primer_bloque: int) -> Tensor` — sin gradiente antes de `primer_bloque`, usa `salida_bloque`.
  - `fusionar_lora(m) -> int` (número de capas fusionadas)
  - `contar_entrenables(m) -> int`

- [ ] **Step 1: Crear `.venv` e instalar dependencias de desarrollo**

Run: `python -m venv .venv && .venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && .venv/Scripts/python -m pip install transformers scikit-learn scipy pandas matplotlib pyproj rasterio streamlit pytest nbformat nbconvert ipykernel`
Expected: instala sin error. Si torch no tiene wheel para Python 3.14, usar `py -3.12 -m venv .venv` (instalar Python 3.12 si hace falta) y repetir.

- [ ] **Step 2: Escribir tests que fallan** (`tests/test_modelo.py`)

```python
def test_lora_b_cero_salida_identica(modelo_chico): ...      # aplicar_lora sobre copia → max|Δ last_hidden_state| == 0
def test_parametros_lora_qv_dos_bloques():                    # con D=384: 2 bloques × 2 proy × 8 × (384+384) == 24_576 (calcular con LoRALinear sobre nn.Linear(384,384))
def test_aplicar_lora_solo_q(modelo_chico):                   # proyecciones=('q',), bloques=(2,3) → 2 reemplazos, ninguno 'value'
def test_aplicar_lora_cuatro_bloques(modelo_chico):           # bloques=(0,1,2,3) → 8 reemplazos
def test_adelante_lora_igual_forward_completo(modelo_chico):  # allclose(adelante_lora(m,x,2), representacion(m(x).last_hidden_state), atol=1e-5)
def test_fusion_equivalente(modelo_chico):                    # B aleatoria; salida antes == después de fusionar_lora (atol 1e-5); no quedan LoRALinear
def test_salida_bloque_tupla_o_tensor():                      # salida_bloque((t,)) is t y salida_bloque(t) is t
```

- [ ] **Step 3: Verificar que fallan** — Run: `.venv/Scripts/python -m pytest tests/test_modelo.py -v` · Expected: FAIL (ImportError).
- [ ] **Step 4: Implementar** `config.py` y `modelo.py` copiando la lógica de las celdas 3, 8, 21, 24, 26 y 29 del E2, sin variables globales (todo por parámetro). `pyproject.toml` con `[tool.setuptools.packages.find] where=["src"]` y `[tool.pytest.ini_options] testpaths=["tests"]`; instalar con `pip install -e .`.
- [ ] **Step 5: Verificar** — Run: `.venv/Scripts/python -m pytest tests/test_modelo.py -v` · Expected: 7 passed.
- [ ] **Step 6: Commit** — `git add pyproject.toml src tests .gitignore && git commit -m "E3: paquete eurosat_lora con LoRA generalizado (bloques y proyecciones)"`

---

### Task 2: Datos y partición interna

**Files:**
- Create: `src/eurosat_lora/datos.py`, `tests/test_datos.py`

**Interfaces:**
- Consumes: `Config`.
- Produces:
  - `EUROSAT_RGB_URL = 'https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip'`
  - `preparar_eurosat_rgb(data_dir: str) -> str` (descarga si falta, devuelve raíz de clases)
  - `construir_dataset(raiz: str, img_size: int) -> ImageFolder`
  - `particion_estratificada(y: np.ndarray, frac_val: float, frac_test: float, seed: int, max_por_clase: int|None) -> tuple[list[int], list[int], list[int]]` — idéntica al E2.
  - `archivos_de(dataset, indices) -> list[str]` (ruta relativa `Clase/Clase_N.jpg`); `indices_de(dataset, archivos) -> list[int]` (KeyError si falta alguno).
  - `cargador(dataset, indices, batch: int, shuffle: bool, seed: int, num_workers: int, pin_memory: bool) -> DataLoader`
  - `guardar_particion_interna(dataset, ruta: str, seed: int, max_por_clase) -> dict` / `cargar_particion(ruta) -> dict` con claves `train`, `val`, `test` (listas de archivos).

- [ ] **Step 1: Tests que fallan**

```python
CONTEOS = dict(AnnualCrop=3000, Forest=3000, HerbaceousVegetation=3000, Highway=2500, Industrial=2500,
               Pasture=2000, PermanentCrop=2500, Residential=3000, River=2500, SeaLake=3000)
def test_particion_tamanos_e2():        # y sintético con CONTEOS → len = 18_900, 4_050, 4_050; disjuntos; unión = 27_000
def test_particion_determinista():      # dos llamadas seed 42 → listas idénticas
def test_archivos_indices_ida_vuelta(tmp_path):  # ImageFolder con 3 clases × 4 jpg 64×64 → indices_de(archivos_de(ix)) == ix
def test_indices_de_archivo_faltante(tmp_path):  # pytest.raises(KeyError)
```

- [ ] **Step 2:** Run `pytest tests/test_datos.py -v` · Expected: FAIL.
- [ ] **Step 3:** Implementar `datos.py` (transform y partición del E2, celdas 5–6).
- [ ] **Step 4:** Run `pytest tests/test_datos.py -v` · Expected: 4 passed.
- [ ] **Step 5: Commit** — `git commit -m "E3: módulo de datos y partición interna reproducible"`

---

### Task 3: Métricas y McNemar exacto

**Files:**
- Create: `src/eurosat_lora/metricas.py`, `tests/test_metricas.py`

**Interfaces:**
- Produces:
  - `metricas(y, p) -> dict` con `accuracy`, `macro_f1`
  - `f1_por_clase(y, p, clases: list[str]) -> pd.Series`
  - `mcnemar_exacto(y, p1, p2) -> dict` con `b` (solo p1 acierta), `c` (solo p2 acierta), `n_discordantes`, `p_valor` (`scipy.stats.binomtest(b, b + c, 0.5).pvalue`, bilateral; si b+c=0 → p_valor 1.0)

- [ ] **Step 1: Tests que fallan**

```python
def test_mcnemar_conteos():   # y=[0,0,0,0,1,1], p1=[0,0,1,1,1,0], p2=[0,1,0,0,1,1] → b=1, c=3, n_discordantes=4
def test_mcnemar_valor_conocido():  # b=10, c=30 → p_valor == pytest.approx(binomtest(10, 40, 0.5).pvalue)
def test_mcnemar_sin_discordantes(): # p1 == p2 → p_valor == 1.0
def test_metricas_perfectas():       # y == p → accuracy 1.0, macro_f1 1.0
```

- [ ] **Step 2:** FAIL · **Step 3:** implementar · **Step 4:** Run `pytest tests/test_metricas.py -v` → 4 passed.
- [ ] **Step 5: Commit** — `git commit -m "E3: métricas y McNemar exacto"`

---

### Task 4: Entrenamiento base y LoRA

**Files:**
- Create: `src/eurosat_lora/entrenamiento.py`, `tests/test_entrenamiento.py`

**Interfaces:**
- Consumes: `modelo.*`, `metricas.metricas`, `Config`.
- Produces:
  - `extraer_features(m, dl, device: str, amp: bool) -> tuple[Tensor, Tensor]`
  - `entrenar_base(cache: dict[str, tuple[Tensor,Tensor]], n_clases: int, cfg: Config, seed: int, device: str) -> tuple[nn.Linear, pd.DataFrame]` (claves de cache: `train`, `val`)
  - `predecir_base(cabeza, X: Tensor, device) -> tuple[np.ndarray, np.ndarray]` (predicción, probabilidad máx.)
  - `entrenar_lora(m, dl_train, dl_val, n_clases: int, cfg: Config, seed: int, device: str, amp: bool) -> tuple[nn.Linear, pd.DataFrame, dict]` — `primer_bloque = min(cfg.lora_bloques)`; restaura los mejores pesos (cabeza + LoRA) por macro-F1 de validación; dict = `{'cabeza': state_dict, 'lora': {nombre: tensor}}`.
  - `predecir_lora(m, cabeza, dl, primer_bloque: int, device, amp) -> tuple[np.ndarray, np.ndarray]`
  - Historial con columnas `epoca, loss_train, val_accuracy, val_macro_f1`.

- [ ] **Step 1: Tests que fallan** (dataset sintético `TensorDataset` de 40 imágenes 3×28×28, 2 clases separables por brillo; `cfg` con `epocas=2`, `batch=8`, `lora_bloques=(2,3)`)

```python
def test_entrenar_base_historial_y_seleccion(modelo_chico):   # hist tiene 2 filas; macro_f1 val de la cabeza devuelta == max(hist.val_macro_f1)
def test_entrenar_lora_restaura_mejor(modelo_chico):           # macro_f1 val con pesos devueltos == max(hist.val_macro_f1); solo A/B y cabeza cambian (W de query igual a la original)
def test_entrenar_lora_determinista(modelo_chico):             # dos corridas seed 0 → mismos pesos LoRA (allclose)
def test_predecir_devuelve_prob_en_0_1(modelo_chico):          # 0 <= prob <= 1, len == n
```

- [ ] **Step 2:** FAIL · **Step 3:** implementar desde las celdas 21, 22 y 26 del E2 (los DataLoaders entran por parámetro; nada global).
- [ ] **Step 4:** Run `pytest tests/test_entrenamiento.py -v` → 4 passed.
- [ ] **Step 5: Commit** — `git commit -m "E3: entrenamiento base y LoRA sin estado global"`

---

### Task 5: Coordenadas y partición geográfica

**Files:**
- Create: `src/eurosat_lora/geo.py`, `scripts/E3_coordenadas_eurosat_ms.py`, `scripts/E3_particion_geo.py`, `tests/test_geo.py`

**Interfaces:**
- Consumes: `datos.particion_estratificada` (para el test dentro, sobre las imágenes de train).
- Produces:
  - `EUROSAT_MS_URL = 'https://zenodo.org/records/7711810/files/EuroSAT_MS.zip'`
  - `centro_lonlat(transform: affine.Affine, crs, ancho: int, alto: int) -> tuple[float, float]`
  - `verificar_emparejamiento(nombres_ms: set[str], nombres_rgb: set[str]) -> None` — compara por `Clase/Clase_N` sin extensión; `ValueError` con hasta 20 nombres faltantes de cada lado.
  - `proyectar(lon, lat) -> np.ndarray` (N, 2) en EPSG:3035
  - `agrupar_regiones(xy, k: int, seed: int) -> np.ndarray` (KMeans, `n_init=10`)
  - `asignar_regiones(region, y, k, min_frac_clase, rango_frac=(0.12, 0.18), n_intentos=20_000, seed=42) -> dict | None` → `{'test_fuera': [regiones], 'val': [regiones]}`; minimiza `|f_test−0.15| + |f_val−0.15|` entre las válidas.
  - `RESPALDO = [(12, 0.05), (15, 0.05), (20, 0.05), (20, 0.03)]`
  - `particion_geografica(coords: pd.DataFrame, seed=42) -> dict` — recorre `RESPALDO`, registra `k`, `min_frac_clase`, `intento_respaldo`; separa test dentro (10 % del total, estratificado) de las regiones restantes; devuelve `{'train','val','test_dentro','test_fuera'}` (listas de `archivo_rgb`) + `region` por archivo + parámetros. `RuntimeError` si ningún nivel de respaldo funciona.
  - `distancias_test_train(xy_test, xy_train) -> dict` (min y mediana en km, `scipy.spatial.cKDTree`)
  - Script coords: descarga MS, lee cada `.tif` con `rasterio`, escribe `splits/eurosat_coords.csv` (`archivo_rgb, clase, lon, lat, crs_epsg`); comprueba 27,000 filas; llama `verificar_emparejamiento` contra los nombres del zip RGB.
  - Script partición: lee el CSV; escribe `splits/split_geo.json`, `results/C0_particion_geo.csv` (imágenes por clase × conjunto), `figures/C0_mapa_regiones.png` (lon/lat coloreado por conjunto, número de región anotado en su centroide); imprime distancias.

- [ ] **Step 1: Tests que fallan** (coords sintéticas: 24 grupos gaussianos sobre Europa, 10 clases mezcladas, 6,000 puntos)

```python
def test_centro_lonlat_utm():            # transform de un tile 64×64 a 10 m en EPSG:32632 con origen (500000, 5300000) → lon≈9.0, lat≈47.8 (comparar con pyproj, atol 1e-6)
def test_emparejamiento_falla_con_faltantes():  # pytest.raises(ValueError, match='River_9')
def test_particion_disjunta_y_completa():       # 4 conjuntos disjuntos, unión == todas
def test_regiones_completas():                  # ninguna región aparece en test_fuera y en train/val a la vez; test_fuera y val sin regiones comunes
def test_restriccion_clases_y_tamanos():        # cada clase ≥ 5 % en test_fuera y val; 0.12 ≤ frac ≤ 0.18; test_dentro ≈ 10 % (±1 %)
def test_determinista():                        # dos llamadas → mismo dict
def test_respaldo_se_registra():                # coords donde una clase está en solo 2 regiones → intento_respaldo > 0 o RuntimeError; nunca devuelve partición que viole la restricción
```

- [ ] **Step 2:** FAIL · **Step 3:** implementar `geo.py` y ambos scripts (los scripts solo orquestan; la lógica vive en `geo.py`).
- [ ] **Step 4:** Run `pytest tests/test_geo.py -v` → 7 passed.
- [ ] **Step 5: Commit** — `git commit -m "E3: coordenadas desde EuroSAT MS y partición geográfica por bloques"`

---

### Task 6: Predicciones previas y decisiones en el cuaderno (puerta del equipo)

**Files:**
- Modify: `DESIGN_LOG.md` (estado, sección 1: D17–D20; sección 2: regla de A4 y predicciones nuevas; sección 4: uso de IA 24/11)

**Interfaces:**
- Produces: en la tabla de predicciones, filas con IDs exactos `A5`, `A6`, `A3g`, `A5g`, `A6g` (en la primera columna, formato `| A5 |`) y la subsección "Regla de decisión de A4". Task 7 verifica su presencia por esos IDs.

- [ ] **Step 1:** Redactar D17 (partición geográfica: K, restricción, respaldo, test dentro), D18 (módulo `src/` + notebook E3), D19 (corridas idempotentes en Drive), D20 (ablaciones en ambas particiones), con fundamento y alternativas, como en D01–D16.
- [ ] **Step 2:** Redactar la regla de decisión de A4 copiando la tabla de 5.10 del spec.
- [ ] **Step 3:** Redactar predicciones A5, A6 (interna) y A3g, A5g, A6g (geográfica: efecto sobre test dentro, test fuera y caída respecto a r8), cada una con predicción numérica, razonamiento basado **solo** en literatura y diseño (citar [3] Hu et al. para Q vs Q+V y para rango; [5] Yosinski para profundidad) y criterio de rechazo. Fecha 24/11/2026, marcadas "escritas ANTES de ejecutar".
- [ ] **Step 4:** Commit — `git commit -m "Cuaderno: decisiones D17–D20, regla de A4 y predicciones previas de las ablaciones"`
- [ ] **Step 5: PARAR y pedir al equipo** que revise y ajuste este commit antes del Task 8 en Colab. No continuar a corridas reales sin su confirmación.

---

### Task 7: Experimentos idempotentes

**Files:**
- Create: `src/eurosat_lora/experimentos.py`, `tests/test_experimentos.py`

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces:
  - `@dataclass(frozen=True) Corrida`: `id: str`, `experimento: str` (`A2`, `A4`, `A3`, `A5`, `A6`, `A3g`, `A5g`, `A6g`), `particion: Literal['interna','geo']`, `condicion: Literal['base','lora']`, `semilla: int`, `rango: int = 8`, `bloques: tuple = (10, 11)`, `proyecciones: tuple = ('q','v')`. `id` = `f"{particion}_{condicion}_r{rango}_b{bloques}_{proy}_s{semilla}"` normalizado sin espacios.
  - `MATRIZ_E3: list[Corrida]` — 20 corridas: base y LoRA r8 × {interna, geo} × {42, 0, 1} (12), + {interna, geo} × {r4, r16, bloques (8,9,10,11), proyecciones ('q',)} semilla 42 (8).
  - `@dataclass Contexto`: `dataset`, `clases: list[str]`, `particiones: dict[str, dict[str, list[int]]]` (interna: `train/val/test`; geo: `train/val/test_dentro/test_fuera`), `cfg: Config`, `device: str`, `amp: bool`, `cache_features: dict` (por partición, se llena una vez).
  - `ejecutar_corrida(c: Corrida, ctx: Contexto, salida: str) -> dict` — si `salida/c.id/metricas.json` existe, lo carga y devuelve; si no, borra la carpeta parcial, entrena, escribe `historial.csv`, `predicciones_<conjunto>.csv` (`archivo, real, predicho, prob`), pesos (`cabeza.pt` o `lora_y_cabeza.pt`) y al final `metricas.json` atómico (incluye métricas por conjunto de test, segundos, parámetros entrenables, config y `ENTORNO`).
  - `PREDICCION_REQUERIDA = {'A5','A6','A3g','A5g','A6g'}`; `verificar_predicciones(ids: set[str], ruta_log='DESIGN_LOG.md') -> None` — `RuntimeError` nombrando los IDs sin fila `| ID |` en el cuaderno.

- [ ] **Step 1: Tests que fallan**

```python
def test_matriz_e3():                    # 20 corridas, ids únicos, 14 lora + 6 base, 12 con semillas {42,0,1}
def test_corrida_idempotente(tmp_path, ctx_chico):   # 1ª llamada crea metricas.json; 2ª no reentrena (monkeypatch entrenar_lora para que falle) y devuelve lo mismo
def test_corrida_parcial_se_rehace(tmp_path, ctx_chico):  # carpeta con historial.csv pero sin metricas.json → se ejecuta y se completa
def test_geo_evalua_dentro_y_fuera(tmp_path, ctx_chico):  # existen predicciones_test_dentro.csv y predicciones_test_fuera.csv
def test_verificar_predicciones_falta(tmp_path):      # log sin '| A6 |' → RuntimeError con 'A6'
```

(`ctx_chico`: dataset `ImageFolder` sintético de 2 clases × 20 imágenes, `modelo_chico` inyectado vía monkeypatch de `cargar_dinov2`, `epocas=1`.)

- [ ] **Step 2:** FAIL · **Step 3:** implementar · **Step 4:** Run `pytest tests/test_experimentos.py -v` → 5 passed.
- [ ] **Step 5: Commit** — `git commit -m "E3: matriz de experimentos y corridas reanudables"`

---

### Task 8: Análisis, veredicto de A4 y diagnósticos

**Files:**
- Create: `src/eurosat_lora/analisis.py`, `src/eurosat_lora/diagnosticos.py`, `tests/test_analisis.py`, `tests/test_diagnosticos.py`

**Interfaces:**
- Consumes: carpetas de corridas (Task 7), `metricas.*`.
- Produces:
  - `cargar_corridas(salida: str) -> pd.DataFrame` (una fila por corrida y conjunto de test)
  - `tabla_semillas(df) -> pd.DataFrame` → `results/C2_semillas.csv`
  - `tabla_mcnemar(salida) -> pd.DataFrame` → `results/C3_mcnemar_exacto.csv` (base vs LoRA r8 por semilla y conjunto)
  - `tabla_fase_c(df) -> pd.DataFrame` → `results/C4_fase_c.csv`
  - `veredicto_a4(tabla_fase_c) -> dict` con `veredicto ∈ {'se_cumple','se_rechaza','intermedio'}`, `caida_media_base`, `caida_media_lora`, `ventaja_media_fuera`, `signo_consistente: bool`
  - `f1_por_clase_geo`, `f1_por_region` → `C5`, `C6`; `tabla_ablaciones(df) -> C7`
  - Figuras: `C4_interno_vs_geo.png`, `C5_confusion_geo.png`, `C6_mapa_errores.png`, `C8_ejemplos_fallo_geo.png`
  - `norma_tokens(m, x) -> np.ndarray` (B, N) norma L2 por token del último bloque → `C9_norma_tokens.csv` + figura
  - `medir_latencia(m, batch, img_size, device, reps, warmup, amp) -> dict` (del E2) y `remedir_fp16_b1(m, device) -> pd.DataFrame` (warmup 5 vs 50; con/sin autocast; `.half()` puro) → `C10_latencia_fp16_b1.csv`

- [ ] **Step 1: Tests que fallan**

```python
# veredicto_a4 con tablas sintéticas (3 semillas):
def test_a4_se_cumple():     # caídas base [5,6,4], lora [6,7,5]; ventaja fuera [0.2,-0.1,0.3] → 'se_cumple'
def test_a4_rechazo_sin_caida():   # caída base media 0.5 → 'se_rechaza'
def test_a4_rechazo_ventaja():     # ventaja fuera [1.5,1.4,1.3] → 'se_rechaza'
def test_a4_intermedio():          # caídas [2,2,2] → 'intermedio'
def test_a4_signo_inconsistente(): # ventaja fuera [0.4,-0.3,0.2] → signo_consistente False
def test_norma_tokens_forma(modelo_chico):  # x (2,3,28,28) → (2, 5)
```

- [ ] **Step 2:** FAIL · **Step 3:** implementar (umbrales exactos de spec 5.10: [3,10], ≤0.5, <1, ≥1.3) · **Step 4:** Run `pytest tests/test_analisis.py tests/test_diagnosticos.py -v` → 6 passed.
- [ ] **Step 5: Commit** — `git commit -m "E3: análisis, veredicto de A4 y diagnósticos"`

---

### Task 9: Notebook de Colab y prueba de humo local

**Files:**
- Create: `E3_regiones_ablaciones.ipynb`, `scripts/E3_humo_local.py`
- Modify: `requirements.txt` (añadir `scipy`, `rasterio`, `pyproj`, `streamlit`, `pytest`)

**Interfaces:**
- Consumes: todo lo anterior.
- Secciones del notebook (markdown + código delgado): 0) clonar repo / `pip install -e .` / montar Drive (`SALIDA='/content/drive/MyDrive/E3'`) / `MODO_PRUEBA=False`; 1) datos y partición interna; 2) coordenadas (ejecuta el script si falta el CSV) y partición geo (carga `split_geo.json`; **falla si no existe en el repo**, para no generarla tras entrenar); 3) `verificar_predicciones(PREDICCION_REQUERIDA)`; 4) bucle sobre `MATRIZ_E3` con `ejecutar_corrida`, imprimiendo progreso; 5) análisis y figuras; 6) diagnósticos; 7) copiar `results/`, `figures/`, `checkpoints/` (solo semilla 42 geo e interna) a Drive en un zip.
- `scripts/E3_humo_local.py`: ejecuta la cadena completa con `max_por_clase=20`, `epocas=1`, coordenadas **sintéticas** escritas en un directorio temporal (nunca en `splits/`), salida en temporal; termina imprimiendo `HUMO OK`.

- [ ] **Step 1:** Escribir el notebook y el script de humo.
- [ ] **Step 2:** Run `.venv/Scripts/python scripts/E3_humo_local.py` · Expected: última línea `HUMO OK`, sin archivos nuevos en `splits/`, `results/`, `figures/` (`git status` limpio salvo lo creado en esta tarea).
- [ ] **Step 3:** Run `.venv/Scripts/python -m pytest -q` · Expected: todos pasan.
- [ ] **Step 4:** Validar el JSON del notebook: `.venv/Scripts/python -c "import nbformat; nbformat.validate(nbformat.read('E3_regiones_ablaciones.ipynb', 4))"` → sin error.
- [ ] **Step 5: Commit + push** — `git commit -m "E3: notebook de Colab y prueba de humo" && git push origin e3-regiones-ablaciones`

---

### Task 10: Demo Streamlit

**Files:**
- Create: `app/demo_nucleo.py`, `app/streamlit_app.py`, `tests/test_demo.py`

**Interfaces:**
- Consumes: `modelo.*`, `datos.construir_dataset` (transform), checkpoints `checkpoints/geo_s42/{cabeza_base.pt, lora_y_cabeza.pt}` y `splits/split_geo.json`.
- Produces:
  - `RUTA_CKPT = 'checkpoints/geo_s42'`
  - `checkpoints_disponibles(ruta) -> bool`
  - `cargar_modelos(ruta, device='cpu') -> dict` (`base`, `lora`, `clases`)
  - `predecir_imagen(modelos, img: PIL.Image) -> dict` → `{'base': [(clase, prob)]*3, 'lora': [(clase, prob)]*3}`
  - App: selector "subir imagen" / "ejemplo de test fuera"; dos columnas base vs LoRA con top-3 y acierto/error si hay etiqueta; si faltan checkpoints, `st.info` con los pasos para obtenerlos (Task 9, sección 7) y `st.stop()`.

- [ ] **Step 1: Tests que fallan**

```python
def test_sin_checkpoints(tmp_path):            # checkpoints_disponibles(tmp_path) is False
def test_predecir_imagen_top3(modelos_chicos): # 3 pares por modelo, probs descendentes, suman ≤ 1
def test_app_sin_checkpoints_no_falla(monkeypatch, tmp_path):  # streamlit.testing.v1.AppTest.from_file('app/streamlit_app.py') con RUTA_CKPT vacía → sin excepción y con un st.info
```

- [ ] **Step 2:** FAIL · **Step 3:** implementar · **Step 4:** Run `pytest tests/test_demo.py -v` → 3 passed.
- [ ] **Step 5: Commit** — `git commit -m "E3: demo Streamlit base vs LoRA"`

---

### Task 11: Ejecución por el equipo (fuera de esta sesión)

- [ ] **Step 1:** Colab sin GPU: `python scripts/E3_coordenadas_eurosat_ms.py` → commit de `splits/eurosat_coords.csv`.
- [ ] **Step 2:** `python scripts/E3_particion_geo.py` → revisar `figures/C0_mapa_regiones.png` y `results/C0_particion_geo.csv` → commit **antes de entrenar**.
- [ ] **Step 3:** Confirmar el commit del Task 6 (predicciones y regla de A4).
- [ ] **Step 4:** Colab T4: ejecutar `E3_regiones_ablaciones.ipynb` completo (~2.5–3.5 h, reanudable). Descargar el zip de Drive y copiar su contenido al repo.

---

### Task 12: Cierre con resultados reales

**Files:**
- Modify: `DESIGN_LOG.md`, `README.md`, `requirements.txt`, `results/README.md`
- Create: `docs/Informe_final_TA_DINOv2_LoRA_EuroSAT.docx`

- [ ] **Step 1:** Verificar que existen todos los archivos `results/C0`–`C10` y figuras `C0`–`C9`; `veredicto_a4` corre sobre `C4_fase_c.csv`.
- [ ] **Step 2:** `DESIGN_LOG.md`: completar columna "Resultado" de A2, A3, A4, A5, A6, A3g, A5g, A6g sin tocar las predicciones; contraste y lectura de cada una; reproducción de semilla 42 vs E2; estado "E3 completo"; marcar pendientes de la sección 5.
- [ ] **Step 3:** `README.md`: tabla de fases, resultados E3, cómo correr el E3 y la demo (`streamlit run app/streamlit_app.py`). `results/README.md`: aclarar que los archivos `C*` son generados directamente por el código. `requirements.txt`: versiones fijas de `results/entorno.json`.
- [ ] **Step 4:** Informe final (skill `docx`): parte del informe de avance y añade Fase C, semillas, ablaciones, análisis de fallos y conclusiones; campos del equipo quedan como `[completar]`.
- [ ] **Step 5:** Run `.venv/Scripts/python -m pytest -q` → todos pasan. Commit + push — `git commit -m "E3: resultados, cuaderno, README e informe final"`.
