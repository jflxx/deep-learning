# Cuaderno de decisiones (design log)

**Proyecto:** Clasificación de cobertura del suelo con DINOv2 (EuroSAT RGB) · 1INF52 Deep Learning 2026-2
**Pregunta de investigación:** ¿Añadir adaptadores LoRA mejora la clasificación en regiones geográficas nuevas frente a usar DINOv2 congelado?
**Estado (E2, 21/10/2026):** Fase A completa · Fase B preliminar (partición interna, una semilla) · Fase C pendiente.

Reglas de este cuaderno (de la consigna):
1. Cada decisión de diseño se registra con su **justificación** y las **alternativas** consideradas.
2. Antes de cada ablación se escribe **qué esperamos y por qué**; después se contrasta con el resultado.
3. Se indica **dónde se usó asistencia de IA** y quién del equipo la revisó.

---

## 1. Decisiones

Cada decisión indica su fundamento: evidencia propia (medición en este proyecto), literatura [ref] o restricción práctica. Cuando una elección es convencional o arbitraria, se dice explícitamente. Las referencias están en la sección 6.

| ID | Fecha | Decisión | Alternativas consideradas | Fundamento | Uso de IA | Responsable |
|---|---|---|---|---|---|---|
| D01 | 07/10 | Tema: DINOv2 + datos satelitales (banco: "¿Sirven las features de DINOv2 fuera de ImageNet?") | SAM en dominio médico; DETR vs YOLO | Tema del banco oficial; modelo y datos abiertos (Apache-2.0 y MIT); cabe en una GPU gratuita; permite las 3 fases porque EuroSAT tiene una versión georreferenciada (MS) para la Fase C [1][2] | Propuesta E1 redactada con asistencia de IA | |
| D02 | 07/10 | Modelo base: **DINOv2 ViT-S/14 sin registros** (22,056,576 parámetros) | ViT-B/14 (86 M), ViT-L/14 (300 M), variantes con registros | **Restricción práctica:** es la variante más pequeña de la familia [1]; con 22 M de parámetros cabe con holgura en los 15 GB de una T4 y permite cachear embeddings. El artículo reporta que las features de ViT-S ya son competitivas en linear probing [1]. Variante sin registros por ser la publicada originalmente; la versión con registros [4] queda como comparación en E3 | — | |
| D03 | 07/10 | Dataset: **EuroSAT RGB** (27,000 imágenes, 10 clases, 64×64) | EuroSAT MS (13 bandas), BigEarthNet | DINOv2 se preentrenó con imágenes RGB de 3 canales [1]; las 13 bandas de MS requerirían cambiar la capa de entrada, que sería una segunda intervención. EuroSAT es un benchmark estándar de uso del suelo con licencia MIT [2] | — | |
| D04 | 20/10 | Implementación de **Hugging Face** (`facebook/dinov2-small`) en lugar de `torch.hub` | `torch.hub` `dinov2_vits14` | **Evidencia de código:** en `torch.hub`, Q, K y V están fusionadas en una sola capa `qkv`; aplicar LoRA solo a Q y V exigiría cortar su salida. La versión HF tiene los mismos pesos con Q, K y V separadas. Verificado: la carga reporta 223 tensores y 22,056,576 parámetros, igual que el modelo oficial | Detectado con asistencia de IA; revisado por el equipo | |
| D05 | 20/10 | Preprocesamiento: **redimensionar 64→224 (bicúbico) sin recorte** + normalización con media/desviación de ImageNet | Recorte central 256→224 (procesador por defecto de DINOv2); entrada a 112 px | Bicúbico y normalización de ImageNet son los del preprocesamiento original de DINOv2 [1]. Sin recorte porque el recorte central eliminaría el 23 % del área (256→224 por lado), y en EuroSAT la información está en toda la imagen (escenas, no objetos centrados). 224 px = 16×16 parches, la resolución de evaluación habitual. **Medido:** a 112 px el costo baja a 2.87 GFLOPs, pero cada parche cubriría 7×7 px originales (más información mezclada por token); queda como ablación posible | — | |
| D06 | 20/10 | Partición **E2: estratificada aleatoria 70/15/15, semilla 42** (18,900 / 4,050 / 4,050), guardada en `splits/` | Partición geográfica (planeada para Fase C); 80/10/10 | Estratificada para conservar la proporción de clases (Pasture tiene 2,000 imágenes frente a 3,000 de otras). 70/15/15 es una proporción convencional que deja 4,050 imágenes de test: suficiente para detectar diferencias de ~1 punto (error estándar de la accuracy ≈ 0.3 pts). La semilla 42 es **arbitraria** y se fija solo por reproducibilidad. Los JPG RGB no tienen coordenadas, así que la partición geográfica requiere EuroSAT MS (pendiente). **Esta partición no mide generalización geográfica** | — | |
| D07 | 20/10 | Representación para el clasificador: **[CLS ; media de tokens de parche]** (768 dims) tras la norma final | Solo CLS (384); solo media de parches | Es el protocolo de linear probing del artículo de DINOv2, que concatena el token de clase y el promedio de los tokens de parche [1]. Se usa igual en ambas condiciones para que la única diferencia sea la intervención | — | |
| D08 | 20/10 | Línea base con **embeddings cacheados** | Pasar las imágenes por el backbone en cada época | Matemáticamente equivalente cuando el backbone está congelado y no hay augmentation (la salida del backbone es la misma en cada época). **Medido:** 61 s de extracción frente a ~8 × 61 s si se recalculara | — | |
| D09 | 07/10 | Intervención: **LoRA r = 8, α = 8, dropout 0, en Q y V de los bloques 11 y 12** | LoRA en todos los bloques; también en K y en la salida; fine-tuning completo; rango 4/16 | **Q y V:** con un presupuesto fijo de parámetros, el artículo de LoRA obtuvo los mejores resultados adaptando W_q y W_v juntos [3]. **Rango 8:** el mismo artículo muestra que rangos bajos (1–8) ya capturan la adaptación necesaria [3]; 8 es un punto intermedio, no un óptimo demostrado (ablación de rango pendiente). **α = r:** escala α/r = 1, la actualización entra con el mismo peso que W sin hiperparámetro extra [3]. **Últimos 2 bloques:** las capas profundas son las más específicas de la tarea de preentrenamiento y las que más se benefician de adaptarse [5]; además, nuestro PCA (Fase A) muestra que los bloques finales son los que codifican información semántica de la escena. **Dropout 0:** sin evidencia de sobreajuste en validación que lo requiera. Resultado: 24,576 parámetros (0.11 % del backbone) | Método de [3] | |
| D10 | 20/10 | **LoRA implementado a mano** (clase `LoRALinear`, ~20 líneas) | Librería `peft` | Control total y explicable en la defensa oral. Sigue la formulación de [3]: W' = W + (α/r)·B·A, con B inicializada en 0 y A con inicialización Kaiming uniforme. **Verificado:** con B = 0 la diferencia con el modelo base es exactamente 0.00e+00, y el modelo fusionado da la misma salida que el no fusionado | Código escrito con asistencia de IA; revisado por el equipo | |
| D11 | 20/10 | **Mismo presupuesto** para ambas condiciones: AdamW, lr = 1e-3, weight decay = 1e-4, batch 64, 8 épocas, misma semilla, misma inicialización de la cabeza, selección por mejor macro-F1 en validación | Ajustar lr y épocas por separado para cada condición | **Control experimental:** una hipótesis, un cambio, una medición (consigna). **AdamW** desacopla el weight decay de la actualización adaptativa [6] y es el optimizador usado en [3]. **lr = 1e-3:** valor por defecto de Adam [6]; LoRA tolera lr más altos que el fine-tuning completo porque solo entrena matrices pequeñas que parten de B = 0 [3]. No se hizo búsqueda de lr (limitación declarada; se usa el mismo valor en ambas condiciones para no favorecer a ninguna). **Batch 64:** convencional; la memoria no es restricción (pico medido en inferencia: 344 MB a batch 32 y 833 MB a batch 128, frente a 15 GB de la T4). **8 épocas:** presupuesto fijado por tiempo de cómputo; **medido a posteriori:** desde la época 3 el macro-F1 de validación queda a menos de 0.5 pts de su máximo en ambas condiciones (base 0.957→0.962; LoRA 0.974→0.975), así que 8 épocas no recortan el aprendizaje. **Selección por validación** para no usar el test en ninguna decisión | — | |
| D12 | 20/10 | Métrica principal: **macro-F1**; secundarias: accuracy, F1 por clase, matriz de confusión | Solo accuracy; F1 ponderado | El macro-F1 da el mismo peso a cada clase, y las clases no están balanceadas (2,000 a 3,000 imágenes). La accuracy se reporta por comparabilidad con la literatura de EuroSAT [2]. Es la métrica que se comprometió en E1 | — | |
| D13 | 20/10 | FLOPs con `torch.utils.flop_counter` usando atención **eager**; convención 1 MAC = 2 FLOPs | `fvcore`/`thop` (reportan MACs) | **Evidencia propia:** con la atención optimizada (SDPA) el contador omitía la parte de atención (~10 %, 11.03 vs 12.25 GFLOPs); con eager el valor medido coincide exactamente con la fórmula analítica 12·(12ND² + 2N²D) + parches. Se declara la convención porque otras herramientas reportan la mitad | Detectado durante la prueba con asistencia de IA | |
| D14 | 20/10 | Latencia: calentamiento de 5 pasadas, 20 repeticiones, mediana, sincronización CUDA; batch 1/32/128, fp32 y fp16 | Una sola medición; media en lugar de mediana | Las operaciones CUDA son asíncronas: sin `synchronize` se mediría solo el lanzamiento de kernels. Las primeras pasadas incluyen inicialización. La mediana es robusta a valores atípicos (interrupciones del sistema). Tres tamaños de batch para separar la latencia (batch 1) del throughput (batch grande) | — | |
| D15 | 20/10 | Entorno: **Google Colab con GPU Tesla T4** (torch 2.11, transformers 5.20.1) | GPU local AMD RX 580 | PyTorch requiere CUDA (NVIDIA) o ROCm (AMD en Linux); la RX 580 (arquitectura Polaris) ya no está soportada por ROCm. Colab es gratuito y deja el entorno documentado en `results/entorno.json` | — | |
| D16 | 21/10 | Corrección: los parámetros A y B de LoRA se crean en el **mismo dispositivo y tipo de dato** que W | — | **Evidencia:** `RuntimeError` de dispositivo al ejecutar en GPU. La primera prueba del notebook se hizo solo en CPU, por eso no apareció antes. No cambia la metodología | Corregido con asistencia de IA | |
| D17 | 24/11 | Partición **geográfica por bloques espaciales** (K-means K=12 en EPSG:3035 con restricción de ≥5% de cada clase en cada partición, respaldo determinista [(12,0.05), (15,0.05), (20,0.05), (20,0.03)], test dentro y test fuera disjuntos) | Buffer circular, partición por tiles de Sentinel-2, K-means lat/lon sin proyectar | Proyección conforme EPSG:3035 preserva distancias euclidianas en Europa. Bloques espaciales evitan fuga de información espacial entre parches contiguos. Restricción 5% asegura presencia de las 10 clases en train, val y tests. Orden de respaldo fijo elimina arbitrariedades | Asistencia de IA en diseño de fallback; revisado por el equipo | |
| D18 | 24/11 | Arquitectura modular con paquete `src/eurosat_lora` editable y notebook E3 delgado | Código monolítico en un único notebook (estilo E2) | Facilita pruebas unitarias automáticas con pytest en CPU local antes de subir a Colab; notebook E3 solo orquesta llamadas, evitando estado global y celdas desordenadas | Código modular estructurado con asistencia de IA; revisado por el equipo | |
| D19 | 24/11 | Corridas **idempotentes** con guardado atómico de `metricas.json` y checkpoints por corrida en Drive | Guardar solo al final del notebook o en memoria | Permite reanudar experimentos interrumpidos por desconexiones de Colab sin reentrenar lo ya computado; atomic replace evita archivos corruptos por desconexión a mitad de escritura | Implementado con asistencia de IA; revisado por el equipo | |
| D20 | 24/11 | Matriz simétrica de ablaciones (r4/r16, 4 bloques 8–11, solo Q) en ambas particiones (interna y geográfica) | Evaluar ablaciones solo en partición interna | Permite responder si el impacto del rango, la profundidad y las proyecciones se mantiene invariante ante un domain shift geográfico severo (test fuera) o si la capacidad del adaptador interactúa con la generalización espacial | Propuesto por el equipo con asistencia de IA | |
| D21 | 25/11 | **Se conservan las 20 corridas de Colab** (ejecutadas el 25/11) pese a dos desviaciones del presupuesto del E2 detectadas después en revisión de código: (a) LoRA se entrenó en fp16 **sin `GradScaler`** (el E2 lo usaba); (b) la inicialización de A **no dependía de la semilla** de la corrida (se fijaba después de crear LoRA). Ambas se corrigieron en el código. Además: la partición geográfica se generó en la misma sesión, justo antes de entrenar, en lugar de commitearse antes (es determinista y se regeneró idéntica); se usó el **respaldo K = 15** porque con K = 12 ninguna asignación cumplía la restricción de clases | Repetir las 14 corridas LoRA (~2.5 h de GPU) | Restricción de tiempo. **Evidencia:** la validación de LoRA r8 semilla 42 interna (0.9746) reproduce la del E2 (0.9752), así que el entrenamiento no quedó frenado; las 6 corridas base no se ven afectadas (fp32, sin LoRA). Una **corrida de control** con el código corregido (`scripts/E3_control_correccion.py`, `results/C11_control_correccion.csv`; LoRA r8, geográfica, semilla 42) difiere de la original en −0.15 (val), +0.46 (test dentro) y −0.08 pts (test fuera); la ventaja de LoRA fuera pasa de +1.70 a +1.63 pts. **Las desviaciones no cambian los resultados.** Consecuencia declarada: las semillas no reproducen exactamente cada corrida LoRA | Revisión de código con asistencia de IA; decisión del equipo | |

---

## 2. Ablaciones (hipótesis antes, resultado después)

### A1 · LoRA (Q, V; bloques 11–12; r = 8) vs linear probing — partición interna

**Hipótesis registrada antes de ejecutar** (propuesta E1, 07/10/2026):
- **Hipótesis:** LoRA aumentará el macro-F1 respecto al linear probing.
- **Por qué:** las features de DINOv2 vienen de imágenes naturales; EuroSAT es vista cenital, a 64 px y con texturas en lugar de objetos. Adaptar la atención de los bloques finales podría ajustar la representación a ese dominio con pocos parámetros.
- **Riesgo previsto:** sobreajuste a la partición interna (y posible caída en regiones nuevas en la Fase C).
- **Predicción numérica** — ⚠️ *completada el 21/10/2026, **después** de ejecutar.* No se registró antes; se completa a pedido del equipo con un razonamiento basado **solo en la literatura y en el diseño** (no en nuestros resultados). Por haberse escrito después, **no tiene valor confirmatorio**: sirve para mostrar si el resultado es coherente con lo que la literatura permitía esperar. Las predicciones de las ablaciones siguientes (A2–A4, más abajo) sí están escritas antes.

| Magnitud | Predicción | Fundamento |
|---|---|---|
| Macro-F1 línea base (linear probing) | **0.94 – 0.97** | (a) **Techo de referencia:** el fine-tuning completo de una ResNet-50 en EuroSAT RGB alcanza 98.57 % de accuracy [2]; un clasificador lineal sobre features congeladas debería quedar por debajo. (b) DINOv2 reporta que sus features congeladas transfieren bien con linear probing a tareas de clasificación fuera de ImageNet [1]. (c) A 64 px reescalado a 224, cada parche de 14 px cubre solo ~4×4 px originales: información poco nítida que limita la representación. Por eso: alta, pero 2–4 pts debajo del techo |
| Δ macro-F1 (LoRA − base) | **+0.5 a +2.0 pts** | (a) La brecha disponible hasta el techo es de ~2–4 pts. (b) LoRA solo agrega 24,576 parámetros (0.11 %) en 2 de 12 bloques (D09): no puede recuperar toda la brecha del fine-tuning completo. (c) En [3], LoRA de rango bajo iguala o se acerca al fine-tuning completo, pero allí se aplicó en **todas** las capas; aquí solo en 2. (d) Cerca de 96–97 % hay efecto techo: quedan pocos errores y muchos son ambiguos (vegetación a 64 px) |
| Criterio de rechazo | **Δ < 0.6 pts**, o mejora en validación que no se mantiene en test | 0.6 pts ≈ 2 errores estándar del macro-F1 con 4,050 imágenes de test (√(0.96 × 0.04 / 4,050) ≈ 0.3 pts cada uno). Una diferencia menor no se distingue del ruido del test |

**Resultado** (`results/B1_comparacion.csv`, `results/B_resumen.json`; Colab, Tesla T4, semilla 42):

| Condición | Parámetros entrenables | Accuracy test | Macro-F1 test | Errores (de 4,050) | Mejor época (val) | Tiempo (s) |
|---|---|---|---|---|---|---|
| Linear probing | 7,690 | 0.9600 | 0.9580 | 162 | 6 | 64 (incluye 61 s de extracción) |
| LoRA r=8 (Q, V, bloques 11–12) | 32,266 | **0.9728** | **0.9714** | **110** | 7 | 454 |
| **Δ** | +24,576 | +1.28 pts | **+1.33 pts** | −52 (−32 %) | | ×7.1 |

- **Contraste con la predicción numérica:** base 0.958 (dentro de 0.94–0.97); Δ = +1.33 pts (dentro de +0.5 a +2.0 y por encima del umbral de 0.6). Coherente con lo esperable por la literatura; recordar que la predicción se escribió después.
- **¿Se cumplió la hipótesis?** **Sí, en la partición interna.** LoRA mejora el macro-F1 en 1.33 puntos y reduce los errores un 32 %. La mejora en validación (0.9618 → 0.9752) se mantiene en test (0.9580 → 0.9714), así que no es un artefacto de la selección de época.
- **¿Es mayor que el ruido?** Test de McNemar [8] en el **peor caso posible** (sin las predicciones pareadas, suponiendo que ningún error es compartido): χ² = 9.56, **p ≈ 0.002**. Solo cubre la variabilidad del conjunto de test, **no** la del entrenamiento: con una sola semilla no sabemos cuánto cambia Δ al reentrenar.
- **Las mejoras se concentran en estructuras lineales** (`results/B2_f1_por_clase.csv`, `B4_confusiones_principales.csv`):
  - **Highway: +5.85 pts de F1** y **River: +3.71 pts**, las dos mayores mejoras.
  - La confusión **River → Highway pasa de 21 a 2 errores**. Ríos y carreteras son estructuras delgadas y alargadas; con features genéricas se parecen, y adaptar la atención parece ayudar a separarlas (por ejemplo, por textura y color del borde).
- **No todo mejora; aparecen confusiones nuevas entre vegetación:**
  - Pasture → AnnualCrop: 2 → 10; Pasture → HerbaceousVegetation: 7 → 12; PermanentCrop → HerbaceousVegetation: 1 → 13.
  - El **recall** de Pasture baja (280 → 272 aciertos) y el de PermanentCrop también (352 → 343). Su F1 sube igual (+0.46 y +0.74 pts) porque gana precisión.
  - Lectura: LoRA redistribuye los errores hacia las clases de vegetación, que son visualmente las más parecidas a 64 px. Son el foco natural de la Fase C.
- **Sobreajuste:** la pérdida de entrenamiento de LoRA llega a ~0.0003, pero el macro-F1 de validación se mantiene estable (0.972–0.975 desde la época 3). No hay caída en validación en 8 épocas, aunque el modelo ya "memoriza" el train.
- **Costo de la intervención** (`results/B3_costo_lora.csv`):
  - +24,576 parámetros (0.11 % del backbone).
  - Entrenamiento 7× más lento: hay que pasar cada imagen por el backbone en cada época, en lugar de reutilizar embeddings cacheados.
  - **Inferencia sin costo:** 4.43 (base) vs 4.35 (LoRA sin fusionar) vs 4.41 ms/img (fusionado), batch 32, todo dentro del ruido de medición. Al fusionar B·A en W el modelo es idéntico en tamaño y costo al original.
- **Limitaciones:** una sola semilla; partición aleatoria (no geográfica), así que todavía **no responde la pregunta de investigación**, que es sobre regiones nuevas; no hubo búsqueda de hiperparámetros.

### Predicciones registradas para las próximas ablaciones (escritas el 21/10/2026, ANTES de ejecutarlas)

Para cada una: predicción numérica, razonamiento y criterio de rechazo. Al ejecutarlas se completa la columna "Resultado" sin modificar lo demás.

| ID | Ablación (un solo cambio) | Predicción | Razonamiento | Se rechaza si | Resultado |
|---|---|---|---|---|---|
| A2 | Repetir A1 con 3 semillas (42, 0, 1) | Δ macro-F1 (LoRA − base) = **+1.0 a +1.6 pts** en las 3 semillas; desviación estándar de Δ ≤ 0.3 pts | Con 4,050 imágenes de test el error estándar del macro-F1 es ~0.3 pts; la mejora de A1 (+1.33) es ~4 veces eso, así que debería repetirse | Alguna semilla da Δ ≤ 0, o la desviación > 0.5 pts | |
| A3 | Rango de LoRA: r = 4 y r = 16 (α = r, resto igual) | Ambos dentro de **±0.3 pts** del macro-F1 de r = 8 | [3] reporta que rangos bajos ya capturan la adaptación y que subir el rango da rendimientos decrecientes | r = 16 supera a r = 8 por más de 0.5 pts, o r = 4 queda más de 0.5 pts por debajo | |
| A4 | Fase C: evaluar ambas condiciones en regiones geográficas reservadas | Caída del macro-F1 (interno − geográfico) de **3 a 10 pts** en ambas; la ventaja de LoRA se **reduce a ≤ +0.5 pts** o se invierte | El cambio de región altera textura y color de cultivos y vegetación; las clases de vegetación ya son las más confundidas en A1, y la pérdida de entrenamiento de LoRA llega a ~0.0003 (se ajusta mucho a la distribución interna) | La caída es < 1 pt (no hay shift real) o LoRA mantiene ≥ +1.3 pts | |
| A5 | Bloques LoRA: 4 últimos bloques (8–11) vs 2 últimos (10–11) en partición interna | macro-F1 dentro de **+0.1 a +0.5 pts** superior a r8 en bloques 10–11 | [5] indica que capas profundas son más específicas, pero adaptar 4 bloques da mayor capacidad sin exceder el presupuesto (49,152 params) | macro-F1 empeora respecto a 2 bloques por > 0.3 pts o sube > 0.8 pts | |
| A6 | Proyecciones LoRA: solo Q vs Q+V en bloques 10–11 en partición interna | macro-F1 cae de **0.2 a 0.6 pts** respecto a Q+V | [3] encontró que adaptar W_q y W_v juntos rinde mejor que solo W_q con igual rango | solo Q supera a Q+V o cae más de 1.0 pt | |
| A3g | Rango de LoRA (r=4, r=16) en partición geográfica | Test dentro: dentro de **±0.3 pts** de r=8. Test fuera: caída geográfica similar a r=8 (3 a 10 pts) | [3] rangos bajos saturan la adaptación; subir rango no previene desajuste ante cambio de distribución | r=16 reduce la caída geográfica en > 1.5 pts frente a r=8 | |
| A5g | Bloques LoRA (8–11) en partición geográfica | Test dentro: +0.1 a +0.4 pts. Test fuera: **mayor caída geográfica (0.5 a 1.5 pts más de caída)** que con 2 bloques | Adaptar bloques más tempranos altera features genéricas que transfieren mejor congeladas [5] | 4 bloques cae menos que 2 bloques por > 0.5 pts | |
| A6g | Proyecciones LoRA (solo Q) en partición geográfica | Test dentro: −0.2 a −0.5 pts. Test fuera: **caída similar o ligeramente menor** (0.2 a 0.5 pts menor caída) que Q+V | Menor capacidad específica reduce sobreajuste a las regiones de entrenamiento [3] | En test fuera, solo Q cae > 1.5 pts por debajo de base | |

### Regla de decisión de A4 (fijada el 24/11/2026, ANTES de ejecutar la Fase C)

Por semilla (42, 0, 1) y condición:
- $\text{caída} = \text{macro-F1}(\text{test dentro}) - \text{macro-F1}(\text{test fuera})$
- $\text{ventaja LoRA} = \text{macro-F1}(\text{LoRA}) - \text{macro-F1}(\text{base})$, dentro y fuera.

Veredicto con la **media de las 3 semillas**:

| Veredicto | Condición |
|---|---|
| **Se cumple A4** | Caída media de 3 a 10 pts en ambas condiciones **y** ventaja media de LoRA fuera $\le$ +0.5 pts (o negativa) |
| **Se rechaza A4** | Caída media < 1 pt en alguna condición **o** ventaja media de LoRA fuera $\ge$ +1.3 pts |
| **Intermedio** | Cualquier otro caso; se reporta tal cual |

- Se afirma "LoRA mejor/peor fuera" solo si las 3 semillas tienen el mismo signo; si no, "sin diferencia consistente".
- McNemar exacto por semilla (dentro y fuera) se reporta como apoyo; no entra en el veredicto.
- Se reportan media ± desviación estándar de caída y ventaja.

---

## 3. Hallazgos del teardown (Fase A)

| Medición | Resultado | Lectura / interpretación |
|---|---|---|
| Parámetros totales | **22,056,576** (88.2 MB fp32) | |
| Parámetros por bloque | **1,775,232** en cada uno de los 12 bloques: MLP 1,181,568 (67 %) · atención 591,360 (33 %) · normas 1,536 · layerscale 768 | Los 12 bloques son idénticos; el MLP (384→1536→384) concentra 2/3 de cada bloque. LoRA en Q y V toca una parte pequeña de la atención |
| Embeddings | 753,024 (la tabla de posiciones es de 1,370 tokens para 518 px y se interpola a 257 tokens para 224 px) | La tabla de posiciones pesa más que la proyección de parches |
| FLOPs a 224 px | **12.25 GFLOPs** por imagen (medido = analítico); 1.011 por bloque, del cual la atención cuadrática es 0.101 (**9.9 %**) | Costo igual en todos los bloques: un ViT no reduce resolución como una CNN, todos procesan 257 tokens |
| FLOPs vs resolución | 112 px: 2.87 G (2.7 % atención) · 224 px: 12.25 G (9.9 %) · 448 px: 63.36 G (**30.6 %**) | Duplicar la resolución multiplica el costo ×5.2 y la parte cuadrática gana peso. Subir EuroSAT de 64 a 224 px cuesta ~12 GFLOPs por imagen para información que cabía en 64×64 |
| Latencia (T4, fp32) | batch 1: **8.9 ms** (112 img/s) · batch 32: 3.8 ms/img (263 img/s) · batch 128: 3.85 ms/img | Con batch 1 la GPU está subutilizada; el throughput se satura desde batch 32 en fp32 |
| Latencia (T4, fp16) | batch 32: **1.25 ms/img** (801 img/s) · batch 128: 1.17 ms/img (**854 img/s**) | fp16 da ×3.3 de throughput (tensor cores de la T4) |
| Anomalía | fp16 con batch 1: **39.5 ms** (más lento que fp32) | No esperado. Hipótesis a verificar: sobrecarga de los casteos de autocast por operación, que domina con batch 1, o calentamiento insuficiente de los kernels fp16. Se reporta tal cual |
| Memoria pico (T4) | 187 MB (batch 1) · 344 MB (batch 32) · 833 MB (batch 128, fp32) · 641 MB (batch 128, fp16); pesos: 84 MB | La memoria crece con el batch por las activaciones, no por los pesos |
| PCA de tokens (`figures/A4_pca_tokens.png`) | **Bloque 4:** el color sigue la posición del parche (gradiente de arriba a abajo, igual en todas las imágenes). **Bloque 12:** cada imagen toma un color casi uniforme distinto por clase (Forest rosado, HerbaceousVegetation azul, AnnualCrop verde) y aparecen regiones con sentido (carretera vs bosque en Highway; zona industrial) | Los bloques iniciales codifican sobre todo posición y apariencia local; los finales, información semántica de la escena completa. Justifica intervenir en los últimos bloques |
| Tokens atípicos | En el bloque 8 (PCA) y en la atención del CLS del bloque 12 aparecen **parches aislados de valor muy alto** que no corresponden a nada visual (por ejemplo, una esquina en Forest) | Consistente con los "artefactos" de alta norma descritos para DINOv2 **sin registros** [4]. **Por verificar** midiendo la norma de los tokens; motiva comparar con `dinov2_vits14_reg` |
| Atención del CLS (bloque 12) | Dispersa en casi toda la imagen, con picos en pocos parches; no se centra en un objeto | Esperable en escenas de textura sin objeto central; los picos aislados coinciden con los tokens atípicos |
| Campo receptivo efectivo (`results/A5_erf.csv`, 16 imágenes) | r50: **69 → 57 → 67 → 85 px** y r90: 117 → 120 → 122 → 127 px (bloques 1, 4, 8, 12). Masa en el parche central: 9.3 % → **16.1 %** → 7.2 % → 3.3 % | Medido con el método de [7]. A diferencia de una CNN, el ERF ya es **global desde el bloque 1** (solo 9 % del gradiente en el propio parche). El bloque 4 es el más local, y luego se vuelve a expandir hasta cubrir más de la mitad del radio de la imagen en el bloque 12. El patrón de cuadrícula refleja los límites de los parches de 14 px |

---

## 4. Uso de asistencia de IA

| Fecha | Herramienta | Para qué | Qué revisó o cambió el equipo |
|---|---|---|---|
| 07/10 | | Propuesta E1 (ilustraciones generadas, prompts en `imagenes/prompts.json`) | |
| 20/10 | Claude (Anthropic) | Estructura del notebook E2, implementación de `LoRALinear`, medición de FLOPs/latencia/ERF, borrador de este cuaderno | |
| 21/10 | Claude (Anthropic) | Corrección del error de dispositivo (D16); extracción de resultados y figuras de las salidas del notebook; borrador del análisis de las secciones 2 y 3 | |
| 28/10 | Claude (Anthropic) | Informe de avance (Word), presentación y diagrama de Gantt (`docs/`) | |
| 18/11 | Claude (Anthropic) | Póster WAIMLAp corregido según los revisores (pipeline, diagramas, ejemplos de clasificación), celda `scripts/C1_ejemplos_prediccion.py`, abstract en LaTeX con referencias verificadas | |
| 24/11 | Claude / Antigravity | Diseño y especificación del plan E3: generalización de LoRA, partición geográfica por bloques espaciales (K-means EPSG:3035), matriz simétrica de 20 experimentos, suite de pruebas unitarias y script de humo | |
| 25/11 | Claude / Antigravity | Implementación modular del paquete `eurosat_lora`, corrección de determinismo en tests, resolución de compatibilidad transformers 5.x y verificación de los 36 tests unitarios en CPU | |

> Completar la última columna: qué verificó o cambió el equipo en cada caso (por ejemplo, "revisamos la fórmula de LoRA y la comprobación B = 0", "verificamos los conteos de la matriz de confusión").

---

## 5. Pendientes para E3
- [ ] Partición **geográfica**: vincular EuroSAT RGB con EuroSAT MS (GeoTIFF georreferenciados) y fijar regiones de prueba **antes** de entrenar.
- [ ] Fase C: evaluar ambas condiciones en regiones reservadas; reportar la caída de macro-F1 (interno − geográfico) y analizar fallos, con foco en las clases de vegetación (Pasture, PermanentCrop, HerbaceousVegetation), donde LoRA ya mostró nuevas confusiones.
- [ ] **3 semillas** para estimar la variación de Δ entre entrenamientos (y guardar las predicciones de test para un McNemar exacto).
- [ ] Ablaciones adicionales, cada una con predicción numérica previa: rango (4/8/16), bloques (2 vs 4 últimos), Q vs Q+V.
- [ ] Verificar los tokens atípicos (norma por token) y, opcionalmente, comparar con `dinov2_vits14_reg`.
- [ ] Investigar la anomalía de latencia fp16 con batch 1.
- [ ] Volver a ejecutar para guardar `checkpoints/` (quedaron en la sesión de Colab) y armar la demo Streamlit.

---

## 6. Referencias
1. Oquab, M. et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision.* arXiv:2304.07193.
2. Helber, P., Bischke, B., Dengel, A., Borth, D. (2019). *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.* IEEE JSTARS 12(7), 2217–2226.
3. Hu, E. J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.
4. Darcet, T., Oquab, M., Mairal, J., Bojanowski, P. (2023). *Vision Transformers Need Registers.* arXiv:2309.16588.
5. Yosinski, J., Clune, J., Bengio, Y., Lipson, H. (2014). *How transferable are features in deep neural networks?* NeurIPS.
6. Loshchilov, I., Hutter, F. (2019). *Decoupled Weight Decay Regularization.* ICLR.
7. Luo, W., Li, Y., Urtasun, R., Zemel, R. (2016). *Understanding the Effective Receptive Field in Deep Convolutional Neural Networks.* NeurIPS.
8. McNemar, Q. (1947). *Note on the sampling error of the difference between correlated proportions or percentages.* Psychometrika 12(2), 153–157.
