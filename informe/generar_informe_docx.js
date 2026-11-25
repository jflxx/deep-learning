// Genera el informe final en Word con formato tipo IEEE conference (A4, dos columnas).
// Requiere: npm install docx. Uso desde la raíz del repo: node informe/generar_informe_docx.js .
// Uso: node informe/generar_informe_docx.js <raiz_repo> [--sin-control]
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  AlignmentType, WidthType, BorderStyle, SectionType, Footer, PageNumber,
} = require("docx");

const RAIZ = process.argv[2];
const FIG = (f) => path.join(RAIZ, "figures", f);
const SIN_CONTROL = process.argv.includes("--sin-control");
const SALIDA = path.join(RAIZ, "informe", SIN_CONTROL
  ? "Informe_final_DINOv2_EuroSAT_LoRA_sin_control.docx"
  : "Informe_final_DINOv2_EuroSAT_LoRA.docx");

const FONT = "Times New Roman";
const BODY = 20; // 10 pt (IEEE)
const COL_W = 4696; // ancho de columna en DXA (A4, márgenes 1.9 cm, separación 0.25")
const FULL_W = 9752;

// ---------- texto con marcado simple: **negrita**, *cursiva*, {{pendiente}}
function runs(texto, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|\{\{[^}]+\}\}|~[^~]+~)/g;
  let i = 0, m;
  while ((m = re.exec(texto))) {
    if (m.index > i) out.push(new TextRun({ text: texto.slice(i, m.index), font: FONT, size: BODY, ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, size: BODY, ...base }));
    else if (t.startsWith("~")) out.push(new TextRun({ text: t.slice(1, -1), subScript: true, font: FONT, size: BODY, ...base }));
    else if (t.startsWith("{{")) out.push(new TextRun({ text: `[PENDIENTE: ${t.slice(2, -2)}]`, color: "C00000", bold: true, font: FONT, size: BODY, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, font: FONT, size: BODY, ...base }));
    i = m.index + t.length;
  }
  if (i < texto.length) out.push(new TextRun({ text: texto.slice(i), font: FONT, size: BODY, ...base }));
  return out;
}
const P = (texto, opts = {}) => new Paragraph({
  children: runs(texto, opts.run || {}),
  alignment: opts.align ?? AlignmentType.JUSTIFIED,
  indent: opts.noIndent ? undefined : { firstLine: 202 },
  spacing: { after: 40, line: 240 },
  keepNext: opts.keepNext,
});

const ROMANOS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"];
let nSec = 0, nSub = 0;
function H1(t) {
  nSub = 0;
  const pre = t === "Referencias" || t.startsWith("Reproducibilidad") ? "" : `${ROMANOS[nSec++]}. `;
  return new Paragraph({
    children: [new TextRun({ text: pre + t.toUpperCase(), font: FONT, size: BODY, smallCaps: false })],
    alignment: AlignmentType.CENTER, spacing: { before: 200, after: 80 }, keepNext: true,
  });
}
function H2(t) {
  const letra = String.fromCharCode(65 + nSub++);
  return new Paragraph({
    children: [new TextRun({ text: `${letra}. ${t}`, italics: true, font: FONT, size: BODY })],
    spacing: { before: 120, after: 60 }, keepNext: true,
  });
}

// ---------- figuras
function tamanoPng(f) {
  const b = fs.readFileSync(f);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
let nFig = 0, nTab = 0;
function figura(archivo, pie, anchoDxa) {
  const { w, h } = tamanoPng(FIG(archivo));
  const anchoPx = Math.round(anchoDxa / 1440 * 96);
  nFig++;
  return [
    new Paragraph({
      children: [new ImageRun({ type: "png", data: fs.readFileSync(FIG(archivo)),
        transformation: { width: anchoPx, height: Math.round(anchoPx * h / w) },
        altText: { title: `Fig. ${nFig}`, description: pie, name: archivo } })],
      alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 }, keepNext: true,
    }),
    new Paragraph({
      children: [new TextRun({ text: `Fig. ${nFig}. `, font: FONT, size: 16, bold: true }), ...runs(pie, { size: 16 })],
      alignment: AlignmentType.JUSTIFIED, spacing: { after: 160 },
    }),
  ];
}

// ---------- tablas (bordes horizontales tipo booktabs)
const NADA = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const LINEA = (sz) => ({ style: BorderStyle.SINGLE, size: sz, color: "000000" });
function tabla(titulo, encabezado, filas, anchos) {
  nTab++;
  const total = anchos.reduce((a, b) => a + b, 0);
  const celda = (txt, j, i, n) => new TableCell({
    width: { size: anchos[j], type: WidthType.DXA },
    margins: { top: 30, bottom: 30, left: 60, right: 60 },
    borders: {
      top: i === 0 ? LINEA(8) : (i === 1 ? LINEA(4) : NADA),
      bottom: i === n - 1 ? LINEA(8) : NADA, left: NADA, right: NADA,
    },
    children: [new Paragraph({ children: runs(txt, { size: 16, bold: i === 0 || undefined }), keepNext: i < n - 1,
      alignment: j === 0 ? AlignmentType.LEFT : AlignmentType.CENTER })],
  });
  const todas = [encabezado, ...filas];
  return [
    new Paragraph({
      children: [new TextRun({ text: `TABLA ${ROMANOS[nTab - 1]}`, font: FONT, size: 16 })],
      alignment: AlignmentType.CENTER, spacing: { before: 120, after: 0 }, keepNext: true,
    }),
    new Paragraph({ children: runs(titulo, { size: 16 }), alignment: AlignmentType.CENTER, spacing: { after: 60 }, keepNext: true }),
    new Table({
      width: { size: total, type: WidthType.DXA }, columnWidths: anchos, alignment: AlignmentType.CENTER,
      rows: todas.map((fila, i) => new TableRow({ cantSplit: true,
        children: fila.map((t, j) => celda(t, j, i, todas.length)) })),
    }),
    new Paragraph({ children: [], spacing: { after: 100 } }),
  ];
}

// ================== CONTENIDO (idéntico al .tex) ==================
const titulo = [
  new Paragraph({
    children: [new TextRun({ text: "¿Adaptar o congelar? LoRA sobre DINOv2 para clasificar cobertura del suelo en regiones geográficas no vistas", font: FONT, size: 44 })],
    alignment: AlignmentType.CENTER, spacing: { after: 200 },
  }),
  new Paragraph({ children: [new TextRun({ text: "Alfredo Jiménez, Jhonatan Davila, Italo Ramos", font: FONT, size: 22 })], alignment: AlignmentType.CENTER, spacing: { after: 40 } }),
  new Paragraph({ children: runs("*Facultad de Ciencias e Ingeniería, Pontificia Universidad Católica del Perú (PUCP), Lima, Perú*", { size: 18 }), alignment: AlignmentType.CENTER }),
  new Paragraph({ children: runs("1INF52 Deep Learning, 2026-2 · Grupo 4", { size: 18 }), alignment: AlignmentType.CENTER, spacing: { after: 240 } }),
];

const resumen = [
  new Paragraph({
    children: [
      new TextRun({ text: "Resumen—", bold: true, italics: true, font: FONT, size: 18 }),
      ...runs("Los modelos fundacionales de visión como DINOv2 se suelen transferir a imágenes satelitales con el backbone congelado (*linear probing*). Adaptar la representación con LoRA mejora la clasificación en la distribución de entrenamiento, pero podría ajustarse a las regiones vistas y perder ventaja en regiones nuevas. Estudiamos esa hipótesis (H2) en EuroSAT RGB con DINOv2 ViT-S/14 y LoRA de rango 8 en las proyecciones Q y V de los dos últimos bloques (24,576 parámetros, 0.11 % del backbone). Construimos una partición geográfica por bloques espaciales a partir de las coordenadas de EuroSAT MS, que permite evaluar el mismo modelo en regiones vistas (*test dentro*) y reservadas (*test fuera*), y fijamos la regla de decisión antes de ejecutar. Con 3 semillas, ambos modelos caen al pasar a regiones nuevas (7.2 pts de macro-F1 el congelado, 5.9 pts LoRA), pero la ventaja de LoRA **aumenta** de +0.9 pts dentro a +2.2 pts fuera, con el mismo signo en las 3 semillas y McNemar exacto p < 10⁻⁶ en cada una. H2 se rechaza: en este caso, adaptar los bloques profundos corrige un desajuste de dominio común a todas las regiones en lugar de sobreajustarse a las de entrenamiento. La mejora se concentra en vegetación herbácea, cultivos permanentes, carreteras y ríos.", { size: 18, bold: true }),
    ],
    alignment: AlignmentType.JUSTIFIED, spacing: { after: 80 },
  }),
  new Paragraph({
    children: [new TextRun({ text: "Palabras clave—", bold: true, italics: true, font: FONT, size: 18 }),
      ...runs("DINOv2, LoRA, ajuste eficiente en parámetros, teledetección, EuroSAT, generalización geográfica, validación espacial.", { size: 18, bold: true })],
    alignment: AlignmentType.JUSTIFIED, spacing: { after: 160 },
  }),
];

const cuerpo1 = [
  H1("Introducción"),
  P("DINOv2 [1] es un Vision Transformer (ViT) [2] entrenado sin etiquetas sobre 142 millones de imágenes naturales; sus autores reportan que sus *features* congeladas transfieren bien con un clasificador lineal. Las imágenes Sentinel-2 son distintas: vista cenital, unos 10 m por píxel, texturas repetitivas y ningún objeto central. Entre las opciones para adaptar el modelo, LoRA [3] entrena solo una actualización de rango bajo sobre pesos congelados y, una vez fusionada, no cambia el costo de inferencia."),
  P("Un clasificador de cobertura del suelo se entrena con unas regiones y se usa en otras, donde cambian cultivos, suelo y vegetación. Una partición aleatoria de EuroSAT [4] sobreestima esa generalización, porque teselas vecinas, casi idénticas, caen en entrenamiento y prueba [5]. Nuestra pregunta es: **¿añadir LoRA a DINOv2 mejora la clasificación en regiones geográficas no vistas frente a DINOv2 congelado?** Planteamos dos hipótesis: H1, LoRA mejora el macro-F1 en la partición interna; y H2, en regiones reservadas ambos modelos caen y la ventaja de LoRA se reduce, porque parte de ella vendría de ajustarse a las regiones de entrenamiento."),
  P("Contribuciones: (i) un análisis del costo y de las representaciones de DINOv2 ViT-S/14 que motiva dónde intervenir; (ii) una partición geográfica por bloques espaciales de EuroSAT, construida a partir de las coordenadas de EuroSAT MS, con un *test dentro* y un *test fuera* para el mismo modelo; (iii) una comparación controlada con 3 semillas, McNemar exacto y una regla de decisión fijada antes de ejecutar; y (iv) ablaciones de rango, profundidad y proyecciones en ambas particiones, con predicciones registradas antes de correrlas. El código, las particiones, las 20 corridas y el cuaderno de decisiones están en el repositorio del proyecto."),

  H1("Trabajos relacionados"),
  P("**Transferencia de modelos fundacionales.** El protocolo de *linear probing* de DINOv2 concatena el token de clase y la media de los tokens de parche [1]. Darcet *et al.* [6] describen tokens de norma atípica en ViT sin registros. **Ajuste eficiente en parámetros.** LoRA [3] obtiene sus mejores resultados, con presupuesto fijo, adaptando W~q~ y W~v~; Yosinski *et al.* [7] muestran que las capas profundas son las más específicas de la tarea original. **EuroSAT y validación espacial.** EuroSAT [4] tiene 27,000 teselas de 34 países; una ResNet-50 ajustada por completo alcanza 98.57 % de *accuracy* en RGB con partición aleatoria. Roberts *et al.* [5] recomiendan bloques espaciales para datos con autocorrelación espacial, que es la estrategia que seguimos."),

  H1("Metodología"),
  H2("Datos y particiones"),
  P("Usamos EuroSAT RGB (10 clases, 64×64 px). Cada tesela se redimensiona a 224 px (bicúbico, sin recorte central, que eliminaría el 23 % de la escena) y se normaliza con las estadísticas de ImageNet, como en el preentrenamiento."),
  P("**Partición interna:** estratificada 70/15/15 con semilla 42 (18,900/4,050/4,050), idéntica a la del E2."),
  P("**Partición geográfica:** las teselas RGB no tienen coordenadas, pero sus pares de EuroSAT MS (mismo nombre de archivo) son GeoTIFF georreferenciados. Verificamos el emparejamiento 1:1 de las 27,000 teselas y tomamos el centro de cada una en WGS84. Proyectamos a ETRS89-LAEA (EPSG:3035) y agrupamos con K-means. Regiones completas, distintas entre sí, forman la validación y el *test fuera* (cada una entre 12 y 18 % del total); cada clase debe tener al menos el 5 % de sus imágenes en ambas. Entre las asignaciones válidas de 20,000 intentos se elige la más cercana a 15/15. De las regiones restantes se separa un *test dentro* aleatorio estratificado (10 % del total); el resto es entrenamiento. El orden de respaldo si no hay solución (K = 12 → 15 → 20, luego mínimo 3 %) se fijó antes de ejecutar; con K = 12 no hubo asignación válida y se usó K = 15. Resultado: 15,383 entrenamiento, 4,754 validación, 2,700 *test dentro* y 4,163 *test fuera*. El *test fuera* son tres regiones: dos en Europa central (centros aprox. 18.5°E 52.2°N y 16.9°E 48.0°N) y una en el Mediterráneo oriental (33.0°E 34.9°N). La distancia mediana de una imagen de *test fuera* a la imagen de entrenamiento más cercana es 492 km (mínimo 12.4 km), frente a 0.9 km en *test dentro*."),
  ...figura("C0_mapa_regiones.png", "Partición geográfica por bloques espaciales (K = 15). Gris: regiones de entrenamiento (de ellas sale el test dentro); azul: validación; naranja: test fuera.", COL_W),
  H2("Modelo y análisis del modelo base"),
  P("DINOv2 ViT-S/14 sin registros (facebook/dinov2-small) tiene 12 bloques iguales (d = 384, 6 cabezas, MLP de 1,536). La Tabla I resume su costo medido en una Tesla T4; los FLOPs medidos coinciden con la fórmula analítica. Un PCA de los tokens muestra que el bloque 4 codifica sobre todo la posición del parche, mientras que en el bloque 12 los tokens se agrupan por clase de escena; el campo receptivo efectivo [8] es global desde el bloque 1. Esto motiva intervenir en los bloques finales."),
  ...tabla("Análisis del modelo base (Fase A), DINOv2 ViT-S/14 en una Tesla T4.",
    ["Medición", "Valor"],
    [["Parámetros totales", "22,056,576 (88.2 MB en fp32)"],
     ["Parámetros por bloque (×12)", "1,775,232 (MLP 67 %, atención 33 %)"],
     ["GFLOPs por imagen a 112 / 224 / 448 px", "2.87 / 12.25 / 63.36"],
     ["Atención cuadrática en el total", "2.7 / 9.9 / 30.6 %"],
     ["Latencia fp32, batch 1", "8.89 ms (112 img/s)"],
     ["Throughput fp32, batch 32", "263 img/s (3.80 ms/img)"],
     ["Throughput fp16, batch 128", "854 img/s (1.17 ms/img)"],
     ["Memoria pico fp32, batch 1 / 128", "187 / 833 MB"],
     ["Campo receptivo efectivo r50, bloques 1/4/8/12", "69 / 57 / 67 / 85 px"]],
    [2296, 2400]),
  H2("Condiciones comparadas"),
  P("**Línea base (linear probing):** backbone congelado; representación [CLS; media de parches] tras la norma final (768 dimensiones) y una capa lineal de 7,690 parámetros, entrenada sobre *embeddings* precalculados. **LoRA:** en Q y V de los bloques 11 y 12, W′ = W + (α/r)·B·A, con r = α = 8, B = 0 al inicio (el modelo parte exactamente del congelado) y A con inicialización Kaiming uniforme; 24,576 parámetros entrenables más la cabeza. Tras entrenar, B·A se fusiona en W sin costo de inferencia. **Presupuesto idéntico:** AdamW [9] (tasa 10⁻³, *weight decay* 10⁻⁴), batch 64, 8 épocas, selección por el mejor macro-F1 de validación; precisión mixta fp16 en GPU. El conjunto de prueba no interviene en ninguna decisión."),
  H2("Protocolo de evaluación"),
  P("Métrica principal: macro-F1 (las clases tienen entre 2,000 y 3,000 imágenes). Cada comparación base vs. LoRA se repite con las semillas 42, 0 y 1 (las particiones no cambian con la semilla). La significancia frente al ruido del conjunto de prueba se mide con McNemar exacto [10] sobre predicciones pareadas. Para H2 definimos, por semilla y condición, la *caída* = macro-F1(*test dentro*) − macro-F1(*test fuera*) y la *ventaja* de LoRA = macro-F1(LoRA) − macro-F1(base). La regla, registrada el 24/11/2026 antes de ejecutar, usa la media de las 3 semillas: *se cumple* la predicción de H2 si la caída está entre 3 y 10 pts en ambas condiciones y la ventaja fuera es ≤ +0.5 pts; *se rechaza* si alguna caída es < 1 pt o la ventaja fuera es ≥ +1.3 pts; en otro caso es intermedio. Solo se afirma una dirección si las 3 semillas tienen el mismo signo."),
  H2("Ablaciones"),
  P("Con la semilla 42 y en ambas particiones: rango r ∈ {4, 16} (α = r), LoRA en los 4 últimos bloques (9–12) y solo Q. Cada una tiene una predicción numérica y un criterio de rechazo registrados antes de ejecutar. Con una semilla, diferencias menores a ~0.6 pts (partición interna) no se distinguen del ruido."),

  H1("Resultados"),
  H2("Partición interna (H1)"),
  P("La Tabla II resume las 3 semillas. LoRA mejora el macro-F1 en las tres (+1.19, +1.32 y +2.06 pts; media +1.52 ± 0.47) con McNemar p ≤ 2×10⁻⁴ en cada semilla. H1 se confirma."),
  ...tabla("Macro-F1 (%) por conjunto de prueba, media ± desviación de 3 semillas. Δ = LoRA − base.",
    ["Conjunto", "Base", "LoRA", "Δ"],
    [["Interna (test aleatorio)", "95.66 ± 0.19", "97.18 ± 0.39", "+1.52 ± 0.47"],
     ["Geográfica, test dentro", "96.40 ± 0.22", "97.32 ± 0.19", "+0.92 ± 0.36"],
     ["Geográfica, test fuera", "89.25 ± 0.88", "91.45 ± 0.57", "+2.20 ± 1.00"]],
    [1636, 1020, 1020, 1020]),
  H2("Regiones no vistas (H2)"),
  P("Ambos modelos pierden rendimiento en regiones reservadas (Fig. 2): la caída media es 7.15 ± 0.68 pts para la base y 5.87 ± 0.40 pts para LoRA. El cambio de distribución es real y similar en magnitud a lo previsto (3–10 pts). Sin embargo, la ventaja de LoRA no se reduce: pasa de +0.92 pts en *test dentro* a +2.20 pts en *test fuera* (+1.70, +3.35 y +1.56 por semilla), siempre positiva, con McNemar exacto p < 10⁻⁶ en cada semilla (Tabla III). Como la ventaja fuera supera el umbral de +1.3 pts, **la predicción de H2 se rechaza**. Comparando de forma secundaria con los modelos de la partición interna, la diferencia entre su test aleatorio y el *test fuera* de los modelos geográficos es de 6.41 pts (base) y 5.73 pts (LoRA): la partición aleatoria sobreestima en unos 6 pts el rendimiento en regiones nuevas."),
  ...figura("C4_interno_vs_geo.png", "Macro-F1 por conjunto de prueba. Puntos: semillas; línea: media. La ventaja de LoRA crece en regiones no vistas.", COL_W),
  ...tabla("McNemar exacto base vs. LoRA en test fuera: imágenes que solo acierta cada modelo.",
    ["Semilla", "Solo base", "Solo LoRA", "p"],
    [["42", "67", "161", "4.2×10⁻¹⁰"], ["0", "95", "211", "2.9×10⁻¹¹"], ["1", "77", "153", "6.1×10⁻⁷"]],
    [1000, 1150, 1150, 1300]),
  P("**¿Dónde gana LoRA fuera de las regiones de entrenamiento?** La Fig. 3 muestra el F1 por clase. Las mayores ventajas fuera están en vegetación herbácea (+5.2 pts), cultivos permanentes (+4.4), carreteras (+4.1) y ríos (+4.0). Carreteras y ríos ya eran las clases con más ganancia en la partición interna. La caída por región se concentra en las clases agrícolas: cultivos permanentes pierde 34 pts (base) y 31 pts (LoRA), y pastizales unos 12 pts en ambos, mientras que bosque, agua e industria casi no cambian. Es decir, el cambio de región afecta sobre todo a clases cuyo aspecto depende de prácticas agrícolas locales, y LoRA no lo elimina, pero lo reduce en vegetación herbácea (caída de 11.9 a 6.8 pts) y en cultivos anuales (8.1 a 5.8). Por región de *test fuera*, LoRA mejora en las tres (+1.1 a +1.8 pts de macro-F1); la región mediterránea es la más difícil para ambos (47.5 vs. 49.3), aunque solo contiene 5 clases y 424 imágenes."),
];

const anchoFig2 = [...figura("C5_f1_por_clase_geo.png", "F1 por clase (media de 3 semillas) en test dentro y test fuera. Las clases agrícolas concentran la caída; LoRA gana sobre todo en vegetación herbácea, cultivos permanentes, carreteras y ríos.", 8640)];

const cuerpo2 = [
  H2("Ablaciones"),
  P("La Tabla IV compara cada variante con LoRA r = 8, Q+V, bloques 11–12 (semilla 42). En la partición interna, adaptar los 4 últimos bloques es la única variante con una diferencia mayor que el ruido (+1.19 pts); la predicción (+0.1 a +0.5) se rechaza por subestimar la mejora. Solo Q rindió +0.45 pts sobre Q+V, contra la predicción de una caída de 0.2–0.6 pts (se rechaza, aunque la diferencia está dentro del ruido de una semilla). El rango casi no importa (r = 16: +0.08; r = 4: +0.49), coherente con [3]. En la partición geográfica, 4 bloques es también la mejor variante fuera de las regiones de entrenamiento (+1.48 pts) y *reduce* la caída en 0.69 pts; habíamos predicho lo contrario (más caída, por alterar *features* más genéricas), así que esa predicción se rechaza. Rango y solo Q apenas cambian la caída (≤ 0.5 pts). La Tabla V contrasta todas las predicciones registradas antes de ejecutar con sus resultados."),
  ...tabla("Ablaciones (semilla 42), macro-F1 (%). Δ: diferencia con LoRA r = 8 Q+V bloques 11–12. Interna: test aleatorio; geográfica: test fuera y Δ de la caída (dentro − fuera).",
    ["Variante", "Interna F1 (Δ)", "Geo fuera F1 (Δ)", "Δ caída"],
    [["Referencia (r = 8, Q+V, b. 11–12)", "96.78", "90.79", "—"],
     ["r = 4", "97.27 (+0.49)", "90.54 (−0.25)", "−0.09"],
     ["r = 16", "96.86 (+0.08)", "90.97 (+0.18)", "+0.46"],
     ["Bloques 9–12", "97.97 (+1.19)", "92.27 (+1.48)", "−0.69"],
     ["Solo Q", "97.24 (+0.45)", "90.60 (−0.19)", "+0.30"]],
    [1636, 1100, 1160, 800]),
  H2("Costo"),
  P("Entrenar LoRA toma unos 500–580 s por corrida en la T4, frente a unos 5 s de la línea base una vez extraídos los *embeddings* (~70 s la primera vez). Tras fusionar B·A en W, el modelo tiene el mismo tamaño y la misma latencia que el original (4.43 vs. 4.41 ms/imagen con batch 32)."),

];

const tablaPredicciones = [
  ...tabla("Predicciones registradas antes de ejecutar (DESIGN_LOG.md) frente a los resultados. Valores en puntos de macro-F1; Δ respecto a LoRA r = 8, Q+V, bloques 11–12.",
    ["ID", "Experimento", "Predicción (criterio de rechazo)", "Resultado", "Veredicto"],
    [["A2", "Base vs. LoRA, 3 semillas, partición interna", "Δ de +1.0 a +1.6 en las 3 semillas; desv. ≤ 0.3 (rechazo: alguna Δ ≤ 0 o desv. > 0.5)", "Δ = +1.19, +1.32, +2.06; desv. 0.47", "No se rechaza; semilla 1 y desv. fuera de lo predicho"],
     ["A3", "Rango r = 4 y r = 16, interna", "Ambos a ±0.3 de r = 8 (rechazo: r16 > +0.5 o r4 < −0.5)", "r4: +0.49; r16: +0.08", "No se rechaza; r4 fuera del rango"],
     ["A4", "Fase C, 3 semillas", "Caída de 3 a 10 en ambos; ventaja de LoRA fuera ≤ +0.5 (rechazo: caída < 1 o ventaja ≥ +1.3)", "Caída 7.15 / 5.87; ventaja fuera +2.20 (3/3 positivas)", "Se rechaza (H2)"],
     ["A5", "Bloques 9–12, interna", "+0.1 a +0.5 (rechazo: < −0.3 o > +0.8)", "+1.19", "Se rechaza (mejora mayor)"],
     ["A6", "Solo Q, interna", "Cae 0.2 a 0.6 (rechazo: solo Q supera a Q+V o cae > 1.0)", "+0.45", "Se rechaza (dentro del ruido)"],
     ["A3g", "Rango, geográfica", "Dentro ±0.3; caída similar a r = 8 (rechazo: r16 reduce la caída > 1.5)", "Dentro r4 −0.35, r16 +0.65; Δ caída −0.09 / +0.46", "No se rechaza"],
     ["A5g", "Bloques 9–12, geográfica", "Dentro +0.1 a +0.4; 0.5 a 1.5 más de caída (rechazo: cae > 0.5 menos)", "Dentro +0.79; caída 0.69 menor", "Se rechaza (generaliza mejor)"],
     ["A6g", "Solo Q, geográfica", "Dentro −0.2 a −0.5; caída 0.2 a 0.5 menor (rechazo: fuera > 1.5 bajo la base)", "Dentro +0.10; caída 0.30 mayor; fuera 1.51 sobre la base", "No se rechaza; dirección opuesta"]],
    [560, 1750, 3150, 2350, 1942]),
];

const cuerpo2b = [
  H1("Discusión"),
  P("Esperábamos que la ventaja de LoRA fuera, en parte, sobreajuste a las regiones de entrenamiento, ya que su pérdida de entrenamiento llega a valores muy bajos. Los datos dicen lo contrario: la ventaja es mayor fuera que dentro y la caída de LoRA es menor que la de la base. Una lectura coherente con el análisis del modelo base es que el desajuste principal de DINOv2 con EuroSAT no es regional sino de dominio: vista cenital, baja resolución reescalada y texturas en lugar de objetos. Ese desajuste es común a todas las regiones y adaptar la atención de los bloques profundos lo corrige en todas. Que adaptar 4 bloques generalice aún mejor apunta en la misma dirección. Lo que sí es regional, la apariencia de los cultivos, sigue limitando a ambos modelos. El F1 de cultivos permanentes cae más de 30 pts sobre todo por falsos positivos: en regiones nuevas, vegetación herbácea y cultivos anuales se predicen como cultivos permanentes (89 y 68 imágenes por semilla con la base; 56 y 49 con LoRA). LoRA reduce los errores en *test fuera* de 352 a 257 por semilla (−27 %), en gran parte en esas confusiones, pero no las elimina; el bosque oscuro confundido con agua (~11 por semilla) persiste en ambos (Figs. 4 y 5)."),
];

const anchoFig3 = [...figura("C5_confusion_geo.png", "Matrices de confusión en test fuera (3 semillas sumadas). Color: porcentaje de la fila (recall); número: imágenes. Los principales errores de ambos modelos son vegetación herbácea y cultivos anuales predichos como cultivos permanentes.", 8640), ...figura("C8_ejemplos_fallo_geo.png", "Ejemplos de test fuera (semilla 42) donde solo LoRA acierta, solo la base acierta o ambos fallan.", 8640)];

const cuerpo3 = [
  P(`**Amenazas a la validez.** (i) El *test fuera* tiene solo tres regiones, dos de ellas en Europa central; otra partición podría dar otra magnitud de caída. (ii) Las ablaciones usan una semilla. (iii) No hubo búsqueda de hiperparámetros; ambos modelos usan el mismo presupuesto. (iv) Después de las corridas, una revisión de código encontró dos desviaciones del presupuesto del E2 en el entrenamiento de LoRA: se usó fp16 sin escalado de gradientes (*GradScaler*) y la inicialización de A no dependía de la semilla de la corrida. Las 6 corridas de la línea base no se ven afectadas. La validación de LoRA (semilla 42, partición interna: 97.46) reproduce la del E2 (97.52), lo que indica que el entrenamiento no quedó frenado; el test interno (96.78 vs. 97.14 en el E2) difiere en 0.36 pts, del orden del ruido.${SIN_CONTROL ? "" : " Para medir su efecto repetimos una corrida (LoRA, partición geográfica, semilla 42) con el código corregido: frente a la original, el macro-F1 cambia −0.15 pts en validación, +0.46 en test dentro y −0.08 en test fuera, y la ventaja de LoRA fuera pasa de +1.70 a +1.63 pts. Las dos desviaciones no alteran los resultados ni las conclusiones."} (v) La partición geográfica se generó en la misma sesión, justo antes de entrenar, y no se commiteó antes; es determinista y se regeneró idéntica, imagen por imagen.`),

  H1("Conclusiones"),
  P("En EuroSAT, adaptar DINOv2 ViT-S/14 con LoRA en Q y V de los dos últimos bloques (0.11 % de parámetros adicionales y sin costo de inferencia tras fusionar) mejora el macro-F1 en la partición interna (+1.5 pts) y aún más en regiones geográficas no vistas (+2.2 pts, consistente en 3 semillas). Ambos modelos pierden 6–7 pts al pasar a regiones nuevas, sobre todo en clases agrícolas, y la partición aleatoria sobreestima en unos 6 pts el rendimiento real en regiones nuevas. La hipótesis de que LoRA perdería su ventaja fuera de las regiones de entrenamiento se rechaza. Como trabajo futuro: más regiones de prueba (validación cruzada espacial con varios pliegues), más semillas en las ablaciones, adaptar más bloques y comparar con la variante con registros."),

  H1("Reproducibilidad y uso de IA"),
  P("Las particiones (splits/), las 20 corridas con sus predicciones (results/E3_corridas/), el código (src/eurosat_lora/) y el cuaderno de decisiones con las predicciones previas están en el repositorio. Se usaron asistentes de IA (Claude, Antigravity) para el diseño del protocolo, el código, la revisión de código y la redacción; el detalle y lo que revisó el equipo está en DESIGN_LOG.md, sección 4."),

  H1("Referencias"),
  ...[
    "M. Oquab, T. Darcet, T. Moutakanni *et al.*, “DINOv2: Learning robust visual features without supervision,” *Trans. Mach. Learn. Res.*, 2024.",
    "A. Dosovitskiy *et al.*, “An image is worth 16x16 words: Transformers for image recognition at scale,” in *Proc. ICLR*, 2021.",
    "E. J. Hu *et al.*, “LoRA: Low-rank adaptation of large language models,” in *Proc. ICLR*, 2022.",
    "P. Helber, B. Bischke, A. Dengel, and D. Borth, “EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification,” *IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens.*, vol. 12, no. 7, pp. 2217–2226, 2019.",
    "D. R. Roberts *et al.*, “Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure,” *Ecography*, vol. 40, no. 8, pp. 913–929, 2017.",
    "T. Darcet, M. Oquab, J. Mairal, and P. Bojanowski, “Vision transformers need registers,” in *Proc. ICLR*, 2024.",
    "J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, “How transferable are features in deep neural networks?” in *Proc. NeurIPS*, 2014.",
    "W. Luo, Y. Li, R. Urtasun, and R. Zemel, “Understanding the effective receptive field in deep convolutional neural networks,” in *Proc. NeurIPS*, 2016.",
    "I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in *Proc. ICLR*, 2019.",
    "Q. McNemar, “Note on the sampling error of the difference between correlated proportions or percentages,” *Psychometrika*, vol. 12, no. 2, pp. 153–157, 1947.",
  ].map((r, i) => new Paragraph({
    children: [new TextRun({ text: `[${i + 1}]\t`, font: FONT, size: 16 }), ...runs(r, { size: 16 })],
    indent: { left: 360, hanging: 360 }, spacing: { after: 30 }, alignment: AlignmentType.JUSTIFIED,
  })),
];

// ================== DOCUMENTO ==================
const pagina = { size: { width: 11906, height: 16838 }, margin: { top: 1080, bottom: 1080, left: 1077, right: 1077 } };
const pie = { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
  children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16 })] })] }) };
const dosCol = { count: 2, space: 360, equalWidth: true };
const unaCol = { count: 1 };

const doc = new Document({
  creator: "Equipo 1INF52",
  title: "¿Adaptar o congelar? LoRA sobre DINOv2 en regiones geográficas no vistas",
  styles: { default: { document: { run: { font: FONT, size: BODY } } } },
  sections: [
    { properties: { page: pagina, column: unaCol }, footers: pie, children: titulo },
    { properties: { type: SectionType.CONTINUOUS, page: pagina, column: dosCol }, footers: pie, children: [...resumen, ...cuerpo1] },
    { properties: { type: SectionType.CONTINUOUS, page: pagina, column: unaCol }, footers: pie, children: anchoFig2 },
    { properties: { type: SectionType.CONTINUOUS, page: pagina, column: dosCol }, footers: pie, children: cuerpo2 },
    { properties: { type: SectionType.CONTINUOUS, page: pagina, column: unaCol }, footers: pie, children: tablaPredicciones },
    { properties: { type: SectionType.CONTINUOUS, page: pagina, column: dosCol }, footers: pie, children: cuerpo2b },
    { properties: { type: SectionType.CONTINUOUS, page: pagina, column: unaCol }, footers: pie, children: anchoFig3 },
    { properties: { type: SectionType.CONTINUOUS, page: pagina, column: dosCol }, footers: pie, children: cuerpo3 },
  ],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(SALIDA, buf); console.log("Escrito", SALIDA); });
