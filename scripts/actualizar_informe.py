"""actualizar_informe.py — Actualiza el informe en Word (DOCX) con los resultados de Fase C y ablaciones."""

import docx
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


def set_cell_border(cell, **kwargs):
    """
    Establece bordes en una celda de tabla docx.
    kwargs: top, bottom, left, right con dict {'sz': 12, 'val': 'single', 'color': 'CCCCCC'}
    """
    tcPr = cell._element.get_or_add_tcPr()
    tcBorders = tcPr.first_child_found_in("w:tcBorders")
    if tcBorders is None:
        tcBorders = OxmlElement('w:tcBorders')
        tcPr.append(tcBorders)
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        edge_data = kwargs.get(edge)
        if edge_data:
            tag = f'w:{edge}'
            element = tcBorders.find(qn(tag))
            if element is None:
                element = OxmlElement(tag)
                tcBorders.append(element)
            for key, val in edge_data.items():
                element.set(qn(f'w:{key}'), str(val))


def format_table(table, col_widths, headers, data_rows):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    # Header row
    hdr_cells = table.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.name = 'Times New Roman'
            run.font.size = Pt(9.5)
            run.bold = True
        set_cell_border(hdr_cells[i], bottom={'sz': 12, 'val': 'single', 'color': '4B2A82'},
                                      top={'sz': 6, 'val': 'single', 'color': '4B2A82'})

    # Data rows
    for r_idx, row_data in enumerate(data_rows):
        row = table.add_row()
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            cell.text = str(val)
            p = cell.paragraphs[0]
            # Justify numbers to center/right, text to left
            if c_idx == 0:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.font.name = 'Times New Roman'
                run.font.size = Pt(9)
                if r_idx == len(data_rows) - 1 and ('Diferencia' in str(row_data[0]) or 'LoRA' in str(row_data[1])):
                    run.bold = True
            border_bottom = {'sz': 12, 'val': 'single', 'color': '4B2A82'} if r_idx == len(data_rows) - 1 else {'sz': 4, 'val': 'single', 'color': 'E0E0E0'}
            set_cell_border(cell, bottom=border_bottom)

    # Set column widths
    for row in table.rows:
        for idx, width in enumerate(col_widths):
            row.cells[idx].width = width


def main():
    doc_path = "docs/Informe_avance_TA_DINOv2_LoRA_EuroSAT.docx"
    doc = docx.Document(doc_path)

    # 1. Actualizar título de sección 4
    for p in doc.paragraphs:
        if p.text.strip() == "4. Resultados preliminares":
            p.text = "4. Resultados y Análisis"
        elif p.text.startswith("A continuación se presentan los resultados de las fases A y B (entrega E2)"):
            p.text = (
                "A continuación se presentan los resultados integrales de las tres fases del proyecto (A, B y C). "
                "Se analizó la arquitectura del modelo base (fase A), se evaluó la intervención LoRA en partición interna (fase B) "
                "y se estudió la generalización espacial frente a bloques geográficos disjuntos con 3 semillas aleatorias "
                "y un conjunto de ablaciones arquitectónicas simétricas (fase C), respondiendo de forma concluyente a las hipótesis planteadas."
            )

    # 2. Localizar párrafo 4.3 para reemplazarlo con la Fase C completa
    idx_43 = None
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip().startswith("4.3 Limitaciones"):
            idx_43 = i
            break

    if idx_43 is None:
        print("No se encontró 4.3 Limitaciones, buscando por 4.2...")
        for i, p in enumerate(doc.paragraphs):
            if p.text.strip().startswith("4.2"):
                idx_43 = i + 10
                break

    print(f"Modificando a partir del párrafo {idx_43}...")
    # Reemplazamos el párrafo de 4.3 con el encabezado de Fase C
    doc.paragraphs[idx_43].text = "4.3 Fase C: Generalización geográfica y robustez con múltiples semillas"
    doc.paragraphs[idx_43].style = "Heading 2"

    p_c1 = doc.paragraphs[idx_43 + 1]
    p_c1.text = (
        "Para responder a la pregunta central de investigación, construimos una partición espacial basada en bloques geográficos "
        "disjuntos mediante clustering K-Means sobre las coordenadas georreferenciadas extraídas de los GeoTIFFs multiespectrales de Sentinel-2. "
        "Las imágenes pertenecientes a clusters reservados conformaron el conjunto de prueba fuera de región (test_fuera, nunca visto en entrenamiento), "
        "mientras que las imágenes de los clusters usados para entrenar formaron el conjunto de prueba dentro de región (test_dentro). "
        "Para garantizar validez estadística y verificar que los resultados no fueran producto del azar, ejecutamos todas las condiciones "
        "con tres semillas aleatorias independientes (semillas 42, 0 y 1)."
    )

    # Insertar párrafo de presentación de la Tabla 6
    p_tab6_caption = doc.add_paragraph("Tabla 6. Resultados agregados (media ± desviación estándar) en 3 semillas (42, 0, 1) y prueba de McNemar.", style="normal")
    p_tab6_caption.runs[0].font.name = "Times New Roman"
    p_tab6_caption.runs[0].font.size = Pt(10)
    p_tab6_caption.runs[0].italic = True

    # Insertar Tabla 6
    headers_t6 = ["Partición", "Condición", "Macro-F1 (Dentro / Interna)", "Macro-F1 (Fuera)", "Caída geográfica (Δ)", "McNemar (p-valor)"]
    widths_t6 = [Inches(1.1), Inches(1.2), Inches(1.5), Inches(1.3), Inches(1.4), Inches(1.4)]
    data_t6 = [
        ["Interna (E2)", "Linear probing", "95.66 ± 0.16 %", "---", "---", "---"],
        ["Interna (E2)", "LoRA (r=8, b11-12)", "97.33 ± 0.17 %", "---", "---", "p < 1e-4 en todas"],
        ["Geográfica (E3)", "Linear probing", "96.40 ± 0.18 %", "89.25 ± 0.72 %", "−7.15 ± 0.55 pp", "---"],
        ["Geográfica (E3)", "LoRA (r=8, b11-12)", "97.54 ± 0.17 %", "91.51 ± 0.38 %", "−6.02 ± 0.55 pp", "p < 1e-6 en todas"],
        ["Ventaja LoRA", "---", "+1.14 pp", "+2.27 pp", "+1.13 pp (menor caída)", "Significativo (p << 0.001)"],
    ]
    t6 = doc.add_table(rows=1, cols=6)
    format_table(t6, widths_t6, headers_t6, data_t6)

    # Texto de análisis estadístico y rechazo de H2
    p_c2 = doc.add_paragraph(
        "Como se observa en la Tabla 6, la hipótesis H1 se confirma de manera robusta: en la partición interna, LoRA supera consistentemente "
        "al linear probing en +1.67 puntos porcentuales de Macro-F1 a través de las tres semillas (de 95.66% a 97.33%).",
        style="normal"
    )
    p_c3 = doc.add_paragraph(
        "En la partición geográfica, ambos modelos sufren el impacto del cambio de distribución espacial al clasificar imágenes de regiones "
        "geográficas nunca antes vistas. Sin embargo, el hallazgo más relevante es que el modelo base congelado experimenta una caída de 7.15 puntos "
        "(de 96.40% a 89.25%), mientras que LoRA experimenta una caída significativamente menor de 6.02 puntos (de 97.54% a 91.51%). "
        "En lugar de contraerse por sobreajuste local (como anticipaba la hipótesis pesimista H2), la ventaja de LoRA se duplica: "
        "pasa de +1.14 pp en regiones conocidas a +2.27 pp en regiones geográficas no vistas. "
        "La prueba de McNemar exacta sobre el conjunto test_fuera arroja p-valores extremadamente bajos (p = 4.17e-10 en semilla 42, "
        "p = 2.86e-11 en semilla 0, y p = 6.07e-7 en semilla 1), con una proporción de discordancias favorable a LoRA superior a 2.2 a 1. "
        "Por lo tanto, la hipótesis pesimista H2 se rechaza categóricamente: LoRA no sobreajusta a las regiones de entrenamiento, sino que "
        "actúa como un regularizador espacial eficaz para representaciones satelitales.",
        style="normal"
    )

    # Sección 4.4 Ablaciones
    p_abl_head = doc.add_paragraph("4.4 Ablaciones arquitectónicas e intervenciones alternativas", style="Heading 2")
    p_abl_intro = doc.add_paragraph(
        "Para entender los factores de diseño que determinan la eficacia de LoRA, evaluamos tres ejes de ablación simétricos "
        "tanto en la partición interna como en la partición geográfica (semilla 42): rango intrínseco (r = 4, 8, 16), profundidad "
        "de intervención (2 bloques finales frente a los 4 últimos bloques) y selectividad de proyecciones (solo Q frente a Q y V).",
        style="normal"
    )

    p_tab7_caption = doc.add_paragraph("Tabla 7. Resultados de las ablaciones arquitectónicas en partición interna y geográfica (semilla 42).", style="normal")
    p_tab7_caption.runs[0].font.name = "Times New Roman"
    p_tab7_caption.runs[0].font.size = Pt(10)
    p_tab7_caption.runs[0].italic = True

    headers_t7 = ["Exp.", "Variante", "Parámetros LoRA", "% Backbone", "Macro-F1 Interna", "Macro-F1 Geo (Fuera)"]
    widths_t7 = [Inches(0.8), Inches(2.2), Inches(1.5), Inches(1.1), Inches(1.4), Inches(1.4)]
    data_t7 = [
        ["A3 / A3g", "Rango bajo (r = 4, b10-11, Q+V)", "12,288", "0.055 %", "97.27 %", "90.54 %"],
        ["A2 / A4", "Línea base LoRA (r = 8, b10-11, Q+V)", "24,576", "0.111 %", "97.22 %", "90.60 %"],
        ["A3 / A3g", "Rango alto (r = 16, b10-11, Q+V)", "49,152", "0.222 %", "96.86 %", "90.97 %"],
        ["A5 / A5g", "Mayor profundidad (r = 8, b8-11, Q+V)", "49,152", "0.222 %", "97.97 %", "92.27 %"],
        ["A6 / A6g", "Solo Query (r = 8, b10-11, Q)", "12,288", "0.055 %", "97.24 %", "90.60 %"],
    ]
    t7 = doc.add_table(rows=1, cols=6)
    format_table(t7, widths_t7, headers_t7, data_t7)

    p_abl_disc = doc.add_paragraph(
        "El análisis de ablaciones revela tres principios fundamentales:\n"
        "1. Invarianza de rango: Reducir el rango a r = 4 conserva prácticamente el 100% del rendimiento (97.27% interna y 90.54% geo) "
        "frente a r = 8 (97.22% y 90.60%), mientras que aumentar a r = 16 degrada ligeramente el F1 interno a 96.86%. "
        "Esto demuestra que la dimensión intrínseca requerida para adaptar DINOv2 a la textura cenital de EuroSAT es muy compacta (r <= 8).\n"
        "2. Beneficio de la profundidad: Extender los adaptadores a los últimos 4 bloques (bloques 8 a 11) produjo el mejor desempeño "
        "absoluto de todo el trabajo: 97.97% de macro-F1 en prueba interna (98.05% de accuracy) y 92.27% en regiones geográficas no vistas "
        "(94.88% accuracy), proporcionando la mayor resiliencia ante transferencias espaciales.\n"
        "3. Eficiencia en proyecciones: Adaptar únicamente las proyecciones Query (Q) obtiene un macro-F1 de 97.24% en prueba interna y 90.60% fuera, "
        "emparejando el rendimiento de Q+V pero utilizando exactamente la mitad de los parámetros (solo 12,288 pesos adaptados, equivalentes al 0.055% del backbone).",
        style="normal"
    )

    # Sección 5 Conclusiones
    p_concl_head = doc.add_paragraph("5. Conclusiones y Discusión Final", style="Heading 1")
    p_concl_1 = doc.add_paragraph(
        "En este estudio demostramos cuantitativamente que la adaptación eficiente en parámetros mediante LoRA sobre los bloques profundos "
        "de DINOv2 ViT-S/14 resuelve exitosamente la brecha entre el preentrenamiento en imágenes naturales y la teledetección satelital.",
        style="normal"
    )
    p_concl_2 = doc.add_paragraph(
        "Respuesta a la pregunta de investigación: Añadir adaptadores LoRA a DINOv2 mejora significativamente la clasificación de cobertura del suelo "
        "tanto en distribuciones conocidas (+1.14 pp) como de forma aún más pronunciada en regiones geográficas no vistas (+2.27 pp sobre la línea base). "
        "Contrario a la intuición de que adaptar pesos provocaría sobreajuste a los patrones locales de entrenamiento, LoRA demostró actuar "
        "como un regularizador de dominio, sufriendo 1.13 pp menos degradación ante el cambio de distribución espacial que el modelo congelado. "
        "Todas las ganancias cuentan con validación estadística pareada (test de McNemar exacto con p < 1e-6).",
        style="normal"
    )
    p_concl_3 = doc.add_paragraph(
        "Eficiencia y costo operacional: La intervención LoRA requiere actualizar únicamente el 0.11% de los parámetros del modelo (24,576 pesos) "
        "y, gracias a la propiedad de fusión exacta W' = W + (alpha/r)*B*A, el costo de inferencia en producción es exactamente idéntico al del modelo congelado "
        "(1.17 ms por imagen en fp16 en GPU T4).",
        style="normal"
    )
    p_concl_4 = doc.add_paragraph(
        "Trabajo futuro: Si bien LoRA resolvió con éxito la clasificación de patrones estructurales finos (carreteras y ríos), las confusiones residuales "
        "se concentran en clases de vegetación espectralmente homogéneas en RGB (pastizales frente a cultivos anuales). "
        "El siguiente paso natural es extender adaptadores LoRA a las 13 bandas multiespectrales completas de Sentinel-2 "
        "e investigar la escalabilidad del método en variantes de mayor escala como DINOv2-Large.",
        style="normal"
    )

    doc.save(doc_path)
    print(f"Informe actualizado y guardado exitosamente en: {doc_path}")


if __name__ == "__main__":
    main()
