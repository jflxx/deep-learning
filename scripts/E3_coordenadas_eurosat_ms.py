"""scripts/E3_coordenadas_eurosat_ms.py

Extrae coordenadas geográficas de los 27,000 parches de EuroSAT MS (GeoTIFF)
y genera splits/eurosat_coords.csv para la partición geográfica.
"""

import argparse
import os
import urllib.request
import zipfile
from glob import glob
from pathlib import Path

import pandas as pd
import rasterio

from eurosat_lora.geo import centro_lonlat, verificar_emparejamiento

EUROSAT_MS_URL = "https://zenodo.org/records/7711810/files/EuroSAT_MS.zip"


def descargar_y_extraer_ms(destino_dir: str) -> str:
    os.makedirs(destino_dir, exist_ok=True)
    zip_path = os.path.join(destino_dir, "EuroSAT_MS.zip")
    extract_dir = os.path.join(destino_dir, "EuroSAT_MS")

    if not os.path.exists(extract_dir):
        if not os.path.exists(zip_path):
            print(f"Descargando EuroSAT MS desde {EUROSAT_MS_URL}...")
            urllib.request.urlretrieve(EUROSAT_MS_URL, zip_path)
            print("Descarga completada.")
        print("Extrayendo archivo zip...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(destino_dir)
        print("Extracción completada.")

    return extract_dir


def procesar_coordenadas(ms_dir: str, ruta_salida: str = "splits/eurosat_coords.csv", rgb_dir: str | None = None) -> pd.DataFrame:
    archivos_tif = sorted(glob(os.path.join(ms_dir, "**", "*.tif"), recursive=True))
    if not archivos_tif:
        # Intentar con .TIF
        archivos_tif = sorted(glob(os.path.join(ms_dir, "**", "*.TIF"), recursive=True))

    print(f"Encontrados {len(archivos_tif)} archivos GeoTIFF en {ms_dir}")
    if len(archivos_tif) != 27000:
        raise SystemExit(f"Se esperaban 27,000 GeoTIFF y se encontraron {len(archivos_tif)}. No se genera el CSV.")
    if not rgb_dir or not os.path.exists(rgb_dir):
        raise SystemExit("Falta --rgb-dir (EuroSAT RGB): la verificación RGB <-> MS es obligatoria.")

    filas = []
    nombres_ms = set()

    for idx, fpath in enumerate(archivos_tif):
        if (idx + 1) % 5000 == 0 or idx == len(archivos_tif) - 1:
            print(f"Procesando {idx + 1}/{len(archivos_tif)}...")

        p = Path(fpath)
        clase = p.parent.name
        stem = p.stem  # Ej: Forest_1
        nombre_sin_ext = f"{clase}/{stem}"
        nombres_ms.add(nombre_sin_ext)
        archivo_rgb = f"{clase}/{stem}.jpg"

        with rasterio.open(fpath) as ds:
            lon, lat = centro_lonlat(ds.transform, ds.crs, ds.width, ds.height)
            epsg = ds.crs.to_epsg() if hasattr(ds.crs, "to_epsg") else None

        filas.append({
            "archivo_rgb": archivo_rgb,
            "clase": clase,
            "lon": lon,
            "lat": lat,
            "crs_epsg": epsg,
        })

    df = pd.DataFrame(filas)

    archivos_rgb = glob(os.path.join(rgb_dir, "**", "*.jpg"), recursive=True)
    nombres_rgb = {f"{Path(f).parent.name}/{Path(f).stem}" for f in archivos_rgb}
    print(f"Verificando emparejamiento con {len(nombres_rgb)} archivos RGB...")
    verificar_emparejamiento(nombres_ms, nombres_rgb)
    print("Emparejamiento perfecto MS <-> RGB verificado.")
    duplicadas = df.duplicated(subset=["lon", "lat"]).sum()
    print(f"Coordenadas repetidas (mismo centro en dos archivos): {duplicadas}")

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    df.to_csv(ruta_salida, index=False)
    print(f"Coordenadas guardadas en {ruta_salida} ({len(df)} registros)")
    return df


def main():
    parser = argparse.ArgumentParser(description="Extrae coordenadas de EuroSAT MS")
    parser.add_argument("--ms-dir", type=str, default=None, help="Ruta a EuroSAT MS extraído o directorio de descarga")
    parser.add_argument("--rgb-dir", type=str, default=None, help="Ruta a EuroSAT RGB para verificar emparejamiento")
    parser.add_argument("--salida", type=str, default="splits/eurosat_coords.csv", help="Ruta del CSV de salida")
    args = parser.parse_args()

    ms_dir = args.ms_dir
    if ms_dir is None or not os.path.exists(ms_dir):
        ms_dir = descargar_y_extraer_ms("data")

    procesar_coordenadas(ms_dir, ruta_salida=args.salida, rgb_dir=args.rgb_dir)


if __name__ == "__main__":
    main()
