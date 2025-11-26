#!/usr/bin/env python3
"""
Extract centroids from santander_municipios.geojson and save CSV/JSON mapping.
Generates:
 - municipio_centroids.csv
 - municipio_centroids.json
 - centroids_report.json

Usage: python extract_municipio_centroids.py
"""
import json
import os
import sys
import csv
import unicodedata
import re
from collections import OrderedDict

try:
    from shapely.geometry import shape
    SHAPELY_AVAILABLE = True
except Exception:
    SHAPELY_AVAILABLE = False

GEOJSON_FILES = ["santander_municipios.geojson", "santander_municipios.geojson.bak"]

OUT_CSV = "municipio_centroids.csv"
OUT_JSON = "municipio_centroids.json"
OUT_REPORT = "centroids_report.json"

re_non_alnum = re.compile(r"[^A-Z0-9]+")

def normalize_name(s):
    if s is None:
        return ""
    s = str(s).strip()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = s.upper()
    s = re_non_alnum.sub(' ', s)
    s = ' '.join(s.split())
    return s


def bbox_centroid(coords):
    # coords can be nested; flatten to list of (lon,lat)
    pts = []
    def walk(c):
        if isinstance(c, (int, float)):
            return
        if isinstance(c[0], (int, float)) and len(c) >= 2:
            pts.append((float(c[0]), float(c[1])))
            return
        for e in c:
            walk(e)
    walk(coords)
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    # shapely and geojson use lon,lat -> return lat,lon for convenience later
    return (cy, cx)


def feature_centroid(feat):
    geom = feat.get('geometry')
    if not geom:
        return None
    if SHAPELY_AVAILABLE:
        try:
            geom_obj = shape(geom)
            pt = geom_obj.centroid
            return (pt.y, pt.x)
        except Exception:
            pass
    # fallback -> use bbox centroid
    try:
        coords = geom.get('coordinates')
        return bbox_centroid(coords)
    except Exception:
        return None


def find_geojson_file():
    for f in GEOJSON_FILES:
        if os.path.exists(f):
            return f
    return None


def main():
    gf = find_geojson_file()
    if gf is None:
        print("No se encontró 'santander_municipios.geojson' ni su backup. Coloca el archivo en el directorio.")
        sys.exit(1)
    print(f"Leyendo GeoJSON: {gf}")
    with open(gf, 'r', encoding='utf-8') as fh:
        gj = json.load(fh)
    features = gj.get('features', [])
    rows = []
    for feat in features:
        props = feat.get('properties', {})
        # posibles claves: MPIO_CNMBR, MPIO_CNMBR_ORIG, NOMBRE, name
        name = props.get('MPIO_CNMBR') or props.get('MPIO_CNMBR_ORIG') or props.get('NOMBRE') or props.get('name')
        norm = normalize_name(name)
        cent = feature_centroid(feat)
        if cent is None:
            print(f"Advertencia: no se pudo calcular centrodide para '{name}'")
            continue
        lat, lon = cent
        rows.append(OrderedDict([('MPIO_CNMBR_ORIG', name), ('MPIO_CNMBR', props.get('MPIO_CNMBR')), ('name_norm', norm), ('lat', lat), ('lon', lon)]))

    # write CSV
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ['MPIO_CNMBR_ORIG','MPIO_CNMBR','name_norm','lat','lon'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # write JSON mapping and report
    mapping = {r['name_norm']: {'orig': r['MPIO_CNMBR_ORIG'], 'mpio': r['MPIO_CNMBR'], 'lat': r['lat'], 'lon': r['lon']} for r in rows}
    with open(OUT_JSON, 'w', encoding='utf-8') as fh:
        json.dump(mapping, fh, ensure_ascii=False, indent=2)

    report = {
        'file_used': gf,
        'count': len(rows),
        'csv': OUT_CSV,
        'json': OUT_JSON
    }
    with open(OUT_REPORT, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print(f"Generados: {OUT_CSV}, {OUT_JSON}. Procesados: {len(rows)} municipios.")

if __name__ == '__main__':
    main()
