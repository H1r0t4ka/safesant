import json
import requests
import unicodedata
import re
from difflib import get_close_matches

GEOJSON = 'santander_municipios.geojson'
API_BASE = 'https://www.datos.gov.co/resource/'
RESOURCES = ['fpe5-yrmw', 'd4fr-sbn2', 'vuyt-mqpw']

def normalize_name(s):
    if s is None:
        return ''
    s = str(s).strip()
    # remove (CT) etc
    s = re.sub(r"\s+\(CT\)$", "", s, flags=re.IGNORECASE)
    # decompose accents
    s = ''.join(ch for ch in unicodedata.normalize('NFKD', s) if not unicodedata.combining(ch))
    # keep letters, numbers and spaces
    s = ''.join(ch for ch in s if ch.isalnum() or ch.isspace())
    s = ' '.join(s.split())
    return s.upper()

def load_geo_municipios():
    with open(GEOJSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    vals = []
    for feat in data.get('features', []):
        props = feat.get('properties', {})
        v = props.get('MPIO_CNMBR') or props.get('MPIO_CNMBR'.lower())
        if v is not None:
            vals.append(normalize_name(v))
    return set(v for v in vals if v)

def load_api_municipios(limit=50000):
    muni_set = set()
    for res in RESOURCES:
        url = f"{API_BASE}{res}.json?$limit={limit}"
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
            for row in data:
                # find municipio-like key
                key = next((k for k in row.keys() if 'municipio' in k.lower()), None)
                if key:
                    v = row.get(key)
                    if v:
                        muni_set.add(normalize_name(v))
        except Exception as e:
            print('Error fetching', res, e)
    return muni_set

def fuzzy_map(missing_api, geo_names, n=3):
    suggestions = {}
    geo_list = sorted(geo_names)
    for name in sorted(missing_api):
        matches = get_close_matches(name, geo_list, n=n, cutoff=0.6)
        suggestions[name] = matches
    return suggestions

def main():
    geo = load_geo_municipios()
    api = load_api_municipios()
    print('Geo municipal count:', len(geo))
    print('API municipal count:', len(api))

    inter = geo & api
    only_api = sorted(api - geo)
    only_geo = sorted(geo - api)

    print('Matched count:', len(inter))
    print('Only in API (examples):', only_api[:30])
    print('Only in GeoJSON (examples):', only_geo[:30])

    print('\nGenerating fuzzy suggestions for API names not in GeoJSON...')
    suggestions = fuzzy_map(only_api, geo, n=5)
    # print up to 50 suggestions
    printed = 0
    for k, v in suggestions.items():
        if v:
            print(f'{k} -> {v}')
            printed += 1
        if printed >= 50:
            break

    # Save a small report
    report = {
        'geo_count': len(geo),
        'api_count': len(api),
        'matched_count': len(inter),
        'only_api_examples': only_api[:200],
        'only_geo_examples': only_geo[:200],
        'suggestions_sample': {k: suggestions[k] for k in list(suggestions.keys())[:200]}
    }
    with open('compare_geo_api_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print('\nReport written to compare_geo_api_report.json')

if __name__ == '__main__':
    main()
