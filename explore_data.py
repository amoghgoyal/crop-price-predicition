import csv
import os

base = '/Users/amoghgoyal/Desktop/AI 2'

files_to_check = [
    'crop_yield.csv',
    'commodity-wise-report-for-last-5-years.csv',
    'daily-rainfall-data-district-level.csv',
    'export/export product wise/2017-18.csv',
    'export/export state wise/2017-18.csv',
    'import/import country wise/2022-23.csv',
    'import/import product wise/2022-23.csv',
]

for fname in files_to_check:
    fpath = os.path.join(base, fname)
    print(f"\n{'='*60}")
    print(f"FILE: {fname}")
    print(f"{'='*60}")
    try:
        with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                print(row)
                if i >= 3:
                    break
    except Exception as e:
        print(f"Error: {e}")
