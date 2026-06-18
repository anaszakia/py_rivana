"""
das_selector.py — Modul Pemilihan DAS untuk RIVANA
====================================================
Tiga mode pemilihan:
  A. Otomatis via HydroSHEDS (pilih level 3–8)
  B. Pilih dari daftar DAS yang mengandung titik
  C. Upload shapefile / GeoJSON sendiri

Semua mode mengembalikan objek DASResult yang seragam,
sehingga fungsi fetch_gee_data(), fetch_morphology_data(),
create_river_network_map() cukup menerima das.geometry
tanpa perubahan logika internal.
"""

import os
import json
import ee
import numpy as np


# ──────────────────────────────────────────────────────────────
# Struktur data hasil pemilihan DAS
# ──────────────────────────────────────────────────────────────
class DASResult:
    """
    Objek tunggal yang diteruskan ke seluruh pipeline RIVANA.

    Atribut
    -------
    geometry    : ee.Geometry  — batas DAS di GEE
    area_km2    : float        — luas DAS (km²)
    hybas_id    : int | None   — ID HydroSHEDS (None jika custom)
    level       : int | None   — level HydroSHEDS (None jika custom)
    name        : str          — label ramah-manusia
    source      : str          — 'hydrosheds' | 'shapefile' | 'geojson'
    properties  : dict         — semua properti asli dari GEE / file
    """

    def __init__(self, geometry, area_km2, name, source,
                 hybas_id=None, level=None, properties=None):
        self.geometry   = geometry
        self.area_km2   = float(area_km2)
        self.hybas_id   = hybas_id
        self.level      = level
        self.name       = name
        self.source     = source
        self.properties = properties or {}

    def summary(self):
        lines = [
            "╔══════════════════════════════════════════════════════╗",
            "║            DAS YANG AKAN DIANALISIS                  ║",
            "╠══════════════════════════════════════════════════════╣",
            f"║  Nama    : {self.name:<41}║",
            f"║  Sumber  : {self.source:<41}║",
            f"║  Luas    : {self.area_km2:>10.1f} km²{' '*27}║",
        ]
        if self.level:
            lines.append(f"║  Level   : HydroSHEDS {self.level:<33}║")
        if self.hybas_id:
            lines.append(f"║  HYBAS ID: {str(self.hybas_id):<41}║")
        lines.append("╚══════════════════════════════════════════════════════╝")
        return "\n".join(lines)

    def to_dict(self):
        return {
            "name":      self.name,
            "source":    self.source,
            "area_km2":  self.area_km2,
            "hybas_id":  self.hybas_id,
            "level":     self.level,
            "properties": self.properties,
        }


# ──────────────────────────────────────────────────────────────
# Label deskriptif per level HydroSHEDS
# ──────────────────────────────────────────────────────────────
LEVEL_INFO = {
    3: ("DAS sangat besar",  "> 10.000 km²"),
    4: ("DAS besar",         "1.000 – 10.000 km²"),
    5: ("DAS menengah",      "100 – 1.000 km²"),
    6: ("Sub-DAS",           "10 – 100 km²"),
    7: ("Sub-DAS kecil",     "1 – 10 km²"),
    8: ("Mikro-DAS",         "< 1 km²"),
}


# ──────────────────────────────────────────────────────────────
# Mode A — Pilih level HydroSHEDS
# ──────────────────────────────────────────────────────────────
def _fetch_hydrosheds(lon, lat, level):
    """
    Ambil DAS dari HydroSHEDS pada level tertentu.
    Kembalikan DASResult atau None jika tidak ada data.
    """
    lokasi = ee.Geometry.Point([lon, lat])
    try:
        fc = ee.FeatureCollection(
            f'WWF/HydroSHEDS/v1/Basins/hybas_{level}'
        ).filterBounds(lokasi)

        count = fc.size().getInfo()
        if count == 0:
            return None

        feat  = fc.first()
        props = feat.getInfo()['properties']
        geom  = feat.geometry()
        area  = props.get('SUB_AREA', 0)

        label, _ = LEVEL_INFO.get(level, ("DAS", ""))
        name = f"{label} (Level {level}, ID {props.get('HYBAS_ID','-')})"

        return DASResult(
            geometry   = geom,
            area_km2   = area,
            name       = name,
            source     = 'hydrosheds',
            hybas_id   = props.get('HYBAS_ID'),
            level      = level,
            properties = props,
        )
    except Exception as e:
        print(f"   ⚠️  Level {level} gagal: {e}")
        return None


def select_by_level(lon, lat):
    """
    Tampilkan semua level HydroSHEDS yang tersedia untuk titik ini,
    biarkan pengguna memilih satu.
    """
    print("\n🔍 Mencari DAS di semua level HydroSHEDS...")

    candidates = {}
    for lvl in range(3, 9):
        result = _fetch_hydrosheds(lon, lat, lvl)
        if result:
            candidates[lvl] = result
            label, size_range = LEVEL_INFO[lvl]
            print(f"   Level {lvl} — {label:<22} {result.area_km2:>10.1f} km²  ({size_range})")

    if not candidates:
        raise ValueError("❌ Tidak ada data HydroSHEDS untuk lokasi ini.")

    print()
    valid_levels = list(candidates.keys())
    default_level = 5 if 5 in candidates else valid_levels[len(valid_levels)//2]

    while True:
        raw = input(
            f"Pilih level ({min(valid_levels)}-{max(valid_levels)}, "
            f"default={default_level}): "
        ).strip()

        if raw == "":
            chosen = default_level
            break
        if raw.isdigit() and int(raw) in candidates:
            chosen = int(raw)
            break
        print(f"   ⚠️  Pilihan tidak valid. Masukkan angka dari {valid_levels}.")

    return candidates[chosen]


# ──────────────────────────────────────────────────────────────
# Mode B — Pilih dari daftar DAS yang mengandung titik
#           (semua level sekaligus, diurutkan dari kecil ke besar)
# ──────────────────────────────────────────────────────────────
def select_from_list(lon, lat):
    """
    Kumpulkan DAS dari level 3–8, tampilkan tabel,
    biarkan pengguna memilih nomor urut.
    """
    print("\n🔍 Mengumpulkan semua DAS yang mengandung titik ini...")

    rows = []
    for lvl in range(3, 9):
        result = _fetch_hydrosheds(lon, lat, lvl)
        if result:
            rows.append(result)

    if not rows:
        raise ValueError("❌ Tidak ada data HydroSHEDS untuk lokasi ini.")

    # Urutkan dari terkecil ke terbesar
    rows.sort(key=lambda r: r.area_km2)

    print()
    print("╔═══╦══════════╦════════════════════════╦══════════════════╗")
    print("║ # ║  Level   ║  Jenis DAS             ║  Luas (km²)     ║")
    print("╠═══╬══════════╬════════════════════════╬══════════════════╣")

    for i, r in enumerate(rows, 1):
        label, _ = LEVEL_INFO.get(r.level, ("DAS", ""))
        print(f"║ {i} ║  Level {r.level}  ║  {label:<22}║  {r.area_km2:>12.1f}   ║")

    print("╚═══╩══════════╩════════════════════════╩══════════════════╝")

    default_idx = 3  # Level 5 biasanya ada di posisi tengah
    default_idx = min(default_idx, len(rows))

    while True:
        raw = input(
            f"\nPilih nomor DAS (1-{len(rows)}, default={default_idx}): "
        ).strip()

        if raw == "":
            chosen = rows[default_idx - 1]
            break
        if raw.isdigit() and 1 <= int(raw) <= len(rows):
            chosen = rows[int(raw) - 1]
            break
        print(f"   ⚠️  Masukkan angka antara 1 dan {len(rows)}.")

    return chosen


# ──────────────────────────────────────────────────────────────
# Mode C — Upload shapefile atau GeoJSON
# ──────────────────────────────────────────────────────────────
def _load_shapefile(path):
    """
    Baca shapefile (.shp) atau GeoJSON (.json/.geojson)
    menggunakan geopandas, kembalikan ee.Geometry.
    """
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError(
            "❌ geopandas tidak terinstall.\n"
            "   Jalankan: pip install geopandas --break-system-packages"
        )

    ext = os.path.splitext(path)[1].lower()
    if ext not in ('.shp', '.json', '.geojson'):
        raise ValueError(f"Format file tidak didukung: {ext}. Gunakan .shp / .json / .geojson")

    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError("File tidak berisi fitur geometri.")

    # Pastikan CRS WGS84
    if gdf.crs is None:
        print("   ⚠️  CRS tidak ditemukan, diasumsikan EPSG:4326 (WGS84).")
    else:
        gdf = gdf.to_crs('EPSG:4326')

    # Jika lebih dari satu fitur, tanyakan mana yang dipilih
    if len(gdf) > 1:
        print(f"\n   File mengandung {len(gdf)} fitur:")
        for i, row in gdf.iterrows():
            area = row.geometry.area * (111_000 ** 2) / 1_000_000  # kasar, km²
            name_col = next(
                (c for c in gdf.columns if 'name' in c.lower() or 'nama' in c.lower()),
                None
            )
            label = row[name_col] if name_col else f"Fitur {i+1}"
            print(f"   {i+1}. {label}  (~{area:.1f} km²)")

        while True:
            raw = input(f"   Pilih fitur (1-{len(gdf)}): ").strip()
            if raw.isdigit() and 1 <= int(raw) <= len(gdf):
                geom_row = gdf.iloc[int(raw) - 1]
                break
            print("   ⚠️  Pilihan tidak valid.")
    else:
        geom_row = gdf.iloc[0]

    geojson_dict = geom_row.geometry.__geo_interface__

    # Sederhanakan geometri jika terlalu kompleks (> 1000 simpul)
    n_coords = sum(len(r) for r in _count_coords(geojson_dict))
    if n_coords > 1000:
        print(f"   ⚠️  Geometri kompleks ({n_coords} simpul), menyederhanakan...")
        simplified = geom_row.geometry.simplify(0.001, preserve_topology=True)
        geojson_dict = simplified.__geo_interface__
        print("   ✅ Geometri disederhanakan.")

    ee_geom = ee.Geometry(geojson_dict)

    # Hitung luas dari GEE
    try:
        area_m2 = ee_geom.area(maxError=100).getInfo()
        area_km2 = area_m2 / 1_000_000
    except Exception:
        area_km2 = geom_row.geometry.area * (111_000 ** 2) / 1_000_000

    name_col = next(
        (c for c in gdf.columns if 'name' in c.lower() or 'nama' in c.lower()),
        None
    )
    das_name = str(geom_row[name_col]) if name_col else os.path.basename(path)

    return DASResult(
        geometry   = ee_geom,
        area_km2   = area_km2,
        name       = das_name,
        source     = 'shapefile' if path.endswith('.shp') else 'geojson',
        properties = geom_row.drop('geometry').to_dict() if hasattr(geom_row, 'drop') else {},
    )


def _count_coords(geojson):
    """Hitung jumlah simpul dalam GeoJSON (untuk cek kompleksitas)."""
    t = geojson.get('type', '')
    c = geojson.get('coordinates', [])
    if t == 'Point':
        return [[c]]
    if t in ('LineString', 'MultiPoint'):
        return [c]
    if t in ('Polygon', 'MultiLineString'):
        return c
    if t == 'MultiPolygon':
        return [ring for poly in c for ring in poly]
    return [[]]


def select_from_file(path=None):
    """
    Muat DAS dari file lokal. Jika path tidak diberikan,
    minta pengguna mengetikkan path-nya.
    """
    if not path:
        print("\n📁 Masukkan path file DAS (.shp / .json / .geojson):")
        path = input("   Path: ").strip()

    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"❌ File tidak ditemukan: {path}")

    print(f"\n📂 Memuat file: {path}")
    result = _load_shapefile(path)
    print(f"   ✅ DAS dimuat: {result.name}  ({result.area_km2:.1f} km²)")
    return result


# ──────────────────────────────────────────────────────────────
# Fungsi utama — tampilkan menu dan kembalikan DASResult
# ──────────────────────────────────────────────────────────────
def select_das(lon, lat, shapefile_path=None, auto_level=None):
    """
    Entry point utama.

    Parameter
    ---------
    lon, lat        : koordinat titik analisis
    shapefile_path  : jika diisi, langsung pakai file ini (lewati menu)
    auto_level      : jika diisi (3-8), langsung pakai level ini (lewati menu)

    Kembalikan
    ----------
    DASResult
    """
    # ── Shortcut: sudah ada file atau level dari argumen ──
    if shapefile_path:
        result = select_from_file(shapefile_path)
        print("\n" + result.summary())
        return result

    if auto_level is not None:
        result = _fetch_hydrosheds(lon, lat, auto_level)
        if result is None:
            raise ValueError(f"❌ Tidak ada DAS HydroSHEDS level {auto_level} untuk lokasi ini.")
        print("\n" + result.summary())
        return result

    # ── Menu interaktif ──
    print("\n" + "=" * 60)
    print("  PILIH METODE PENENTUAN DAS".center(60))
    print("=" * 60)
    print("  1. Pilih level HydroSHEDS (otomatis dari koordinat)")
    print("  2. Tampilkan daftar DAS & pilih dari tabel")
    print("  3. Upload shapefile / GeoJSON DAS sendiri")
    print("=" * 60)

    while True:
        mode = input("\nPilih mode (1/2/3, default=2): ").strip() or "2"
        if mode in ("1", "2", "3"):
            break
        print("   ⚠️  Masukkan 1, 2, atau 3.")

    if mode == "1":
        result = select_by_level(lon, lat)
    elif mode == "2":
        result = select_from_list(lon, lat)
    else:
        result = select_from_file()

    print("\n" + result.summary())

    # Konfirmasi
    ok = input("\nLanjutkan dengan DAS ini? (y/n, default=y): ").strip().lower()
    if ok == "n":
        print("\n🔄 Ulangi pemilihan DAS...")
        return select_das(lon, lat)  # rekursi sekali

    return result


# ──────────────────────────────────────────────────────────────
# Utilitas: simpan info DAS ke JSON
# ──────────────────────────────────────────────────────────────
def save_das_info(das_result, output_dir='.'):
    """Simpan metadata DAS ke RIVANA_DAS_Info.json"""
    from das_selector import DASResult  # hindari circular jika diimport ulang

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'RIVANA_DAS_Info.json')

    data = das_result.to_dict()

    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=4, default=str)
        print(f"   ✅ RIVANA_DAS_Info.json saved")
    except Exception as e:
        print(f"   ⚠️  Gagal menyimpan DAS info: {e}")

    return path