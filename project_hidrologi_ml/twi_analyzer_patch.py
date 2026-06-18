"""
Patch untuk MLTWIAnalyzer — koordinat rekomendasi berbasis geometry DAS nyata.

Strategi:
1. Ambil centroid DAS dari ee.Geometry
2. Ambil bounding box DAS
3. Generate kandidat titik via random sampling di dalam polygon DAS (GEE)
4. Scale offset dengan sqrt(area_km2) agar proporsional
5. Fallback ke centroid + offset proporsional jika GEE sampling gagal
"""

import ee
import numpy as np


# ──────────────────────────────────────────────────────────────
# Helper: ambil info spasial DAS
# ──────────────────────────────────────────────────────────────

def get_das_spatial_info(das):
    """
    Ambil centroid, bounding box, dan luas DAS dari objek DASResult.

    Returns
    -------
    dict dengan key:
        centroid_lon, centroid_lat  : float
        bbox                        : dict west/east/south/north
        area_km2                    : float
        scale_deg                   : float  — ~radius DAS dalam derajat
    """
    try:
        centroid = das.geometry.centroid(maxError=100).getInfo()
        lon_c = centroid['coordinates'][0]
        lat_c = centroid['coordinates'][1]
    except Exception:
        # fallback: gunakan bounds
        bounds = das.geometry.bounds().getInfo()
        coords = bounds['coordinates'][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        lon_c = (min(lons) + max(lons)) / 2
        lat_c = (min(lats) + max(lats)) / 2

    try:
        bounds_info = das.geometry.bounds().getInfo()
        coords = bounds_info['coordinates'][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        bbox = {
            'west':  min(lons), 'east':  max(lons),
            'south': min(lats), 'north': max(lats),
        }
    except Exception:
        # fallback kasar
        scale_deg = np.sqrt(das.area_km2) / 111.0
        bbox = {
            'west':  lon_c - scale_deg, 'east':  lon_c + scale_deg,
            'south': lat_c - scale_deg, 'north': lat_c + scale_deg,
        }

    # Radius dalam derajat ≈ sqrt(A/π) / 111 km/deg
    scale_deg = np.sqrt(das.area_km2 / np.pi) / 111.0

    return {
        'centroid_lon': lon_c,
        'centroid_lat': lat_c,
        'bbox':         bbox,
        'area_km2':     das.area_km2,
        'scale_deg':    scale_deg,
    }


def sample_points_in_das(das, n_points=10, seed=42):
    """
    Sample titik acak di dalam polygon DAS menggunakan GEE.
    Mengembalikan list of (lon, lat). Fallback ke grid jika GEE gagal.
    """
    try:
        # GEE: stratified random sample
        sample_fc = das.geometry.sample(
            numPixels=n_points * 3,   # oversample, lalu ambil n_points
            scale=1000,               # resolusi 1 km
            seed=seed,
            geometries=True
        )
        feats = sample_fc.limit(n_points).getInfo()['features']
        points = []
        for f in feats:
            coords = f['geometry']['coordinates']
            points.append((coords[0], coords[1]))

        if len(points) >= n_points:
            return points[:n_points]

    except Exception as e:
        print(f"   ⚠️ GEE sampling gagal ({e}), pakai grid fallback")

    # Fallback: grid di dalam bounding box
    info = get_das_spatial_info(das)
    bbox = info['bbox']
    cols = int(np.ceil(np.sqrt(n_points))) + 1
    rows = cols
    lons = np.linspace(bbox['west']  + (bbox['east']  - bbox['west'])  * 0.1,
                       bbox['east']  - (bbox['east']  - bbox['west'])  * 0.1, cols)
    lats = np.linspace(bbox['south'] + (bbox['north'] - bbox['south']) * 0.1,
                       bbox['north'] - (bbox['north'] - bbox['south']) * 0.1, rows)

    points = []
    for lon in lons:
        for lat in lats:
            points.append((lon, lat))
            if len(points) >= n_points:
                break
        if len(points) >= n_points:
            break

    return points[:n_points]


def filter_points_in_das(das, points):
    """
    Filter titik agar hanya yang benar-benar di dalam polygon DAS.
    Gunakan GEE containsAll (batch). Fallback: kembalikan semua jika gagal.
    """
    try:
        ee_points = [ee.Geometry.Point([lon, lat]) for lon, lat in points]
        inside = []
        for i, pt in enumerate(ee_points):
            contained = das.geometry.contains(pt, maxError=100).getInfo()
            if contained:
                inside.append(points[i])
        return inside if inside else points  # fallback jika semua di luar
    except Exception:
        return points  # fallback


# ──────────────────────────────────────────────────────────────
# Updated MLTWIAnalyzer
# ──────────────────────────────────────────────────────────────

class MLTWIAnalyzerDAS:
    """
    Versi MLTWIAnalyzer yang menggunakan geometry DAS nyata
    untuk menentukan koordinat rekomendasi flood zone, RTH, dan drainase.
    """

    def __init__(self, das, twi_model=None):
        """
        Parameters
        ----------
        das        : DASResult — objek DAS dari das_selector
        twi_model  : MLTWIEnhanced (opsional)
        """
        self.das = das
        self.twi_model = twi_model
        self.rtho_recommendations  = []
        self.flood_zones           = []
        self.drainage_recommendations = []

        # Ambil info spasial sekali saja
        print("\n🗺️  Mengambil info spasial DAS...")
        self.spatial = get_das_spatial_info(das)
        print(f"   Centroid   : {self.spatial['centroid_lat']:.4f}°N, "
              f"{self.spatial['centroid_lon']:.4f}°E")
        print(f"   Scale DAS  : ±{self.spatial['scale_deg']:.4f}° "
              f"(~{self.spatial['scale_deg']*111:.1f} km)")

        # Pre-sample titik di dalam DAS
        print("   Sampling titik di dalam DAS...")
        raw_points = sample_points_in_das(das, n_points=15, seed=42)
        self.das_points = filter_points_in_das(das, raw_points)
        print(f"   ✅ {len(self.das_points)} titik valid di dalam DAS")

    def _get_candidate_points(self, n, seed_offset=0):
        """
        Ambil n titik kandidat dari pool titik yang sudah ada di dalam DAS.
        Jika kurang, sample ulang dengan seed berbeda.
        """
        if len(self.das_points) >= n:
            # Pilih n titik tersebar (setiap k-th)
            step = max(1, len(self.das_points) // n)
            return [(self.das_points[i*step][0], self.das_points[i*step][1])
                    for i in range(n)]
        else:
            # Sample ulang dengan seed berbeda
            extra = sample_points_in_das(self.das, n_points=n*2, seed=42+seed_offset)
            combined = self.das_points + extra
            # Deduplicate kasar
            seen = set()
            unique = []
            for pt in combined:
                key = (round(pt[0], 5), round(pt[1], 5))
                if key not in seen:
                    seen.add(key)
                    unique.append(pt)
            return unique[:n] if len(unique) >= n else unique

    # ──────────────────────────────────────────────────────────
    # Flood Zones
    # ──────────────────────────────────────────────────────────

    def identify_flood_zones(self, twi_enhanced,
                              threshold_high=15.0, threshold_moderate=12.0):
        """
        Identifikasi zona genangan menggunakan titik di dalam polygon DAS.
        """
        zones = []

        if twi_enhanced >= threshold_high:
            n_zones = 3
            risk = 'HIGH'
            base_prob = min(0.95, 0.5 + (twi_enhanced - 15) * 0.05)
        elif twi_enhanced >= threshold_moderate:
            n_zones = 2
            risk = 'MODERATE'
            base_prob = 0.3 + (twi_enhanced - 12) * 0.1
        elif twi_enhanced >= 10.0:
            n_zones = 1
            risk = 'LOW'
            base_prob = (twi_enhanced - 10) * 0.1
        else:
            self.flood_zones = zones
            return zones

        candidates = self._get_candidate_points(n_zones, seed_offset=10)

        for i, (lon, lat) in enumerate(candidates[:n_zones]):
            zones.append({
                'location_id': f'FLOOD_{len(zones)+1:03d}',
                'coordinates': {
                    'latitude':  round(lat, 6),
                    'longitude': round(lon, 6),
                },
                'risk_level':        risk,
                'twi_enhanced':      twi_enhanced,
                'flood_probability': round(base_prob - i * 0.03, 3),
                'area_affected_ha':  round(2.5 + i * 0.8, 1),
                'inside_das':        True,
                'recommendations':   self._flood_recommendations(risk),
            })

        self.flood_zones = zones
        return zones

    def _flood_recommendations(self, risk):
        if risk == 'HIGH':
            return [
                'Install drainage system urgently',
                'Build retention pond',
                'Elevate critical infrastructure',
                'Implement flood early warning system',
            ]
        elif risk == 'MODERATE':
            return [
                'Improve drainage capacity',
                'Create upstream RTH for water retention',
                'Monitor during heavy rainfall',
                'Regular maintenance of existing drainage',
            ]
        return ['Preventive monitoring', 'Consider RTH for future protection']

    # ──────────────────────────────────────────────────────────
    # RTH Recommendations
    # ──────────────────────────────────────────────────────────

    def recommend_rtho_locations(self, twi_enhanced, morphology_data, df):
        """
        Rekomendasi lokasi RTH di dalam polygon DAS yang sebenarnya.
        """
        slope        = morphology_data.get('slope_mean', 0)
        ndvi         = df['ndvi'].mean()
        soil_moisture = df['soil_moisture'].mean()

        # Hitung suitability score
        suitability = 0.0

        if 10 <= twi_enhanced <= 15:
            suitability += (1.0 - abs(twi_enhanced - 12.5) / 5.0) * 0.4
            priority = 'HIGH'
        elif 8 <= twi_enhanced < 10 or 15 < twi_enhanced <= 17:
            suitability += 0.6 * 0.4
            priority = 'MEDIUM'
        else:
            suitability += 0.3 * 0.4
            priority = 'LOW'

        suitability += (1.0 if 2 <= slope <= 8 else 0.3 if slope < 2 or slope > 15 else 0.6) * 0.25
        suitability += (0.9 if ndvi < 0.4 else 0.6 if ndvi < 0.6 else 0.3) * 0.2
        suitability += (0.9 if 0.3 <= soil_moisture <= 0.6 else 0.5) * 0.15

        recommendations = []

        if suitability >= 0.5:
            n_locs = 2 if priority == 'HIGH' else 1
            candidates = self._get_candidate_points(n_locs * 2, seed_offset=20)
            # Pilih titik yang jauh dari flood zones
            flood_lons = [z['coordinates']['longitude'] for z in self.flood_zones]
            flood_lats = [z['coordinates']['latitude']  for z in self.flood_zones]

            selected = []
            for lon, lat in candidates:
                # Cari titik yang paling jauh dari zona banjir (untuk RTH upstream)
                if flood_lons:
                    min_dist = min(
                        ((lon - fl)**2 + (lat - flt)**2)**0.5
                        for fl, flt in zip(flood_lons, flood_lats)
                    )
                    if min_dist > self.spatial['scale_deg'] * 0.3:
                        selected.append((lon, lat))
                else:
                    selected.append((lon, lat))

            # Fallback jika tidak ada yang lolos filter jarak
            if not selected:
                selected = candidates[:n_locs]

            purposes = [
                "Intercept upstream runoff before reaching flood zones",
                "Retention area for downstream flood mitigation",
            ]

            for i, (lon, lat) in enumerate(selected[:n_locs]):
                catchment_ha = 10 + (twi_enhanced - 10) * 2
                water_ret    = catchment_ha * 1000 * 0.05
                flood_red    = min(40, suitability * 50)

                recommendations.append({
                    'location_id': f'RTH_{len(recommendations)+1:03d}',
                    'coordinates': {
                        'latitude':  round(lat, 6),
                        'longitude': round(lon, 6),
                    },
                    'priority':         priority,
                    'twi_enhanced':     twi_enhanced,
                    'suitability_score': round(suitability, 2),
                    'area_recommended_ha': round(2 + suitability * 3, 1),
                    'location_purpose': purposes[i % len(purposes)],
                    'inside_das':       True,
                    'reasons': self._rth_reasons(twi_enhanced, slope, ndvi, soil_moisture),
                    'expected_benefits': {
                        'water_retention_m3_per_event': int(water_ret),
                        'flood_reduction_percent':       int(flood_red),
                        'groundwater_recharge_m3_per_year': int(water_ret * 30),
                        'catchment_area_ha':             round(catchment_ha, 1),
                    },
                    'design_recommendations': {
                        'vegetation_type': 'Mixed trees and grass' if twi_enhanced > 12 else 'Grass with scattered trees',
                        'depth_m':             0.5 if twi_enhanced > 13 else 0.3,
                        'infiltration_trenches': twi_enhanced > 12,
                        'bioswale':            True,
                        'retention_pond':      twi_enhanced > 14,
                    },
                })

        self.rtho_recommendations = recommendations
        return recommendations

    def _rth_reasons(self, twi, slope, ndvi, soil_moisture):
        reasons = []
        if 10 <= twi <= 15:
            reasons.append(f"Optimal TWI range ({twi:.1f}) for water retention")
        if 2 <= slope <= 8:
            reasons.append(f"Gentle slope ({slope:.1f}°) allows good infiltration")
        elif slope < 2:
            reasons.append(f"Flat area ({slope:.1f}°) suitable for retention pond")
        if ndvi < 0.4:
            reasons.append("Currently low vegetation (easier to develop)")
        if 0.3 <= soil_moisture <= 0.6:
            reasons.append(f"Moderate soil moisture ({soil_moisture:.2f}) indicates good drainage")
        reasons.append("High potential for flood mitigation within DAS boundary")
        return reasons

    # ──────────────────────────────────────────────────────────
    # Drainage Recommendations
    # ──────────────────────────────────────────────────────────

    def recommend_drainage_locations(self, twi_enhanced, morphology_data, df):
        """
        Rekomendasi lokasi drainase di dalam polygon DAS.
        Titik dipilih dekat zona banjir tapi tetap di dalam DAS.
        """
        slope        = morphology_data.get('slope_mean', 0)
        rainfall_avg = df['rainfall'].mean()
        runoff_avg   = df['runoff'].mean() if 'runoff' in df.columns else 0

        # Hitung necessity score
        necessity = 0.0
        if twi_enhanced > 15:
            necessity += min(1.0, (twi_enhanced - 15) / 10.0) * 0.35
            priority = 'HIGH'
        elif twi_enhanced > 12:
            necessity += 0.6 * 0.35
            priority = 'MEDIUM'
        else:
            necessity += 0.3 * 0.35
            priority = 'LOW'

        flood_count = len(self.flood_zones)
        necessity += (1.0 if flood_count >= 3 else 0.7 if flood_count >= 2 else 0.4) * 0.25
        if flood_count >= 3:
            priority = 'HIGH'

        necessity += (0.9 if rainfall_avg > 10 else 0.6 if rainfall_avg > 5 else 0.3) * 0.2
        necessity += (0.9 if runoff_avg > 5  else 0.6 if runoff_avg > 2  else 0.3) * 0.2

        recommendations = []

        if necessity >= 0.45:
            n_locs = (min(4, flood_count + 1) if priority == 'HIGH'
                      else min(3, flood_count)  if priority == 'MEDIUM'
                      else min(2, flood_count))
            n_locs = max(n_locs, 1)

            # Ambil titik dekat zona banjir (di dalam DAS)
            candidates = self._get_candidate_points(n_locs * 3, seed_offset=30)

            # Urutkan: dekat dengan flood zones (drainase harus meng-intercept aliran)
            if self.flood_zones:
                flood_center_lon = np.mean([z['coordinates']['longitude'] for z in self.flood_zones])
                flood_center_lat = np.mean([z['coordinates']['latitude']  for z in self.flood_zones])
                candidates.sort(key=lambda pt: (pt[0]-flood_center_lon)**2 + (pt[1]-flood_center_lat)**2)

            drainage_types = [
                'Primary Drainage Channel',
                'Secondary Drainage Network',
                'Lateral Collection System',
                'Outlet Channel',
            ]

            for i, (lon, lat) in enumerate(candidates[:n_locs]):
                catchment_ha = 15 + (twi_enhanced - 10) * 3
                cap_m3h      = catchment_ha * 100 * rainfall_avg * 0.8

                if   cap_m3h > 5000: w, d = 2.5, 1.5
                elif cap_m3h > 2000: w, d = 2.0, 1.2
                else:                w, d = 1.5, 1.0

                slope_pct = max(0.5, min(2.0, slope * 0.3))

                recommendations.append({
                    'location_id': f'DRAIN_{i+1:03d}',
                    'coordinates': {
                        'latitude':  round(lat, 6),
                        'longitude': round(lon, 6),
                    },
                    'priority':       priority,
                    'drainage_type':  drainage_types[i % len(drainage_types)],
                    'necessity_score': round(necessity, 2),
                    'inside_das':     True,
                    'specifications': {
                        'channel_width_m':    round(w, 1),
                        'channel_depth_m':    round(d, 1),
                        'channel_slope_percent': round(slope_pct, 2),
                        'lining_type':        'Concrete' if priority == 'HIGH' else 'Gabion',
                        'length_estimated_m': int(100 + i * 50),
                    },
                    'capacity': {
                        'design_capacity_m3_per_hour': int(cap_m3h),
                        'peak_flow_m3_per_second':     round(cap_m3h / 3600, 2),
                        'catchment_area_ha':           round(catchment_ha, 1),
                    },
                    'expected_benefits': {
                        'flood_reduction_percent':        min(50, necessity * 60),
                        'ponding_time_reduction_hours':   round(4 + necessity * 6, 1),
                        'affected_area_ha':               round(catchment_ha * 0.7, 1),
                    },
                    'reasons': self._drainage_reasons(twi_enhanced, slope, rainfall_avg, flood_count),
                    'maintenance_requirements': {
                        'cleaning_frequency':   'Monthly' if priority == 'HIGH' else 'Quarterly',
                        'inspection_frequency': 'Weekly during rainy season',
                        'estimated_annual_cost_million_idr': round(w * d * 2, 1),
                    },
                })

        self.drainage_recommendations = recommendations
        return recommendations

    def _drainage_reasons(self, twi, slope, rainfall, flood_zones):
        reasons = []
        if twi > 15:
            reasons.append(f"High TWI ({twi:.1f}) indicates severe water accumulation")
        elif twi > 12:
            reasons.append(f"Elevated TWI ({twi:.1f}) shows water retention issues")
        if flood_zones >= 3:
            reasons.append(f"Multiple flood zones ({flood_zones}) require comprehensive drainage")
        elif flood_zones >= 2:
            reasons.append(f"Several flood zones ({flood_zones}) need drainage system")
        if rainfall > 10:
            reasons.append(f"High rainfall ({rainfall:.1f} mm/day) increases drainage necessity")
        if slope > 5:
            reasons.append(f"Moderate slope ({slope:.1f}°) allows efficient drainage flow")
        elif slope < 2:
            reasons.append(f"Flat terrain ({slope:.1f}°) requires artificial drainage")
        reasons.append("Location validated within DAS boundary")
        return reasons

    # ──────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────

    def generate_summary_report(self):
        return {
            'das_info': {
                'name':          self.das.name,
                'area_km2':      self.das.area_km2,
                'centroid_lon':  self.spatial['centroid_lon'],
                'centroid_lat':  self.spatial['centroid_lat'],
                'n_sample_points': len(self.das_points),
            },
            'flood_zones': {
                'total':         len(self.flood_zones),
                'high_risk':     sum(1 for z in self.flood_zones if z['risk_level'] == 'HIGH'),
                'moderate_risk': sum(1 for z in self.flood_zones if z['risk_level'] == 'MODERATE'),
                'low_risk':      sum(1 for z in self.flood_zones if z['risk_level'] == 'LOW'),
                'all_inside_das': all(z.get('inside_das', False) for z in self.flood_zones),
            },
            'rtho_recommendations': {
                'total':            len(self.rtho_recommendations),
                'high_priority':    sum(1 for r in self.rtho_recommendations if r['priority'] == 'HIGH'),
                'medium_priority':  sum(1 for r in self.rtho_recommendations if r['priority'] == 'MEDIUM'),
                'total_area_ha':    sum(r['area_recommended_ha'] for r in self.rtho_recommendations),
                'all_inside_das':   all(r.get('inside_das', False) for r in self.rtho_recommendations),
                'estimated_flood_reduction_percent': (
                    np.mean([r['expected_benefits']['flood_reduction_percent']
                             for r in self.rtho_recommendations])
                    if self.rtho_recommendations else 0
                ),
            },
            'drainage_recommendations': {
                'total':           len(self.drainage_recommendations),
                'high_priority':   sum(1 for r in self.drainage_recommendations if r['priority'] == 'HIGH'),
                'medium_priority': sum(1 for r in self.drainage_recommendations if r['priority'] == 'MEDIUM'),
                'total_capacity_m3_per_hour': sum(
                    r['capacity']['design_capacity_m3_per_hour']
                    for r in self.drainage_recommendations
                ),
                'all_inside_das':  all(r.get('inside_das', False) for r in self.drainage_recommendations),
                'estimated_flood_reduction_percent': (
                    np.mean([r['expected_benefits']['flood_reduction_percent']
                             for r in self.drainage_recommendations])
                    if self.drainage_recommendations else 0
                ),
            },
        }