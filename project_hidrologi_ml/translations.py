# -*- coding: utf-8 -*-
"""
Translation Module for RIVANA Hydrology ML
Supports Indonesian and English translations for charts, labels, and outputs
"""

# Chart Title Translations
CHART_TITLES = {
    'id': {
        'retention_pond_status': '📦 STATUS VOLUME KOLAM RETENSI',
        'water_supply_demand': '⚖️ KESEIMBANGAN PASOKAN DAN PERMINTAAN AIR',
        'water_allocation': '🥧 DISTRIBUSI ALOKASI AIR',
        'rainfall_forecast': '🌧️ CURAH HUJAN & PREDIKSI',
        'risk_analysis': '⚠️ ANALISIS RISIKO',
        'pond_recommendation': '🎯 REKOMENDASI OPERASI KOLAM RETENSI (ML)',
        'water_rights': '⚖️ ALOKASI BERDASARKAN HAK AIR & PRIORITAS',
        'supply_network': '🌊 DISTRIBUSI JARINGAN PASOKAN',
        'cost_benefit': '💰 ANALISIS BIAYA-MANFAAT',
        'energy_consumption': '⚡ KONSUMSI ENERGI',
        'water_quality': '💧 TINGKAT KUALITAS AIR',
        'quality_parameters': '🔬 PARAMETER KUALITAS AIR',
        'efficiency_ratio': '📈 RASIO EFISIENSI (Benefit/Cost)',
        'cost_distribution': '💵 DISTRIBUSI BIAYA JARINGAN',
        'sediment_transport': '🏔️ PERPINDAHAN TANAH',
        'erosion_deposition': '⚖️ EROSI vs DEPOSISI',
        'channel_geometry': '🌊 PERUBAHAN GEOMETRI CHANNEL',
        'habitat_suitability': '🐟 TINGKAT KESESUAIAN HABITAT',
        'ecosystem_health': '🌿 INDEKS KESEHATAN EKOSISTEM',
        'flow_pattern': '💧 PERUBAHAN POLA ALIRAN AIR',
        'water_balance': '📊 TOTAL KESEIMBANGAN AIR',
        'residual_error': '📉 DAILY RESIDUAL ERROR (ε)',
        'error_distribution': '📊 ERROR DISTRIBUTION',
        'river_network_map': '🌊 PETA JARINGAN ALIRAN SUNGAI',
    },
    'en': {
        'retention_pond_status': '📦 RETENTION POND VOLUME STATUS',
        'water_supply_demand': '⚖️ WATER SUPPLY AND DEMAND BALANCE',
        'water_allocation': '🥧 WATER ALLOCATION DISTRIBUTION',
        'rainfall_forecast': '🌧️ RAINFALL & FORECAST',
        'risk_analysis': '⚠️ RISK ANALYSIS',
        'pond_recommendation': '🎯 RETENTION POND OPERATION RECOMMENDATIONS (ML)',
        'water_rights': '⚖️ ALLOCATION BASED ON WATER RIGHTS & PRIORITIES',
        'supply_network': '🌊 SUPPLY NETWORK DISTRIBUTION',
        'cost_benefit': '💰 COST-BENEFIT ANALYSIS',
        'energy_consumption': '⚡ ENERGY CONSUMPTION',
        'water_quality': '💧 WATER QUALITY LEVEL',
        'quality_parameters': '🔬 WATER QUALITY PARAMETERS',
        'efficiency_ratio': '📈 EFFICIENCY RATIO (Benefit/Cost)',
        'cost_distribution': '💵 NETWORK COST DISTRIBUTION',
        'sediment_transport': '🏔️ SEDIMENT TRANSPORT',
        'erosion_deposition': '⚖️ EROSION vs DEPOSITION',
        'channel_geometry': '🌊 CHANNEL GEOMETRY CHANGES',
        'habitat_suitability': '🐟 HABITAT SUITABILITY LEVEL',
        'ecosystem_health': '🌿 ECOSYSTEM HEALTH INDEX',
        'flow_pattern': '💧 FLOW PATTERN CHANGES',
        'water_balance': '📊 TOTAL WATER BALANCE',
        'residual_error': '📉 DAILY RESIDUAL ERROR (ε)',
        'error_distribution': '📊 ERROR DISTRIBUTION',
        'river_network_map': '🌊 RIVER NETWORK MAP',
    }
}

# Axis Labels Translations
AXIS_LABELS = {
    'id': {
        'volume_mm': 'Volume (mm)',
        'volume_mm_day': 'Volume (mm/hari)',
        'rainfall_mm_day': 'Hujan (mm/hari)',
        'risk_level_pct': 'Tingkat Risiko (%)',
        'action': 'Aksi',
        'sediment_load': 'Sediment Load (mg/L)',
        'rate_ton_ha_day': 'Rate (ton/ha/day)',
        'width_m': 'Width (m)',
        'depth_m': 'Depth (m)',
        'hsi': 'HSI (0-1)',
        'health_index_pct': 'Health Index (%)',
        'index_pct': 'Index (%)',
        'cumulative_mm': 'Cumulative (mm)',
        'residual_mm': 'Residual (mm)',
        'error_pct': 'Error (%)',
        'frequency': 'Frequency',
        'energy_kwh_day': 'Energi (kWh/hari)',
        'efficiency_ratio': 'Efficiency Ratio',
        'supply_avg': 'Pasokan Rata-rata (mm/hari)',
    },
    'en': {
        'volume_mm': 'Volume (mm)',
        'volume_mm_day': 'Volume (mm/day)',
        'rainfall_mm_day': 'Rainfall (mm/day)',
        'risk_level_pct': 'Risk Level (%)',
        'action': 'Action',
        'sediment_load': 'Sediment Load (mg/L)',
        'rate_ton_ha_day': 'Rate (ton/ha/day)',
        'width_m': 'Width (m)',
        'depth_m': 'Depth (m)',
        'hsi': 'HSI (0-1)',
        'health_index_pct': 'Health Index (%)',
        'index_pct': 'Index (%)',
        'cumulative_mm': 'Cumulative (mm)',
        'residual_mm': 'Residual (mm)',
        'error_pct': 'Error (%)',
        'frequency': 'Frequency',
        'energy_kwh_day': 'Energy (kWh/day)',
        'efficiency_ratio': 'Efficiency Ratio',
        'supply_avg': 'Average Supply (mm/day)',
    }
}

# Sector Names
SECTORS = {
    'id': {
        'Domestik': 'Domestik',
        'Pertanian': 'Pertanian',
        'Industri': 'Industri',
        'Lingkungan': 'Lingkungan',
    },
    'en': {
        'Domestik': 'Domestic',
        'Pertanian': 'Agriculture',
        'Industri': 'Industry',
        'Lingkungan': 'Environment',
    }
}

# Water Sources
SOURCES = {
    'id': {
        'Sungai': 'Sungai',
        'Diversi': 'Diversi',
        'Air Tanah': 'Air Tanah',
    },
    'en': {
        'Sungai': 'River',
        'Diversi': 'Diversion',
        'Air Tanah': 'Groundwater',
    }
}

# Status Labels
STATUS = {
    'id': {
        'surplus': 'Surplus',
        'deficit': 'Defisit',
        'balanced': 'Seimbang',
        'excellent': 'Sangat Baik',
        'good': 'Baik',
        'fair': 'Cukup',
        'poor': 'Kurang',
        'bad': 'Buruk',
        'high': 'Tinggi',
        'medium': 'Sedang',
        'low': 'Rendah',
    },
    'en': {
        'surplus': 'Surplus',
        'deficit': 'Deficit',
        'balanced': 'Balanced',
        'excellent': 'Excellent',
        'good': 'Good',
        'fair': 'Fair',
        'poor': 'Poor',
        'bad': 'Bad',
        'high': 'High',
        'medium': 'Medium',
        'low': 'Low',
    }
}

# Recommendations
RECOMMENDATIONS = {
    'id': {
        'critical_retention': 'Kapasitas Kolam Retensi kritis',
        'low_reliability': 'Keandalan sistem rendah',
        'implement_rationing': 'Terapkan rationing air segera',
        'audit_infrastructure': 'Audit infrastruktur, kurangi kebocoran',
        'evaluate_capacity': 'Evaluasi kapasitas Kolam Retensi, pertimbangkan peningkatan/penambahan',
        'optimize_distribution': 'Optimalkan sistem distribusi untuk reduce losses',
        'demand_management': 'Implementasi demand management (rationing, pricing)',
        'alternative_sources': 'Explore alternative water sources (groundwater, recycled water)',
        'repair_leaks': 'Perbaiki kebocoran pipa (leak detection & repair)',
        'upgrade_pumps': 'Upgrade pompa dan valve yang sudah tua',
        'implement_scada': 'Implementasi SCADA system untuk monitoring real-time',
    },
    'en': {
        'critical_retention': 'Critical retention pond capacity',
        'low_reliability': 'Low system reliability',
        'implement_rationing': 'Implement water rationing immediately',
        'audit_infrastructure': 'Audit infrastructure, reduce leakage',
        'evaluate_capacity': 'Evaluate retention pond capacity, consider upgrading/expansion',
        'optimize_distribution': 'Optimize distribution system to reduce losses',
        'demand_management': 'Implement demand management (rationing, pricing)',
        'alternative_sources': 'Explore alternative water sources (groundwater, recycled water)',
        'repair_leaks': 'Repair pipe leaks (leak detection & repair)',
        'upgrade_pumps': 'Upgrade old pumps and valves',
        'implement_scada': 'Implement SCADA system for real-time monitoring',
    }
}

def get_text(key, lang='en', category='CHART_TITLES'):
    """
    Get translated text
    
    Args:
        key: Translation key
        lang: Language code ('id' or 'en')
        category: Category of translation (CHART_TITLES, AXIS_LABELS, etc.)
    
    Returns:
        Translated text or key if not found
    """
    categories = {
        'CHART_TITLES': CHART_TITLES,
        'AXIS_LABELS': AXIS_LABELS,
        'SECTORS': SECTORS,
        'SOURCES': SOURCES,
        'STATUS': STATUS,
        'RECOMMENDATIONS': RECOMMENDATIONS,
    }
    
    if category not in categories:
        return key
    
    translation_dict = categories[category]
    
    if lang not in translation_dict:
        lang = 'en'  # Default to English
    
    return translation_dict[lang].get(key, key)

def translate_sector(sector_name, lang='en'):
    """Translate sector name"""
    return get_text(sector_name, lang, 'SECTORS')

def translate_source(source_name, lang='en'):
    """Translate water source name"""
    return get_text(source_name, lang, 'SOURCES')

def translate_status(status_name, lang='en'):
    """Translate status label"""
    return get_text(status_name, lang, 'STATUS')
