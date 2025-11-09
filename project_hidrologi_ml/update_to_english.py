# -*- coding: utf-8 -*-
"""
Quick script to replace Indonesian chart titles with English in main_weap_ml.py
Run this once to update all visualization titles to English
"""

import re

# Mapping Indonesian titles to English
TITLE_REPLACEMENTS = {
    # Main Dashboard
    "'📦 STATUS VOLUME KOLAM RETENSI'": "'📦 RETENTION POND VOLUME STATUS'",
    "'⚖️ KESEIMBANGAN PASOKAN DAN PERMINTAAN AIR'": "'⚖️ WATER SUPPLY AND DEMAND BALANCE'",
    "'🥧 DISTRIBUSI ALOKASI AIR'": "'🥧 WATER ALLOCATION DISTRIBUTION'",
    "'🌧️ CURAH HUJAN & PREDIKSI'": "'🌧️ RAINFALL & FORECAST'",
    "'⚠️ ANALISIS RISIKO'": "'⚠️ RISK ANALYSIS'",
    "'🎯 REKOMENDASI OPERASI KOLAM RETENSI (ML)'": "'🎯 RETENTION POND OPERATION RECOMMENDATIONS (ML)'",
    
    # Enhanced Dashboard
    "'⚖️ ALOKASI BERDASARKAN HAK AIR & PRIORITAS'": "'⚖️ ALLOCATION BASED ON WATER RIGHTS & PRIORITIES'",
    "'🌊 DISTRIBUSI JARINGAN PASOKAN'": "'🌊 SUPPLY NETWORK DISTRIBUTION'",
    "'💰 ANALISIS BIAYA-MANFAAT'": "'💰 COST-BENEFIT ANALYSIS'",
    "'⚡ KONSUMSI ENERGI'": "'⚡ ENERGY CONSUMPTION'",
    "'💧 TINGKAT KUALITAS AIR'": "'💧 WATER QUALITY LEVEL'",
    "'🔬 PARAMETER KUALITAS AIR'": "'🔬 WATER QUALITY PARAMETERS'",
    "'📈 RASIO EFISIENSI (Benefit/Cost)'": "'📈 EFFICIENCY RATIO (Benefit/Cost)'",
    "'💵 DISTRIBUSI BIAYA JARINGAN'": "'💵 NETWORK COST DISTRIBUTION'",
    
    # Morphology & Ecology
    "'🏔️ PERPINDAHAN TANAH'": "'🏔️ SEDIMENT TRANSPORT'",
    "'⚖️ EROSI vs DEPOSISI'": "'⚖️ EROSION vs DEPOSITION'",
    "'🌊 PERUBAHAN GEOMETRI CHANNEL'": "'🌊 CHANNEL GEOMETRY CHANGES'",
    "'🐟 TINGKAT KESESUAIAN HABITAT'": "'🐟 HABITAT SUITABILITY LEVEL'",
    "'🌿 INDEKS KESEHATAN EKOSISTEM'": "'🌿 ECOSYSTEM HEALTH INDEX'",
    "'💧 PERUBAHAN POLA ALIRAN AIR'": "'💧 FLOW PATTERN CHANGES'",
    
    # Water Balance
    "'📊 TOTAL KESEIMBANGAN AIR'": "'📊 TOTAL WATER BALANCE'",
    
    # River Map
    "'🌊 PETA JARINGAN ALIRAN SUNGAI'": "'🌊 RIVER NETWORK MAP'",
}

# Axis label replacements
AXIS_REPLACEMENTS = {
    "'Hujan (mm/hari)'": "'Rainfall (mm/day)'",
    "'Tingkat Risiko (%)'": "'Risk Level (%)'",
    "'Aksi'": "'Action'",
    "'Pasokan Rata-rata (mm/hari)'": "'Average Supply (mm/day)'",
    "'Energi (kWh/hari)'": "'Energy (kWh/day)'",
}

def update_main_weap_ml():
    """Update main_weap_ml.py with English titles"""
    file_path = 'main_weap_ml.py'
    
    print("Reading main_weap_ml.py...")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Replace titles
    print("\nReplacing chart titles...")
    for indo, eng in TITLE_REPLACEMENTS.items():
        if indo in content:
            content = content.replace(indo, eng)
            print(f"  ✓ Replaced: {indo[:50]}...")
    
    # Replace axis labels
    print("\nReplacing axis labels...")
    for indo, eng in AXIS_REPLACEMENTS.items():
        if indo in content:
            content = content.replace(indo, eng)
            print(f"  ✓ Replaced: {indo[:50]}...")
    
    # Save backup
    print("\nCreating backup...")
    with open('main_weap_ml.py.backup', 'w', encoding='utf-8') as f:
        f.write(original_content)
    print("  ✓ Backup saved as main_weap_ml.py.backup")
    
    # Write updated content
    print("\nWriting updated file...")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("  ✓ main_weap_ml.py updated successfully!")
    
    print("\n" + "="*60)
    print("✅ UPDATE COMPLETE!")
    print("="*60)
    print(f"Total replacements: {len(TITLE_REPLACEMENTS) + len(AXIS_REPLACEMENTS)}")
    print("\nAll chart titles and axis labels are now in English.")
    print("Backup saved as: main_weap_ml.py.backup")

if __name__ == "__main__":
    import os
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    update_main_weap_ml()
