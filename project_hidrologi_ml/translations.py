"""
RIVANA Translations Module
Supports: Indonesian ('id') and English ('en')
Usage:
    from translations import get_text, T
    
    # Option 1: Direct call
    print(get_text('system_title', lang='en'))
    
    # Option 2: Short alias (reads RIVANA_LANG env var automatically)
    print(T('system_title'))
"""

import os

TRANSLATIONS = {
    # ==========================================
    # SYSTEM / GENERAL
    # ==========================================
    'system_title': {
        'id': 'SISTEM MANAJEMEN AIR TERPADU (RIVANA)',
        'en': 'INTEGRATED WATER MANAGEMENT SYSTEM (RIVANA)',
    },
    'system_subtitle': {
        'id': 'Water Evaluation And Planning dengan Machine Learning',
        'en': 'Water Evaluation And Planning with Machine Learning',
    },
    'analysis_started': {
        'id': 'MEMULAI ANALISIS ML',
        'en': 'STARTING ML ANALYSIS',
    },
    'analysis_complete': {
        'id': 'ANALISIS SELESAI!',
        'en': 'ANALYSIS COMPLETE!',
    },
    'saving_results': {
        'id': 'MENYIMPAN HASIL',
        'en': 'SAVING RESULTS',
    },
    'output_summary': {
        'id': 'RINGKASAN OUTPUT',
        'en': 'OUTPUT SUMMARY',
    },
    'thank_you': {
        'id': 'Terima kasih telah menggunakan RIVANA',
        'en': 'Thank you for using RIVANA',
    },
    'summary_title': {
        'id': 'Ringkasan Analisis Hidrologi RIVANA',
        'en': 'RIVANA Hydrological Analysis Summary',
    },
    'yes': {
        'id': 'Ya',
        'en': 'Yes',
    },
    'no': {
        'id': 'Tidak',
        'en': 'No',
    },
    'data_not_available': {
        'id': 'Data tidak tersedia',
        'en': 'N/A',
    },
    'file_not_available': {
        'id': 'File tidak tersedia',
        'en': 'File not available',
    },
    'not_available_for_job': {
        'id': 'Tidak tersedia untuk job ini',
        'en': 'Not available for this job',
    },
    'data_incomplete': {
        'id': 'Data tidak lengkap',
        'en': 'Data incomplete',
    },

    # ==========================================
    # SATELLITE DATA
    # ==========================================
    'fetch_satellite': {
        'id': 'MENGAMBIL DATA SATELIT',
        'en': 'FETCHING SATELLITE DATA',
    },
    'location_label': {
        'id': 'Lokasi',
        'en': 'Location',
    },
    'period_label': {
        'id': 'Periode',
        'en': 'Period',
    },
    'data_downloaded': {
        'id': 'Data berhasil diunduh',
        'en': 'Data successfully downloaded',
    },
    'rainfall_avg': {
        'id': 'Curah hujan rata-rata',
        'en': 'Average rainfall',
    },
    'temp_avg': {
        'id': 'Suhu rata-rata',
        'en': 'Average temperature',
    },
    'soil_moisture_label': {
        'id': 'Kelembaban Tanah',
        'en': 'Soil Moisture',
    },

    # ==========================================
    # MORPHOLOGY
    # ==========================================
    'fetch_morphology': {
        'id': 'MENGAMBIL DATA MORFOLOGI',
        'en': 'FETCHING MORPHOLOGY DATA',
    },
    'morphology_fetched': {
        'id': 'Data morfologi berhasil diambil',
        'en': 'Morphology data successfully retrieved',
    },
    'relief_label': {
        'id': 'Relief',
        'en': 'Relief',
    },
    'slope_avg': {
        'id': 'Kemiringan rata-rata',
        'en': 'Average slope',
    },
    'twi_avg': {
        'id': 'TWI rata-rata',
        'en': 'Average TWI',
    },
    'twi_range': {
        'id': 'Rentang TWI',
        'en': 'TWI range',
    },

    # ==========================================
    # ML TRAINING LABELS
    # ==========================================
    'training_label_gen': {
        'id': 'Melatih Pembuat Label...',
        'en': 'Training Label Generator...',
    },
    'label_gen_trained': {
        'id': 'Pembuat Label Terlatih',
        'en': 'Label Generator Trained',
    },
    'training_et': {
        'id': 'Melatih penghitung penguapan air...',
        'en': 'Training evapotranspiration estimator...',
    },
    'et_trained': {
        'id': 'Penghitung penguapan air berhasil dilatih',
        'en': 'ET estimator successfully trained',
    },

    # ==========================================
    # HYDRO SIMULATOR
    # ==========================================
    'train_hydro': {
        'id': 'MELATIH MODEL PERGERAKAN AIR',
        'en': 'TRAINING HYDROLOGICAL MODEL',
    },
    'data_splitting': {
        'id': 'PEMBAGIAN DATA',
        'en': 'DATA SPLITTING',
    },
    'training_label': {
        'id': 'Training',
        'en': 'Training',
    },
    'validation_label': {
        'id': 'Validasi',
        'en': 'Validation',
    },
    'test_label': {
        'id': 'Test',
        'en': 'Test',
    },
    'samples_label': {
        'id': 'sampel',
        'en': 'samples',
    },
    'training_model': {
        'id': 'Melatih model aliran air canggih...',
        'en': 'Training advanced water flow model...',
    },
    'hydro_trained': {
        'id': 'Model Hidrologi Terlatih',
        'en': 'Hydrological Model Trained',
    },

    # ==========================================
    # SUPPLY DEMAND
    # ==========================================
    'train_supply_demand': {
        'id': 'MELATIH PENYEIMBANG KETERSEDIAAN & KEBUTUHAN AIR',
        'en': 'TRAINING WATER SUPPLY & DEMAND BALANCER',
    },
    'supply_demand_trained': {
        'id': 'Supply-Demand Optimizer Terlatih',
        'en': 'Supply-Demand Optimizer Trained',
    },

    # ==========================================
    # FLOOD & DROUGHT
    # ==========================================
    'train_flood_drought': {
        'id': 'MELATIH PERAMAL BANJIR & KEKERINGAN',
        'en': 'TRAINING FLOOD & DROUGHT PREDICTOR',
    },
    'flood_drought_trained': {
        'id': 'Flood & Drought Predictor Terlatih',
        'en': 'Flood & Drought Predictor Trained',
    },

    # ==========================================
    # RESERVOIR ADVISOR
    # ==========================================
    'train_reservoir': {
        'id': 'MELATIH PENASIHAT PENGELOLAAN KOLAM RETENSI',
        'en': 'TRAINING RETENTION POND MANAGEMENT ADVISOR',
    },
    'reservoir_trained': {
        'id': 'Reservoir Advisor Terlatih',
        'en': 'Reservoir Advisor Trained',
    },

    # ==========================================
    # SEDIMENT
    # ==========================================
    'train_sediment': {
        'id': 'MELATIH MODEL PERGERAKAN TANAH',
        'en': 'TRAINING SEDIMENT TRANSPORT MODEL',
    },
    'training_sediment': {
        'id': 'Melatih model pergerakan sedimen...',
        'en': 'Training sediment transport model...',
    },
    'sediment_trained': {
        'id': 'Sediment Transport Model Terlatih',
        'en': 'Sediment Transport Model Trained',
    },

    # ==========================================
    # TWI
    # ==========================================
    'calculating_twi': {
        'id': 'MENGHITUNG TWI YANG DITINGKATKAN',
        'en': 'CALCULATING ENHANCED TWI',
    },
    'twi_training': {
        'id': 'Training ML TWI Enhancement Model...',
        'en': 'Training ML TWI Enhancement Model...',
    },
    'twi_trained': {
        'id': 'Model terlatih berhasil',
        'en': 'Model trained successfully',
    },
    'twi_results': {
        'id': 'Hasil Analisis TWI',
        'en': 'TWI Analysis Results',
    },
    'twi_physics': {
        'id': 'TWI Berbasis Fisika',
        'en': 'Physics-based TWI',
    },
    'twi_correction': {
        'id': 'Faktor Koreksi ML',
        'en': 'ML Correction Factor',
    },
    'twi_enhanced': {
        'id': 'TWI yang Ditingkatkan',
        'en': 'Enhanced TWI',
    },
    'twi_risk': {
        'id': 'Tingkat Risiko',
        'en': 'Risk Level',
    },
    'twi_complete': {
        'id': 'Analisis TWI Selesai',
        'en': 'TWI Analysis Complete',
    },
    'twi_high_flood_risk': {
        'id': 'Lokasi ini memiliki risiko banjir tinggi. Diperlukan tindakan mitigasi segera.',
        'en': 'This location has high flood risk. Immediate mitigation action is required.',
    },
    'twi_good_drainage': {
        'id': 'Drainase lokasi ini relatif baik. Pemantauan rutin tetap disarankan.',
        'en': 'Drainage at this location is relatively good. Regular monitoring is still recommended.',
    },
    'twi_action_mitigate': {
        'id': 'Implementasikan sistem drainase dan RTH untuk mengurangi risiko banjir.',
        'en': 'Implement drainage systems and green spaces to reduce flood risk.',
    },
    'twi_action_monitor': {
        'id': 'Lanjutkan pemantauan berkala dan pertahankan infrastruktur drainase yang ada.',
        'en': 'Continue periodic monitoring and maintain existing drainage infrastructure.',
    },
    'flood_zones_label': {
        'id': 'Zona Risiko Banjir',
        'en': 'Flood Risk Zones',
    },
    'high_risk': {
        'id': 'Risiko Tinggi',
        'en': 'High Risk',
    },
    'moderate_risk': {
        'id': 'Risiko Sedang',
        'en': 'Moderate Risk',
    },
    'rth_recommendations': {
        'id': 'Rekomendasi RTH',
        'en': 'RTH Recommendations',
    },
    'high_priority': {
        'id': 'Prioritas Tinggi',
        'en': 'High Priority',
    },
    'total_area': {
        'id': 'Total Area',
        'en': 'Total Area',
    },
    'drainage_recommendations': {
        'id': 'Rekomendasi Drainase',
        'en': 'Drainage Recommendations',
    },
    'total_capacity': {
        'id': 'Total Kapasitas',
        'en': 'Total Capacity',
    },

    # ==========================================
    # WATER BALANCE
    # ==========================================
    'water_balance_calc': {
        'id': 'MENGHITUNG KESEIMBANGAN AIR',
        'en': 'CALCULATING WATER BALANCE',
    },
    'mass_conservation': {
        'id': 'MEMERIKSA KEKEKALAN JUMLAH AIR',
        'en': 'CHECKING MASS CONSERVATION',
    },
    'water_balance_done': {
        'id': 'Water Balance Terhitung untuk',
        'en': 'Water Balance Calculated for',
    },
    'days_label': {
        'id': 'hari',
        'en': 'days',
    },
    'total_budget': {
        'id': 'TOTAL BUDGET (mm)',
        'en': 'TOTAL BUDGET (mm)',
    },
    'input_label': {
        'id': 'Input (P)',
        'en': 'Input (P)',
    },
    'output_label': {
        'id': 'Output (ET+R+ΔS)',
        'en': 'Output (ET+R+ΔS)',
    },
    'residual_label': {
        'id': 'Sisa (ε)',
        'en': 'Residual (ε)',
    },
    'validation_passed': {
        'id': 'VALIDASI LULUS - Kekekalan massa memenuhi standar jurnal!',
        'en': 'VALIDATION PASSED - Mass conservation meets journal standards!',
    },
    'validation_warning': {
        'id': 'PERINGATAN - Error melebihi toleransi standar jurnal (5%)',
        'en': 'WARNING - Error exceeds journal standard tolerance (5%)',
    },
    'water_balance_complete': {
        'id': 'Keseimbangan air telah dihitung dengan baik',
        'en': 'Water balance has been calculated successfully',
    },
    'validation_file_not_available': {
        'id': 'File validasi tidak tersedia',
        'en': 'Validation file not available',
    },
    'validation_valid': {
        'id': 'Valid - Kekekalan massa terpenuhi',
        'en': 'Valid - Mass conservation satisfied',
    },
    'validation_needs_review': {
        'id': 'Perlu ditinjau - Error melebihi toleransi',
        'en': 'Needs review - Error exceeds tolerance',
    },
    'mass_balance_maintained': {
        'id': 'Keseimbangan massa terjaga dengan baik',
        'en': 'Mass balance is well maintained',
    },
    'needs_recalibration': {
        'id': 'Diperlukan kalibrasi ulang model',
        'en': 'Model recalibration is required',
    },
    'csv_data_not_available': {
        'id': 'Data CSV tidak tersedia',
        'en': 'CSV data not available',
    },

    # ==========================================
    # VALIDATION / MODEL
    # ==========================================
    'model_validation': {
        'id': 'VALIDASI MODEL',
        'en': 'MODEL VALIDATION',
    },
    'cross_validation': {
        'id': 'VALIDASI SILANG',
        'en': 'CROSS-VALIDATION',
    },
    'overall_pass': {
        'id': 'PENILAIAN KESELURUHAN: KINERJA MODEL DAPAT DITERIMA',
        'en': 'OVERALL ASSESSMENT: MODEL PERFORMANCE ACCEPTABLE',
    },
    'overall_marginal': {
        'id': 'PENILAIAN KESELURUHAN: KINERJA MODEL BATAS',
        'en': 'OVERALL ASSESSMENT: MODEL PERFORMANCE MARGINAL',
    },
    'overall_fail': {
        'id': 'PENILAIAN KESELURUHAN: KINERJA MODEL TIDAK MEMUASKAN',
        'en': 'OVERALL ASSESSMENT: MODEL PERFORMANCE UNSATISFACTORY',
    },
    'performance_metrics': {
        'id': 'METRIK KINERJA',
        'en': 'PERFORMANCE METRICS',
    },
    'very_good': {
        'id': 'SANGAT BAIK',
        'en': 'VERY GOOD',
    },
    'good_label': {
        'id': 'BAIK',
        'en': 'GOOD',
    },
    'satisfactory': {
        'id': 'MEMUASKAN',
        'en': 'SATISFACTORY',
    },
    'acceptable': {
        'id': 'DAPAT DITERIMA',
        'en': 'ACCEPTABLE',
    },
    'unsatisfactory': {
        'id': 'TIDAK MEMUASKAN',
        'en': 'UNSATISFACTORY',
    },
    'excellent_label': {
        'id': 'SANGAT BAGUS',
        'en': 'EXCELLENT',
    },
    'recalibration_rec': {
        'id': 'Rekomendasi: Pertimbangkan kalibrasi ulang model',
        'en': 'Recommendation: Consider model recalibration',
    },
    'model_consistent': {
        'id': 'Model KONSISTEN di semua lipatan',
        'en': 'Model is CONSISTENT across folds',
    },
    'model_variable': {
        'id': 'Model menunjukkan VARIABILITAS di semua lipatan',
        'en': 'Model shows VARIABILITY across folds',
    },

    # ==========================================
    # BASELINE COMPARISON
    # ==========================================
    'baseline_comparison': {
        'id': 'PERBANDINGAN BASELINE: ML vs METODE TRADISIONAL',
        'en': 'BASELINE COMPARISON: ML vs TRADITIONAL METHODS',
    },
    'calc_traditional': {
        'id': 'Menghitung metode tradisional...',
        'en': 'Calculating traditional methods...',
    },
    'comparing_perf': {
        'id': 'Membandingkan performa ML vs Baseline...',
        'en': 'Comparing ML vs Baseline performance...',
    },
    'baseline_conclusion': {
        'id': 'KESIMPULAN PERBANDINGAN BASELINE',
        'en': 'BASELINE COMPARISON CONCLUSION',
    },
    'publication_ready': {
        'id': 'MODEL SIAP DIPUBLIKASIKAN!',
        'en': 'MODEL IS PUBLICATION READY!',
    },
    'needs_refinement': {
        'id': 'Model perlu penyempurnaan lebih lanjut sebelum publikasi',
        'en': 'Model needs further refinement before publication',
    },
    'improvement_analysis': {
        'id': 'ANALISIS PENINGKATAN',
        'en': 'IMPROVEMENT ANALYSIS',
    },
    'avg_improvement': {
        'id': 'Rata-rata Peningkatan NSE',
        'en': 'Average NSE Improvement',
    },
    'ml_outperforms_excellent': {
        'id': 'SANGAT BAIK: ML jauh mengungguli metode tradisional',
        'en': 'EXCELLENT: ML significantly outperforms traditional methods',
    },
    'ml_outperforms_very_good': {
        'id': 'SANGAT BAIK: ML menunjukkan peningkatan substansial',
        'en': 'VERY GOOD: ML shows substantial improvement',
    },
    'ml_outperforms_good': {
        'id': 'BAIK: ML menunjukkan peningkatan sedang',
        'en': 'GOOD: ML shows moderate improvement',
    },
    'ml_outperforms_marginal': {
        'id': 'MARGINAL: ML menunjukkan sedikit peningkatan',
        'en': 'MARGINAL: ML shows slight improvement',
    },
    'ml_no_outperform': {
        'id': 'PERINGATAN: ML tidak mengungguli baseline',
        'en': 'WARNING: ML does not outperform baselines',
    },

    # ==========================================
    # STATUS / CONDITION LABELS
    # ==========================================
    'status_normal': {
        'id': 'Normal',
        'en': 'Normal',
    },
    'status_good': {
        'id': 'Baik',
        'en': 'Good',
    },
    'status_fair': {
        'id': 'Cukup',
        'en': 'Fair',
    },
    'status_low': {
        'id': 'Rendah',
        'en': 'Low',
    },
    'status_medium': {
        'id': 'Sedang',
        'en': 'Medium',
    },
    'status_high': {
        'id': 'Tinggi',
        'en': 'High',
    },
    'needs_monitoring': {
        'id': 'Perlu Pemantauan',
        'en': 'Needs Monitoring',
    },

    # ==========================================
    # RELIABILITY STATUS
    # ==========================================
    'reliability_excellent': {
        'id': 'Sangat Andal (≥90%)',
        'en': 'Excellent Reliability (≥90%)',
    },
    'reliability_good': {
        'id': 'Andal (75-90%)',
        'en': 'Good Reliability (75-90%)',
    },
    'reliability_fair': {
        'id': 'Cukup Andal (60-75%)',
        'en': 'Fair Reliability (60-75%)',
    },
    'reliability_poor': {
        'id': 'Kurang Andal (<60%)',
        'en': 'Poor Reliability (<60%)',
    },

    # ==========================================
    # SUPPLY STATUS
    # ==========================================
    'supply_surplus': {
        'id': 'Surplus',
        'en': 'Surplus',
    },
    'supply_balanced': {
        'id': 'Seimbang',
        'en': 'Balanced',
    },
    'supply_deficit': {
        'id': 'Defisit',
        'en': 'Deficit',
    },
    'supply_surplus_detail': {
        'id': 'Surplus - Pasokan melebihi kebutuhan',
        'en': 'Surplus - Supply exceeds demand',
    },
    'supply_balanced_detail': {
        'id': 'Seimbang - Pasokan memenuhi kebutuhan',
        'en': 'Balanced - Supply meets demand',
    },
    'supply_deficit_detail': {
        'id': 'Defisit - Pasokan tidak mencukupi',
        'en': 'Deficit - Supply is insufficient',
    },

    # ==========================================
    # RISK STATUS
    # ==========================================
    'risk_high': {
        'id': 'Tinggi',
        'en': 'High',
    },
    'risk_medium': {
        'id': 'Sedang',
        'en': 'Medium',
    },
    'risk_low': {
        'id': 'Rendah',
        'en': 'Low',
    },
    'risk_high_flood': {
        'id': 'Risiko Banjir Tinggi - Perlu tindakan segera',
        'en': 'High Flood Risk - Immediate action required',
    },
    'risk_high_drought': {
        'id': 'Risiko Kekeringan Tinggi - Perlu tindakan segera',
        'en': 'High Drought Risk - Immediate action required',
    },
    'risk_medium_monitor': {
        'id': 'Risiko Sedang - Perlu pemantauan',
        'en': 'Medium Risk - Monitoring needed',
    },
    'risk_low_normal': {
        'id': 'Risiko Rendah - Kondisi normal',
        'en': 'Low Risk - Normal conditions',
    },

    # ==========================================
    # WATER QUALITY INDEX
    # ==========================================
    'wqi_excellent': {
        'id': 'Sangat Baik (≥90)',
        'en': 'Excellent (≥90)',
    },
    'wqi_good': {
        'id': 'Baik (70-90)',
        'en': 'Good (70-90)',
    },
    'wqi_fair': {
        'id': 'Cukup (50-70)',
        'en': 'Fair (50-70)',
    },
    'wqi_poor': {
        'id': 'Buruk (30-50)',
        'en': 'Poor (30-50)',
    },
    'wqi_very_poor': {
        'id': 'Sangat Buruk (<30)',
        'en': 'Very Poor (<30)',
    },

    # ==========================================
    # ECOSYSTEM STATUS
    # ==========================================
    'ecosystem_very_healthy': {
        'id': 'Sangat Sehat (≥80%)',
        'en': 'Very Healthy (≥80%)',
    },
    'ecosystem_healthy': {
        'id': 'Sehat (60-80%)',
        'en': 'Healthy (60-80%)',
    },
    'ecosystem_fair': {
        'id': 'Cukup Sehat (40-60%)',
        'en': 'Fair Health (40-60%)',
    },
    'ecosystem_poor': {
        'id': 'Kurang Sehat (<40%)',
        'en': 'Poor Health (<40%)',
    },

    # ==========================================
    # WATER BALANCE STATUS
    # ==========================================
    'balance_excellent': {
        'id': 'Sangat Baik (Error <5%)',
        'en': 'Excellent (Error <5%)',
    },
    'balance_good': {
        'id': 'Baik (Error 5-10%)',
        'en': 'Good (Error 5-10%)',
    },
    'balance_fair': {
        'id': 'Cukup (Error 10-20%)',
        'en': 'Fair (Error 10-20%)',
    },
    'balance_poor': {
        'id': 'Buruk (Error >20%)',
        'en': 'Poor (Error >20%)',
    },

    # ==========================================
    # SOIL CONDITIONS
    # ==========================================
    'soil_optimal': {
        'id': 'Optimal (20-40 mm)',
        'en': 'Optimal (20-40 mm)',
    },
    'soil_dry': {
        'id': 'Kering (<20 mm)',
        'en': 'Dry (<20 mm)',
    },
    'soil_saturated': {
        'id': 'Jenuh (>40 mm)',
        'en': 'Saturated (>40 mm)',
    },

    # ==========================================
    # SLOPE CATEGORIES
    # ==========================================
    'slope_gentle': {
        'id': 'Landai (<2°)',
        'en': 'Gentle (<2°)',
    },
    'slope_moderate': {
        'id': 'Sedang (2-8°)',
        'en': 'Moderate (2-8°)',
    },
    'slope_steep': {
        'id': 'Curam (>8°)',
        'en': 'Steep (>8°)',
    },

    # ==========================================
    # SEDIMENT
    # ==========================================
    'sediment_high_dredging': {
        'id': 'Tinggi - Perlu pengerukan',
        'en': 'High - Dredging required',
    },

    # ==========================================
    # TREND LABELS
    # ==========================================
    'trend_stable': {
        'id': 'Stabil',
        'en': 'Stable',
    },
    'trend_changing': {
        'id': 'Berubah',
        'en': 'Changing',
    },
    'trend_rising': {
        'id': 'Meningkat',
        'en': 'Rising',
    },
    'trend_falling': {
        'id': 'Menurun',
        'en': 'Falling',
    },

    # ==========================================
    # FORECAST
    # ==========================================
    'forecast_high_rain': {
        'id': 'Curah hujan tinggi diprediksi. Waspadai potensi banjir dan pastikan sistem drainase berfungsi baik.',
        'en': 'High rainfall predicted. Watch for flood potential and ensure drainage system is functioning well.',
    },
    'forecast_normal': {
        'id': 'Curah hujan normal diprediksi. Kondisi operasional sistem air diperkirakan stabil.',
        'en': 'Normal rainfall predicted. Water system operational conditions are expected to be stable.',
    },
    'forecast_drought_risk': {
        'id': 'Curah hujan rendah diprediksi. Waspadai risiko kekeringan dan kelola cadangan air dengan bijak.',
        'en': 'Low rainfall predicted. Watch for drought risk and manage water reserves wisely.',
    },
    'forecast_insufficient_data': {
        'id': 'Data tidak cukup untuk menghasilkan rekomendasi prakiraan',
        'en': 'Insufficient data to generate forecast recommendation',
    },
    'forecast_data_not_yet_available': {
        'id': 'Data prakiraan belum tersedia',
        'en': 'Forecast data not yet available',
    },

    # ==========================================
    # MANAGEMENT ADVICE
    # ==========================================
    'advice_reservoir_critical': {
        'id': 'Volume kolam retensi kritis (<20%). Kurangi distribusi air segera dan aktifkan sumber cadangan.',
        'en': 'Retention pond volume is critical (<20%). Reduce water distribution immediately and activate backup sources.',
    },
    'advice_reservoir_low': {
        'id': 'Volume kolam retensi rendah (20-50%). Pantau penggunaan air dan bersiap mengaktifkan sumber tambahan.',
        'en': 'Retention pond volume is low (20-50%). Monitor water usage and prepare to activate additional sources.',
    },
    'advice_reservoir_good': {
        'id': 'Volume kolam retensi dalam kondisi baik (>50%). Lanjutkan operasi normal.',
        'en': 'Retention pond volume is in good condition (>50%). Continue normal operations.',
    },
    'advice_reliability_low': {
        'id': 'Keandalan sistem rendah (<70%). Perlu audit infrastruktur dan perbaikan segera.',
        'en': 'System reliability is low (<70%). Infrastructure audit and immediate repairs are needed.',
    },
    'advice_reliability_moderate': {
        'id': 'Keandalan sistem sedang (70-85%). Lakukan pemeliharaan preventif untuk meningkatkan keandalan.',
        'en': 'System reliability is moderate (70-85%). Perform preventive maintenance to improve reliability.',
    },
    'advice_reliability_good': {
        'id': 'Keandalan sistem baik (>85%). Pertahankan jadwal pemeliharaan rutin.',
        'en': 'System reliability is good (>85%). Maintain regular maintenance schedule.',
    },
    'advice_supply_critical': {
        'id': 'Defisit pasokan air kritis. Aktifkan rencana darurat dan batasi penggunaan non-esensial.',
        'en': 'Water supply deficit is critical. Activate emergency plan and restrict non-essential use.',
    },
    'advice_supply_near_limit': {
        'id': 'Pasokan air mendekati batas. Pertimbangkan pembatasan penggunaan air sementara.',
        'en': 'Water supply is near the limit. Consider temporary water use restrictions.',
    },
    'advice_supply_sufficient': {
        'id': 'Pasokan air mencukupi. Operasi distribusi dapat berjalan normal.',
        'en': 'Water supply is sufficient. Distribution operations can run normally.',
    },
    'advice_system_normal': {
        'id': 'Sistem berjalan dalam kondisi normal. Lanjutkan pemantauan rutin.',
        'en': 'System is operating under normal conditions. Continue routine monitoring.',
    },
    'advice_cannot_generate': {
        'id': 'Tidak dapat menghasilkan saran',
        'en': 'Cannot generate advice',
    },

    # ==========================================
    # RECOMMENDATION CATEGORIES & PRIORITIES
    # ==========================================
    'priority_high': {
        'id': 'Prioritas Tinggi',
        'en': 'High Priority',
    },
    'priority_medium': {
        'id': 'Prioritas Sedang',
        'en': 'Medium Priority',
    },
    'priority_normal': {
        'id': 'Prioritas Normal',
        'en': 'Normal Priority',
    },
    'rec_cat_reliability': {
        'id': 'Keandalan Sistem',
        'en': 'System Reliability',
    },
    'rec_improve_reliability': {
        'id': 'Lakukan audit infrastruktur dan perbaiki komponen yang menurunkan keandalan sistem distribusi air.',
        'en': 'Conduct infrastructure audit and repair components that reduce water distribution system reliability.',
    },
    'rec_cat_water_supply': {
        'id': 'Pasokan Air',
        'en': 'Water Supply',
    },
    'rec_water_conservation': {
        'id': 'Implementasikan program konservasi air dan identifikasi sumber air alternatif untuk mengurangi defisit.',
        'en': 'Implement water conservation programs and identify alternative water sources to reduce deficit.',
    },
    'rec_cat_flood_mitigation': {
        'id': 'Mitigasi Banjir',
        'en': 'Flood Mitigation',
    },
    'rec_flood_early_warning': {
        'id': 'Bangun sistem peringatan dini banjir dan tingkatkan kapasitas drainase di zona risiko tinggi.',
        'en': 'Build flood early warning system and increase drainage capacity in high-risk zones.',
    },
    'rec_cat_drought_mitigation': {
        'id': 'Mitigasi Kekeringan',
        'en': 'Drought Mitigation',
    },
    'rec_drought_strategy': {
        'id': 'Kembangkan strategi penyimpanan air dan irigasi efisien untuk menghadapi periode kekeringan.',
        'en': 'Develop water storage strategy and efficient irrigation to face drought periods.',
    },
    'rec_cat_water_quality': {
        'id': 'Kualitas Air',
        'en': 'Water Quality',
    },
    'rec_improve_water_quality': {
        'id': 'Lakukan pengujian kualitas air secara berkala dan implementasikan sistem pengolahan air yang sesuai.',
        'en': 'Conduct periodic water quality testing and implement appropriate water treatment systems.',
    },
    'rec_cat_ecosystem': {
        'id': 'Kesehatan Ekosistem',
        'en': 'Ecosystem Health',
    },
    'rec_ecosystem_restoration': {
        'id': 'Lakukan program restorasi ekosistem riparian dan pertahankan aliran lingkungan minimum.',
        'en': 'Implement riparian ecosystem restoration programs and maintain minimum environmental flow.',
    },
    'rec_cat_routine': {
        'id': 'Pemeliharaan Rutin',
        'en': 'Routine Maintenance',
    },
    'rec_routine_monitoring': {
        'id': 'Pertahankan jadwal pemantauan dan pemeliharaan rutin untuk menjaga performa sistem.',
        'en': 'Maintain regular monitoring and maintenance schedule to keep system performance optimal.',
    },

    # ==========================================
    # IMPROVEMENT RECOMMENDATION CATEGORIES
    # ==========================================
    'rec_category_sedimentation': {
        'id': 'Sedimentasi',
        'en': 'Sedimentation',
    },
    'rec_problem_high_sediment': {
        'id': 'Beban sedimen tinggi',
        'en': 'High sediment load',
    },
    'rec_solution_dredging': {
        'id': 'Pengerukan sedimen secara berkala',
        'en': 'Regular sediment dredging',
    },
    'rec_solution_sediment_trap': {
        'id': 'Instalasi sediment trap di hulu',
        'en': 'Install sediment traps upstream',
    },
    'rec_solution_reforestation': {
        'id': 'Program penghijauan di daerah tangkapan air',
        'en': 'Reforestation program in catchment area',
    },
    'rec_category_soil_conservation': {
        'id': 'Konservasi Tanah',
        'en': 'Soil Conservation',
    },
    'rec_problem_low_soil_moisture': {
        'id': 'Kelembaban tanah rendah',
        'en': 'Low soil moisture',
    },
    'rec_solution_infiltration_wells': {
        'id': 'Pembangunan sumur resapan',
        'en': 'Build infiltration wells',
    },
    'rec_solution_rainwater_harvesting': {
        'id': 'Sistem pemanenan air hujan',
        'en': 'Rainwater harvesting system',
    },
    'rec_solution_mulching': {
        'id': 'Aplikasi mulsa untuk mempertahankan kelembaban',
        'en': 'Apply mulching to retain moisture',
    },
    'rec_category_reservoir_capacity': {
        'id': 'Kapasitas Kolam Retensi',
        'en': 'Retention Pond Capacity',
    },
    'rec_problem_reservoir_critical': {
        'id': 'Volume kolam retensi kritis',
        'en': 'Retention pond volume is critical',
    },
    'rec_solution_evaluate_reservoir': {
        'id': 'Evaluasi kapasitas tampungan dan potensi perluasan',
        'en': 'Evaluate storage capacity and expansion potential',
    },
    'rec_solution_optimize_distribution': {
        'id': 'Optimalkan jadwal distribusi air',
        'en': 'Optimize water distribution schedule',
    },
    'rec_solution_demand_management': {
        'id': 'Implementasi manajemen permintaan air',
        'en': 'Implement water demand management',
    },
    'rec_solution_alternative_sources': {
        'id': 'Identifikasi sumber air alternatif',
        'en': 'Identify alternative water sources',
    },
    'rec_category_flood_mitigation': {
        'id': 'Mitigasi Banjir',
        'en': 'Flood Mitigation',
    },
    'rec_problem_high_flood_risk': {
        'id': 'Risiko banjir tinggi',
        'en': 'High flood risk',
    },
    'rec_solution_drainage_system': {
        'id': 'Peningkatan kapasitas sistem drainase',
        'en': 'Increase drainage system capacity',
    },
    'rec_solution_detention_pond': {
        'id': 'Pembangunan kolam detensi banjir',
        'en': 'Build flood detention ponds',
    },
    'rec_solution_spillway': {
        'id': 'Optimalisasi operasi spillway',
        'en': 'Optimize spillway operations',
    },
    'rec_solution_early_warning': {
        'id': 'Sistem peringatan dini banjir',
        'en': 'Flood early warning system',
    },
    'rec_category_drought_mitigation': {
        'id': 'Mitigasi Kekeringan',
        'en': 'Drought Mitigation',
    },
    'rec_problem_high_drought_risk': {
        'id': 'Risiko kekeringan tinggi',
        'en': 'High drought risk',
    },
    'rec_solution_additional_reservoir': {
        'id': 'Pembangunan tampungan air tambahan',
        'en': 'Build additional water storage',
    },
    'rec_solution_groundwater': {
        'id': 'Pengembangan sumber air tanah',
        'en': 'Develop groundwater sources',
    },
    'rec_solution_efficient_irrigation': {
        'id': 'Implementasi irigasi tetes yang efisien',
        'en': 'Implement efficient drip irrigation',
    },
    'rec_solution_water_conservation': {
        'id': 'Program konservasi dan daur ulang air',
        'en': 'Water conservation and recycling program',
    },
    'rec_category_infrastructure': {
        'id': 'Infrastruktur',
        'en': 'Infrastructure',
    },
    'rec_problem_low_reliability': {
        'id': 'Keandalan sistem rendah',
        'en': 'Low system reliability',
    },
    'rec_solution_infrastructure_audit': {
        'id': 'Audit infrastruktur distribusi air secara menyeluruh',
        'en': 'Comprehensive water distribution infrastructure audit',
    },
    'rec_solution_leak_repair': {
        'id': 'Perbaikan kebocoran pipa dan saluran',
        'en': 'Repair pipe and channel leaks',
    },
    'rec_solution_pump_upgrade': {
        'id': 'Upgrade pompa dan peralatan distribusi',
        'en': 'Upgrade pumps and distribution equipment',
    },
    'rec_solution_scada': {
        'id': 'Implementasi sistem SCADA untuk monitoring real-time',
        'en': 'Implement SCADA system for real-time monitoring',
    },
    'rec_category_routine_maintenance': {
        'id': 'Pemeliharaan Rutin',
        'en': 'Routine Maintenance',
    },
    'rec_system_running_well': {
        'id': 'Sistem berjalan dengan baik, tidak ada masalah kritis yang terdeteksi',
        'en': 'System is running well, no critical issues detected',
    },
    'rec_solution_continue_monitoring': {
        'id': 'Lanjutkan pemantauan berkala sistem hidrologi',
        'en': 'Continue periodic hydrological system monitoring',
    },
    'rec_solution_predictive_maintenance': {
        'id': 'Terapkan pemeliharaan prediktif berbasis data',
        'en': 'Apply data-driven predictive maintenance',
    },
    'rec_solution_update_database': {
        'id': 'Perbarui database historis secara berkala',
        'en': 'Regularly update historical database',
    },
    'rec_solution_staff_training': {
        'id': 'Pelatihan staf dalam pengelolaan sumber daya air',
        'en': 'Staff training in water resource management',
    },
    'timeline_ongoing': {
        'id': 'Berkelanjutan',
        'en': 'Ongoing',
    },

    # ==========================================
    # WATER SOURCES
    # ==========================================
    'source_river': {
        'id': 'Sungai',
        'en': 'River',
    },
    'source_diversion': {
        'id': 'Saluran Pengalihan',
        'en': 'Diversion',
    },
    'source_groundwater': {
        'id': 'Air Tanah',
        'en': 'Groundwater',
    },

    # ==========================================
    # ECONOMIC LABELS
    # ==========================================
    'econ_operational_cost': {
        'id': 'Biaya Operasional',
        'en': 'Operational Cost',
    },
    'econ_maintenance_cost': {
        'id': 'Biaya Pemeliharaan',
        'en': 'Maintenance Cost',
    },
    'econ_energy_cost': {
        'id': 'Biaya Energi',
        'en': 'Energy Cost',
    },
    'econ_benefit_agriculture': {
        'id': 'Manfaat Pertanian',
        'en': 'Agriculture Benefit',
    },
    'econ_benefit_domestic': {
        'id': 'Manfaat Rumah Tangga',
        'en': 'Domestic Benefit',
    },
    'econ_benefit_industry': {
        'id': 'Manfaat Industri',
        'en': 'Industry Benefit',
    },

    # ==========================================
    # HABITAT LABELS
    # ==========================================
    'habitat_fish': {
        'id': 'Habitat Ikan',
        'en': 'Fish Habitat',
    },
    'habitat_vegetation': {
        'id': 'Habitat Vegetasi',
        'en': 'Vegetation Habitat',
    },

    # ==========================================
    # DASHBOARD TITLES
    # ==========================================
    'dashboard_title': {
        'id': 'SISTEM MANAJEMEN AIR TERPADU\nPerencanaan dan Evaluasi Sumber Air',
        'en': 'INTEGRATED WATER MANAGEMENT SYSTEM\nWater Resource Planning and Evaluation',
    },
    'reservoir_status': {
        'id': 'STATUS VOLUME KOLAM RETENSI',
        'en': 'RETENTION POND VOLUME STATUS',
    },
    'volume_actual': {
        'id': 'Volume Aktual',
        'en': 'Actual Volume',
    },
    'ml_forecast': {
        'id': 'Prakiraan ML',
        'en': 'ML Forecast',
    },
    'optimal_level': {
        'id': 'Level Optimal (70%)',
        'en': 'Optimal Level (70%)',
    },
    'minimum_level': {
        'id': 'Level Minimum (30%)',
        'en': 'Minimum Level (30%)',
    },
    'key_indicators': {
        'id': 'INDIKATOR KINERJA',
        'en': 'KEY PERFORMANCE INDICATORS',
    },
    'system_reliability': {
        'id': 'Keandalan Sistem',
        'en': 'System Reliability',
    },
    'current_reservoir': {
        'id': 'Volume Kolam Saat Ini',
        'en': 'Current Reservoir Volume',
    },
    'avg_deficit': {
        'id': 'Defisit Rata-rata',
        'en': 'Average Deficit',
    },
    'supply_demand_balance': {
        'id': 'KESEIMBANGAN PASOKAN DAN PERMINTAAN AIR',
        'en': 'WATER SUPPLY AND DEMAND BALANCE',
    },
    'total_demand_label': {
        'id': 'Total Permintaan',
        'en': 'Total Demand',
    },
    'allocation_dist': {
        'id': 'DISTRIBUSI ALOKASI AIR',
        'en': 'WATER ALLOCATION DISTRIBUTION',
    },
    'rainfall_forecast': {
        'id': 'CURAH HUJAN & PRAKIRAAN',
        'en': 'RAINFALL & FORECAST',
    },
    'historical_label': {
        'id': 'Historis',
        'en': 'Historical',
    },
    'risk_analysis': {
        'id': 'ANALISIS RISIKO',
        'en': 'RISK ANALYSIS',
    },
    'flood_risk_label': {
        'id': 'Risiko Banjir',
        'en': 'Flood Risk',
    },
    'drought_risk_label': {
        'id': 'Risiko Kekeringan',
        'en': 'Drought Risk',
    },
    'risk_level_label': {
        'id': 'Tingkat Risiko (%)',
        'en': 'Risk Level (%)',
    },
    'reservoir_recommendations': {
        'id': 'REKOMENDASI OPERASI KOLAM RETENSI (ML)',
        'en': 'RETENTION POND OPERATION RECOMMENDATIONS (ML)',
    },

    # ==========================================
    # OPERATION ACTIONS
    # ==========================================
    'action_release': {
        'id': 'LEPAS AIR',
        'en': 'RELEASE',
    },
    'action_maintain': {
        'id': 'PERTAHANKAN',
        'en': 'MAINTAIN',
    },
    'action_store': {
        'id': 'SIMPAN AIR',
        'en': 'STORE',
    },

    # ==========================================
    # SIMPLE REPORT
    # ==========================================
    'report_summary': {
        'id': 'RINGKASAN KONDISI SISTEM',
        'en': 'SYSTEM STATUS SUMMARY',
    },
    'water_availability': {
        'id': 'KEADAAN KETERSEDIAAN AIR',
        'en': 'WATER AVAILABILITY STATUS',
    },
    'current_label': {
        'id': 'Saat ini',
        'en': 'Current',
    },
    'forecast_label': {
        'id': 'Perkiraan',
        'en': 'Forecast',
    },
    'status_very_good': {
        'id': 'SANGAT BAIK',
        'en': 'VERY GOOD',
    },
    'status_good_enough': {
        'id': 'CUKUP BAIK',
        'en': 'GOOD ENOUGH',
    },
    'status_needs_attention': {
        'id': 'PERLU PERHATIAN',
        'en': 'NEEDS ATTENTION',
    },
    'reservoir_condition': {
        'id': 'KONDISI TAMPUNGAN AIR',
        'en': 'WATER RESERVOIR CONDITION',
    },
    'volume_label': {
        'id': 'Volume',
        'en': 'Volume',
    },
    'of_capacity': {
        'id': 'dari kapasitas',
        'en': 'of capacity',
    },
    'status_ideal': {
        'id': 'IDEAL',
        'en': 'IDEAL',
    },
    'status_sufficient': {
        'id': 'CUKUP',
        'en': 'SUFFICIENT',
    },
    'status_low_warning': {
        'id': 'RENDAH - WASPADA',
        'en': 'LOW - WARNING',
    },
    'rainfall_section': {
        'id': 'CURAH HUJAN',
        'en': 'RAINFALL',
    },
    'historical_avg': {
        'id': 'Rata-rata historis',
        'en': 'Historical average',
    },
    'forecast_30day': {
        'id': 'Perkiraan 30 hari',
        'en': '30-day forecast',
    },
    'water_needs': {
        'id': 'PEMENUHAN KEBUTUHAN AIR',
        'en': 'WATER NEEDS FULFILLMENT',
    },
    'demand_label': {
        'id': 'Kebutuhan',
        'en': 'Demand',
    },
    'available_label': {
        'id': 'Tersedia',
        'en': 'Available',
    },
    'fulfilled_label': {
        'id': 'Terpenuhi',
        'en': 'Fulfilled',
    },
    'risk_analysis_section': {
        'id': 'ANALISIS RISIKO',
        'en': 'RISK ANALYSIS',
    },
    'flood_probability': {
        'id': 'KEMUNGKINAN BANJIR',
        'en': 'FLOOD PROBABILITY',
    },
    'drought_probability': {
        'id': 'KEMUNGKINAN KEKERINGAN',
        'en': 'DROUGHT PROBABILITY',
    },
    'detected_days': {
        'id': 'Terdeteksi',
        'en': 'Detected',
    },
    'from_label': {
        'id': 'dari',
        'en': 'from',
    },
    'analysis_days': {
        'id': 'hari analisis',
        'en': 'analysis days',
    },
    'status_high_caution': {
        'id': 'TINGGI - PERLU TINDAKAN',
        'en': 'HIGH - ACTION NEEDED',
    },
    'status_low_safe': {
        'id': 'RENDAH - AMAN',
        'en': 'LOW - SAFE',
    },
    'management_advice': {
        'id': 'SARAN PENGELOLAAN',
        'en': 'MANAGEMENT ADVICE',
    },

    # ==========================================
    # SECTOR NAMES
    # ==========================================
    'sector_domestic': {
        'id': 'Rumah Tangga',
        'en': 'Domestic',
    },
    'sector_agriculture': {
        'id': 'Pertanian',
        'en': 'Agriculture',
    },
    'sector_industry': {
        'id': 'Industri',
        'en': 'Industry',
    },
    'sector_environmental': {
        'id': 'Lingkungan',
        'en': 'Environmental',
    },

    # ==========================================
    # GEE RAW DATA
    # ==========================================
    'save_gee_data': {
        'id': 'MENYIMPAN DATA MENTAH GEE',
        'en': 'SAVING RAW GEE DATA',
    },
    'gee_data_saved': {
        'id': 'Data Mentah GEE berhasil disimpan',
        'en': 'Raw GEE Data successfully saved',
    },
    'data_period': {
        'id': 'Periode',
        'en': 'Period',
    },
    'total_data': {
        'id': 'Total Data',
        'en': 'Total Data',
    },
    'data_sources_label': {
        'id': 'Sumber Data',
        'en': 'Data Sources',
    },
    'climatology_stats': {
        'id': 'Statistik Klimatologi',
        'en': 'Climatology Statistics',
    },
    'column_descriptions': {
        'id': 'Kolom Data GEE',
        'en': 'GEE Data Columns',
    },

    # ==========================================
    # RIVER MAP
    # ==========================================
    'creating_river_map': {
        'id': 'MEMBUAT PETA ALIRAN SUNGAI',
        'en': 'CREATING RIVER NETWORK MAP',
    },
    'analysis_location': {
        'id': 'Lokasi Analisis',
        'en': 'Analysis Location',
    },
    'buffer_area': {
        'id': 'Area Buffer',
        'en': 'Buffer Area',
    },
    'fetching_hydro_data': {
        'id': 'Mengambil data jaringan sungai dari Google Earth Engine...',
        'en': 'Fetching river network data from Google Earth Engine...',
    },
    'creating_interactive_map': {
        'id': 'Membuat peta interaktif...',
        'en': 'Creating interactive map...',
    },
    'river_network_title': {
        'id': 'PETA JARINGAN ALIRAN SUNGAI',
        'en': 'RIVER NETWORK MAP',
    },
    'flow_accumulation': {
        'id': 'Akumulasi Aliran',
        'en': 'Flow Accumulation',
    },
    'water_occurrence': {
        'id': 'Kejadian Air (%)',
        'en': 'Water Occurrence (%)',
    },
    'legend_title': {
        'id': 'LEGENDA PETA SUNGAI',
        'en': 'RIVER MAP LEGEND',
    },
    'analysis_point': {
        'id': 'Titik Analisis',
        'en': 'Analysis Point',
    },
    'main_river': {
        'id': 'Aliran Sungai Utama',
        'en': 'Main River Channel',
    },
    'tributary': {
        'id': 'Anak Sungai',
        'en': 'Tributary',
    },
    'flow_intensity': {
        'id': 'Intensitas Warna: Akumulasi Aliran',
        'en': 'Color Intensity: Flow Accumulation',
    },
    'river_characteristics': {
        'id': 'KARAKTERISTIK JARINGAN SUNGAI',
        'en': 'RIVER NETWORK CHARACTERISTICS',
    },
    'avg_accumulation': {
        'id': 'Rata-rata',
        'en': 'Average',
    },
    'max_accumulation': {
        'id': 'Maksimum',
        'en': 'Maximum',
    },
    'map_tip_html': {
        'id': 'Buka file HTML di browser untuk peta interaktif',
        'en': 'Open HTML file in browser for interactive map',
    },

    # ==========================================
    # WATER QUALITY
    # ==========================================
    'water_quality_section': {
        'id': 'KUALITAS AIR',
        'en': 'WATER QUALITY',
    },
    'wqi_score': {
        'id': 'Skor WQI',
        'en': 'WQI Score',
    },
    'quality_params': {
        'id': 'Parameter Kualitas',
        'en': 'Quality Parameters',
    },
    'standard_label': {
        'id': 'Standar',
        'en': 'Standard',
    },
    'compliance_label': {
        'id': 'Kepatuhan Standar',
        'en': 'Standard Compliance',
    },

    # ==========================================
    # ECOLOGY
    # ==========================================
    'ecology_section': {
        'id': 'KONDISI LINGKUNGAN & HABITAT',
        'en': 'ECOLOGY & HABITAT CONDITIONS',
    },
    'habitat_suitability': {
        'id': 'KESESUAIAN HABITAT',
        'en': 'HABITAT SUITABILITY',
    },
    'fish_label': {
        'id': 'Kehidupan Ikan',
        'en': 'Fish Habitat',
    },
    'macroinvert_label': {
        'id': 'Kehidupan Serangga Air',
        'en': 'Macroinvertebrate Habitat',
    },
    'vegetation_label': {
        'id': 'Kondisi Tumbuhan Tepi',
        'en': 'Riparian Vegetation',
    },
    'ecosystem_health': {
        'id': 'KESEHATAN LINGKUNGAN SECARA UMUM',
        'en': 'OVERALL ECOSYSTEM HEALTH',
    },
    'env_flow': {
        'id': 'KEBUTUHAN AIR UNTUK LINGKUNGAN',
        'en': 'ENVIRONMENTAL FLOW REQUIREMENTS',
    },
    'min_demand': {
        'id': 'Kebutuhan Minimal',
        'en': 'Minimum Requirement',
    },
    'deficit_days': {
        'id': 'Hari Defisit Air',
        'en': 'Water Deficit Days',
    },
    'fulfillment_rate': {
        'id': 'Tingkat Pemenuhan',
        'en': 'Fulfillment Rate',
    },
    'flow_alteration': {
        'id': 'PERUBAHAN POLA ALIRAN AIR',
        'en': 'FLOW REGIME ALTERATION',
    },
    'alteration_level': {
        'id': 'Tingkat Perubahan',
        'en': 'Alteration Level',
    },
    'high_alteration_days': {
        'id': 'Hari Perubahan Besar',
        'en': 'High Alteration Days',
    },

    # ==========================================
    # MORPHOLOGY REPORT
    # ==========================================
    'morphology_section': {
        'id': 'ANALISIS KONDISI SUNGAI & TANAH',
        'en': 'RIVER & SOIL CONDITION ANALYSIS',
    },
    'watershed_conditions': {
        'id': 'KONDISI WILAYAH',
        'en': 'WATERSHED CONDITIONS',
    },
    'high_low_distance': {
        'id': 'Jarak High-Low',
        'en': 'High-Low Distance',
    },
    'elevation_range': {
        'id': 'Ketinggian',
        'en': 'Elevation',
    },
    'avg_slope': {
        'id': 'Kemiringan Rata-rata',
        'en': 'Average Slope',
    },
    'sediment_section': {
        'id': 'KONDISI LUMPUR & ENDAPAN',
        'en': 'SEDIMENT & DEPOSITION CONDITIONS',
    },
    'suspended_label': {
        'id': 'Material Tersuspensi',
        'en': 'Suspended Material',
    },
    'bedload_label': {
        'id': 'Material Dasar',
        'en': 'Bedload Material',
    },
    'total_material': {
        'id': 'Total Material',
        'en': 'Total Material',
    },
    'material_transported': {
        'id': 'Persentase Material Terangkut',
        'en': 'Percentage Material Transported',
    },
    'erosion_section': {
        'id': 'KONDISI EROSI TANAH',
        'en': 'SOIL EROSION CONDITIONS',
    },
    'avg_erosion': {
        'id': 'Erosi Rata-rata',
        'en': 'Average Erosion',
    },
    'max_erosion': {
        'id': 'Erosi Tertinggi',
        'en': 'Maximum Erosion',
    },
    'total_erosion': {
        'id': 'Total Tanah Tererosion',
        'en': 'Total Eroded Soil',
    },
    'erosion_level': {
        'id': 'Tingkat Erosi',
        'en': 'Erosion Level',
    },
    'erosion_light': {
        'id': 'RINGAN',
        'en': 'LIGHT',
    },
    'erosion_moderate': {
        'id': 'SEDANG',
        'en': 'MODERATE',
    },
    'erosion_heavy': {
        'id': 'BERAT',
        'en': 'HEAVY',
    },
    'channel_section': {
        'id': 'UKURAN SUNGAI',
        'en': 'CHANNEL DIMENSIONS',
    },
    'avg_width': {
        'id': 'Lebar Rata-rata',
        'en': 'Average Width',
    },
    'avg_depth': {
        'id': 'Kedalaman Rata-rata',
        'en': 'Average Depth',
    },
    'width_change': {
        'id': 'Perubahan Lebar',
        'en': 'Width Change',
    },

    # ==========================================
    # COMPREHENSIVE REPORT SECTIONS
    # ==========================================
    'section_water_rights': {
        'id': 'PEMBAGIAN & PRIORITAS AIR',
        'en': 'WATER ALLOCATION & PRIORITIES',
    },
    'section_network': {
        'id': 'SUMBER-SUMBER AIR',
        'en': 'WATER SOURCES NETWORK',
    },
    'section_cost_benefit': {
        'id': 'BIAYA & MANFAAT',
        'en': 'COST & BENEFIT ANALYSIS',
    },
    'section_water_quality': {
        'id': 'KUALITAS AIR',
        'en': 'WATER QUALITY',
    },
    'economic_analysis': {
        'id': 'ANALISIS EKONOMI',
        'en': 'ECONOMIC ANALYSIS',
    },
    'energy_consumption': {
        'id': 'KONSUMSI ENERGI',
        'en': 'ENERGY CONSUMPTION',
    },
    'cost_breakdown': {
        'id': 'BREAKDOWN BIAYA',
        'en': 'COST BREAKDOWN',
    },
    'management_recommendations': {
        'id': 'SARAN UNTUK PENGELOLAAN AIR',
        'en': 'WATER MANAGEMENT RECOMMENDATIONS',
    },
    'ai_note': {
        'id': 'Laporan ini dihasilkan oleh 9 Model Machine Learning Terintegrasi',
        'en': 'This report was generated by 9 Integrated Machine Learning Models',
    },
    'validation_export': {
        'id': 'EXPORT MODEL VALIDATION METRICS',
        'en': 'EXPORT MODEL VALIDATION METRICS',
    },
    'validation_collected': {
        'id': 'ML Hydro Simulator validation metrics collected',
        'en': 'ML Hydro Simulator validation metrics collected',
    },
    'validation_summary': {
        'id': 'RINGKASAN VALIDASI',
        'en': 'VALIDATION SUMMARY',
    },
    'total_components': {
        'id': 'Total Komponen',
        'en': 'Total Components',
    },

    # ==========================================
    # VISUALIZATION LABELS
    # ==========================================
    'viz_output': {
        'id': 'MEMBUAT VISUALISASI OUTPUT',
        'en': 'CREATING VISUALIZATION OUTPUT',
    },
    'creating_dashboard': {
        'id': 'Membuat RIVANA Dashboard...',
        'en': 'Creating RIVANA Dashboard...',
    },
    'creating_enhanced': {
        'id': 'Membuat Enhanced Dashboard...',
        'en': 'Creating Enhanced Dashboard...',
    },
    'creating_twi_dashboard': {
        'id': 'Membuat TWI Analysis Dashboard...',
        'en': 'Creating TWI Analysis Dashboard...',
    },
    'creating_wb_dashboard': {
        'id': 'Membuat Water Balance Dashboard...',
        'en': 'Creating Water Balance Dashboard...',
    },
    'creating_morph_dashboard': {
        'id': 'Membuat Morphology Ecology Dashboard...',
        'en': 'Creating Morphology Ecology Dashboard...',
    },
    'creating_baseline_dashboard': {
        'id': 'Membuat Baseline Comparison Dashboard...',
        'en': 'Creating Baseline Comparison Dashboard...',
    },
    'dashboard_main_title': {
        'id': 'GAMBARAN MENYELURUH KONDISI AIR\nTampilan Visual Lengkap untuk Pemantauan Air',
        'en': 'COMPREHENSIVE WATER CONDITIONS OVERVIEW\nComplete Visual Display for Water Monitoring',
    },
    'morphology_dashboard_title': {
        'id': 'ANALISIS KONDISI SUNGAI & LINGKUNGAN\nVisualisasi Perubahan dan Status Environmental Perairan',
        'en': 'RIVER & ENVIRONMENTAL CONDITIONS ANALYSIS\nChanges and Aquatic Environmental Status Visualization',
    },
    'wb_dashboard_title': {
        'id': 'WATER BALANCE ANALYSIS MASUK DAN KELUAR\nPerubahan dan Distribusi Air di Wilayah',
        'en': 'WATER BALANCE INFLOW AND OUTFLOW ANALYSIS\nWater Changes and Distribution in the Region',
    },
    'morphometry_title': {
        'id': 'BENTUK DAN UKURAN WILAYAH ALIRAN SUNGAI',
        'en': 'WATERSHED MORPHOMETRY ANALYSIS',
    },

    # ==========================================
    # STATUS MESSAGES
    # ==========================================
    'status_baik': {
        'id': 'BAIK',
        'en': 'GOOD',
    },
    'status_cukup': {
        'id': 'CUKUP',
        'en': 'FAIR',
    },
    'status_kurang': {
        'id': 'KURANG',
        'en': 'POOR',
    },
    'status_optimal': {
        'id': 'OPTIMAL',
        'en': 'OPTIMAL',
    },
    'status_rendah': {
        'id': 'RENDAH',
        'en': 'LOW',
    },
    'status_sangat_baik': {
        'id': 'SANGAT BAIK',
        'en': 'EXCELLENT',
    },
    'status_sangat_buruk': {
        'id': 'BURUK',
        'en': 'POOR',
    },
    'low_flow_natural': {
        'id': 'RENDAH - Aliran alami terjaga',
        'en': 'LOW - Natural flow preserved',
    },
    'moderate_flow_monitor': {
        'id': 'SEDANG - Perlu pemantauan',
        'en': 'MODERATE - Monitoring needed',
    },
    'high_flow_action': {
        'id': 'TINGGI - Perlu penanganan',
        'en': 'HIGH - Action needed',
    },
}


def get_text(key: str, lang: str = None, **kwargs) -> str:
    """
    Get translated text for a given key.

    Args:
        key:    Translation key (see TRANSLATIONS dict above)
        lang:   Language code: 'id' or 'en'. If None, reads RIVANA_LANG env var.
        **kwargs: Optional format args

    Returns:
        Translated string. Falls back to Indonesian if key/lang not found.
    """
    if lang is None:
        lang = os.environ.get('RIVANA_LANG', 'id')

    lang = lang.lower()
    if lang not in ('id', 'en'):
        lang = 'id'

    entry = TRANSLATIONS.get(key)
    if entry is None:
        return f'[{key}]'

    text = entry.get(lang) or entry.get('id', f'[{key}]')

    if kwargs:
        try:
            text = text.format(**kwargs)
        except KeyError:
            pass

    return text


# Short alias — reads language from environment automatically
def T(key: str, **kwargs) -> str:
    """Short alias for get_text(). Reads RIVANA_LANG from environment."""
    return get_text(key, **kwargs)


# ==========================================
# HELPER: translate status strings
# ==========================================
_STATUS_MAP = {
    # Reliability
    'SANGAT BAIK': {'en': 'EXCELLENT'},
    'CUKUP BAIK': {'en': 'GOOD ENOUGH'},
    'PERLU PERHATIAN': {'en': 'NEEDS ATTENTION'},
    # Reservoir
    'OPTIMAL': {'en': 'OPTIMAL'},
    'CUKUP': {'en': 'SUFFICIENT'},
    'RENDAH - WASPADA': {'en': 'LOW - WARNING'},
    # Risk
    'TINGGI - PERLU TINDAKAN': {'en': 'HIGH - ACTION NEEDED'},
    'RENDAH - AMAN': {'en': 'LOW - SAFE'},
    # Erosion
    'RINGAN': {'en': 'LIGHT'},
    'SEDANG': {'en': 'MODERATE'},
    'BERAT': {'en': 'HEAVY'},
    # Actions
    'LEPAS AIR': {'en': 'RELEASE'},
    'PERTAHANKAN': {'en': 'MAINTAIN'},
    'SIMPAN AIR': {'en': 'STORE'},
    # Ecology
    'SANGAT BURUK': {'en': 'VERY POOR'},
    'BURUK': {'en': 'POOR'},
    'BAIK': {'en': 'GOOD'},
    # Flow
    'RENDAH - Aliran alami terjaga': {'en': 'LOW - Natural flow preserved'},
    'SEDANG - Perlu pemantauan': {'en': 'MODERATE - Monitoring needed'},
    'TINGGI - Perlu penanganan': {'en': 'HIGH - Action needed'},
    # TWI Risk Levels
    'HIGH': {'en': 'HIGH'},
    'MODERATE': {'en': 'MODERATE'},
    'LOW': {'en': 'LOW'},
    'VERY HIGH': {'en': 'VERY HIGH'},
}


def translate_status(status_id: str, lang: str = None) -> str:
    """
    Translate a status string from Indonesian to the target language.

    Args:
        status_id: Status string in Indonesian
        lang:      Target language ('id' or 'en'). Reads RIVANA_LANG if None.

    Returns:
        Translated status string (original returned if not found).
    """
    if lang is None:
        lang = os.environ.get('RIVANA_LANG', 'id')

    if lang == 'id':
        return status_id

    entry = _STATUS_MAP.get(status_id)
    if entry:
        return entry.get(lang, status_id)
    return status_id