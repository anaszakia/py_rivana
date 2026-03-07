import http.server
import socketserver
import json
import os
import sys
import time
import threading
import uuid
import urllib.parse
from datetime import datetime
import traceback
import io
from contextlib import redirect_stdout, redirect_stderr

# Import configuration manager
try:
    from config import config
    CONFIG_LOADED = True
except ImportError:
    CONFIG_LOADED = False
    print("⚠️  Warning: config.py not found, using defaults")

# Import translations module
try:
    from translations import get_text, T, translate_status
    TRANSLATIONS_LOADED = True
except ImportError:
    TRANSLATIONS_LOADED = False
    print("⚠️  Warning: translations.py not found, using fallback strings")
    # Fallback: T() just returns the key as-is
    def T(key, **kwargs):
        return key
    def get_text(key, lang=None, **kwargs):
        return key
    def translate_status(status_id, lang=None):
        return status_id

"""
API SERVER HIDROLOGI ML (RIVANA)
================================

Output Files Generated:
-----------------------
PNG Files (7):
  1. RIVANA_Dashboard.png - Main dashboard
  2. RIVANA_Enhanced_Dashboard.png - Enhanced visualization
  3. RIVANA_Water_Balance_Dashboard.png - Water balance analysis
  4. RIVANA_Morphometry_Summary.png - Morphology summary
  5. RIVANA_Morphology_Ecology_Dashboard.png - Morphology & ecology
  6. RIVANA_Baseline_Comparison.png - ML vs Traditional methods
  7. RIVANA_TWI_Dashboard.png - 🌊 NEW: TWI Analysis Dashboard (flood zones & RTH)

HTML Files (1):
  1. RIVANA_Interactive_River_Map.html - 🗺️ NEW: Interactive river map
     (Can be opened in browser, includes zoom, layer toggle, and markers)

CSV Files (4):
  1. RIVANA_Hasil_Complete.csv - Complete simulation results
  2. RIVANA_Monthly_WaterBalance.csv - Monthly water balance
  3. RIVANA_Prediksi_30Hari.csv - 30-day rainfall & reservoir forecast
  4. GEE_Raw_Data.csv - ⭐ Raw satellite data from Google Earth Engine
     (Columns: date, longitude, latitude, elevation_m, slope_degree, rainfall,
      temperature, soil_moisture, ndvi, evapotranspiration)
     Sorted by date for historical analysis

JSON Files (7):
  1. RIVANA_WaterBalance_Validation.json - Water balance validation (error ≤ 5%)
  2. RIVANA_Model_Validation_Complete.json - NSE, R², PBIAS, RMSE metrics
  3. RIVANA_Baseline_Comparison.json - ML vs Traditional comparison results
  4. RIVANA_Model_Validation_Report.json - Detailed validation report
  5. RIVANA_River_Network_Metadata.json - 🌊 River map metadata & characteristics
  6. GEE_Data_Metadata.json - ⭐ GEE data sources & statistics
  7. RIVANA_TWI_Analysis.json - 🌊 NEW: TWI analysis, flood zones, RTH & drainage recommendations

Additional Files:
  - params.json - Input parameters
  - process.log - Complete execution log
"""

# Import pandas untuk summary (optional - akan dicek saat digunakan)
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️  Warning: pandas not available, fitur summary akan terbatas")

# Simpan hasil proses berdasarkan job ID
RESULTS = {}
PROCESSES = {}

def get_results_dir():
    """Get results directory path - use config if available"""
    if CONFIG_LOADED:
        return config.RESULTS_DIR
    return "results"

def get_job_result_path(job_id):
    """Get full path to job result directory"""
    return os.path.join(get_results_dir(), job_id)

def cleanup_old_jobs(max_age_days=30):
    """
    Delete job directories older than max_age_days

    Args:
        max_age_days (int): Maximum age in days before deletion (default: 30)

    Returns:
        tuple: (deleted_count, freed_space_mb)
    """
    results_dir = get_results_dir()
    if not os.path.exists(results_dir):
        return 0, 0

    deleted_count = 0
    freed_space = 0
    current_time = datetime.now()

    print(f"\n{'='*80}")
    print(f"🧹 AUTO-CLEANUP: Checking for jobs older than {max_age_days} days")
    print(f"{'='*80}")

    for job_id in os.listdir(results_dir):
        job_dir = os.path.join(results_dir, job_id)
        if not os.path.isdir(job_dir):
            continue

        try:
            dir_mtime = os.path.getmtime(job_dir)
            dir_datetime = datetime.fromtimestamp(dir_mtime)
            age_days = (current_time - dir_datetime).days

            if age_days > max_age_days:
                dir_size = 0
                for dirpath, dirnames, filenames in os.walk(job_dir):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        if os.path.exists(filepath):
                            dir_size += os.path.getsize(filepath)

                dir_size_mb = dir_size / (1024 * 1024)

                import shutil
                shutil.rmtree(job_dir)

                deleted_count += 1
                freed_space += dir_size_mb

                print(f"  ✅ Deleted job {job_id} (Age: {age_days} days, Size: {dir_size_mb:.2f} MB)")

                if job_id in RESULTS:
                    del RESULTS[job_id]

        except Exception as e:
            print(f"  ⚠️  Error deleting job {job_id}: {e}")

    if deleted_count > 0:
        print(f"\n📊 Cleanup Summary:")
        print(f"  Jobs Deleted: {deleted_count}")
        print(f"  Space Freed: {freed_space:.2f} MB")
    else:
        print(f"  ✅ No jobs older than {max_age_days} days found")

    print(f"{'='*80}\n")

    return deleted_count, freed_space

def load_existing_jobs():
    """Load existing jobs from results directory"""
    results_dir = get_results_dir()
    if not os.path.exists(results_dir):
        print(f"⚠️  Results directory not found: {results_dir}")
        return

    job_count = 0
    for job_id in os.listdir(results_dir):
        job_dir = os.path.join(results_dir, job_id)
        if not os.path.isdir(job_dir):
            continue

        params_file = os.path.join(job_dir, "params.json")
        params = {}
        if os.path.exists(params_file):
            try:
                with open(params_file, 'r') as f:
                    params = json.load(f)
            except Exception as e:
                print(f"⚠️  Failed to load params for job {job_id}: {e}")

        png_files = [f for f in os.listdir(job_dir) if f.endswith('.png')]
        csv_files = [f for f in os.listdir(job_dir) if f.endswith('.csv')]
        json_files = [f for f in os.listdir(job_dir) if f.endswith('.json') and f != 'params.json']
        html_files = [f for f in os.listdir(job_dir) if f.endswith('.html')]

        has_river_map = any('River_Map' in f or 'River_Network' in f for f in png_files + html_files + json_files)
        has_twi = any('TWI' in f for f in png_files + json_files)

        if len(png_files) > 0 or len(csv_files) > 0:
            status = "completed"
            if len(png_files) == 0:
                status = "completed_with_warning"
        else:
            status = "failed"

        RESULTS[job_id] = {
            "job_id": job_id,
            "status": status,
            "params": params,
            "created_at": "N/A",
            "completed_at": "N/A",
            "result_path": job_dir,
            "files_generated": {
                "png": len(png_files),
                "csv": len(csv_files),
                "json": len(json_files),
                "html": len(html_files),
                "png_files": png_files,
                "csv_files": csv_files,
                "json_files": json_files,
                "html_files": html_files
            },
            "river_map": {
                "available": has_river_map,
                "interactive_html": any('RIVANA_Interactive_River_Map.html' in f for f in html_files),
                "static_png": any('RIVANA_River_Network_Map.png' in f for f in png_files),
                "metadata_json": any('RIVANA_River_Network_Metadata.json' in f for f in json_files)
            },
            "twi_analysis": {
                "available": has_twi,
                "dashboard_png": any('RIVANA_TWI_Dashboard.png' in f for f in png_files),
                "analysis_json": any('RIVANA_TWI_Analysis.json' in f for f in json_files)
            },
            "progress": 100
        }
        job_count += 1

    print(f"✅ Loaded {job_count} existing jobs from disk\n")

class HidrologiRequestHandler(http.server.BaseHTTPRequestHandler):
    def _check_auth(self):
        """Check Bearer Token authentication"""
        auth_header = self.headers.get('Authorization', '')

        if auth_header.startswith('Bearer '):
            token = auth_header[7:]
        else:
            token = auth_header

        if CONFIG_LOADED:
            expected_token = config.API_TOKEN
        else:
            expected_token = "rivana_ml_2024_secure_token_change_this"

        if token != expected_token:
            print(f"❌ Auth failed - Received: {token[:20]}... Expected: {expected_token[:20]}...")
            return False

        print(f"✅ Auth success")
        return True

    def _send_auth_error(self):
        """Send 401 Unauthorized response"""
        self._set_response(401)
        self.wfile.write(json.dumps({
            "error": "Unauthorized",
            "message": "Valid Bearer Token required. Use: Authorization: Bearer YOUR_TOKEN"
        }).encode('utf-8'))

    def _set_response(self, status_code=200, content_type='application/json'):
        self.send_response(status_code)
        self.send_header('Content-type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_OPTIONS(self):
        self._set_response()

    def _get_lang(self):
        """Detect language from RIVANA_LANG env var (set by main_weap_ml per job)"""
        return os.environ.get('RIVANA_LANG', 'en')

    def _generate_forecast_recommendation(self, df):
        """Generate forecast recommendation based on rainfall prediction"""
        lang = self._get_lang()
        try:
            if 'forecast_rainfall' not in df.columns or len(df) < 30:
                return get_text('forecast_insufficient_data', lang)

            recent_forecast = df['forecast_rainfall'].iloc[-30:].mean()

            if recent_forecast > 10:
                return get_text('forecast_high_rain', lang)
            elif recent_forecast > 5:
                return get_text('forecast_normal', lang)
            else:
                return get_text('forecast_drought_risk', lang)
        except Exception as e:
            return f"Failed to generate recommendation: {str(e)}"

    def _generate_management_advice(self, df, validation_data):
        """Generate management advice based on current conditions"""
        lang = self._get_lang()
        advice = []

        try:
            kolam_col = 'reservoir' if 'reservoir' in df.columns else None
            if kolam_col:
                avg_reservoir = df[kolam_col].mean()
                if avg_reservoir < 20:
                    advice.append(get_text('advice_reservoir_critical', lang))
                elif avg_reservoir < 50:
                    advice.append(get_text('advice_reservoir_low', lang))
                else:
                    advice.append(get_text('advice_reservoir_good', lang))

            if 'reliability' in df.columns:
                avg_reliability = df['reliability'].mean()
                if avg_reliability < 0.7:
                    advice.append(get_text('advice_reliability_low', lang))
                elif avg_reliability < 0.85:
                    advice.append(get_text('advice_reliability_moderate', lang))
                else:
                    advice.append(get_text('advice_reliability_good', lang))

            demand_cols = ['demand_Domestic', 'demand_Agriculture', 'demand_Industry']
            supply_cols = ['supply_Domestic', 'supply_Agriculture', 'supply_Industry']

            if all(col in df.columns for col in demand_cols + supply_cols):
                total_demand = sum(df[col].sum() for col in demand_cols)
                total_supply = sum(df[col].sum() for col in supply_cols)
                ratio = total_supply / total_demand if total_demand > 0 else 0

                if ratio < 0.9:
                    advice.append(get_text('advice_supply_critical', lang))
                elif ratio < 1.0:
                    advice.append(get_text('advice_supply_near_limit', lang))
                else:
                    advice.append(get_text('advice_supply_sufficient', lang))

            return advice if advice else [get_text('advice_system_normal', lang)]

        except Exception as e:
            return [f"⚠️ {get_text('advice_cannot_generate', lang)}: {str(e)}"]

    def safe_get_value(self, df, column, agg_func='mean', default=None, format_str="{:.2f}"):
        """Safely get value from DataFrame with fallback"""
        if default is None:
            default = get_text('data_not_available', self._get_lang())
        try:
            if column not in df.columns:
                return default

            series = df[column]
            if series.isna().all() or len(series) == 0:
                return default

            if agg_func == 'mean':
                value = series.mean()
            elif agg_func == 'sum':
                value = series.sum()
            elif agg_func == 'max':
                value = series.max()
            elif agg_func == 'min':
                value = series.min()
            elif agg_func == 'last':
                value = series.iloc[-1]
            elif agg_func == 'first':
                value = series.iloc[0]
            else:
                return default

            if pd.isna(value):
                return default

            return format_str.format(value)
        except Exception as e:
            print(f"⚠️ Error in safe_get_value for column {column}: {e}")
            return default

    def generate_summary_text(self, csv_file, monthly_file, validation_file, job_data):
        """Generate summary text from analysis results - COMPREHENSIVE VERSION"""
        lang = self._get_lang()

        if not PANDAS_AVAILABLE:
            return {
                "error": "Pandas library not available",
                "message": "Install pandas with: pip install pandas"
            }

        na_text = get_text('data_not_available', lang)

        summary = {
            "title": get_text('summary_title', lang),
            "job_info": {},
            "input_parameterers": {},
            "analysis_results": {},
            "statistik_data": {},
            "water_balance": {},
            "model_validation": {},
            "baseline_comparison": {},
            "morfologi": {},
            "ekologi": {},
            "kualitas_data": {},
            "rekomendasi": [],
            "supply_demand": {},
            "alokasi_sektor": {},
            "forecast": {},
            "operasi_reservoir": {}
        }

        try:
            # Job Info
            summary["job_info"] = {
                "job_id": job_data.get("job_id", na_text),
                "status": job_data.get("status", na_text),
                "created_at": job_data.get("created_at", na_text),
                "completed_at": job_data.get("completed_at", na_text),
                "files_generated": {
                    "png": job_data.get("files_generated", {}).get("png", 0),
                    "csv": job_data.get("files_generated", {}).get("csv", 0),
                    "json": job_data.get("files_generated", {}).get("json", 0)
                }
            }

            # Input Parameters
            params = job_data.get("params", {})
            summary["input_parameterers"] = {
                "longitude": params.get("longitude", na_text),
                "latitude": params.get("latitude", na_text),
                "start_date": params.get("start", na_text),
                "end_date": params.get("end", na_text),
                "periode_analisis": f"{params.get('start', na_text)} s/d {params.get('end', na_text)}"
            }

            print(f"\n{'='*80}")
            print(f"SUMMARY GENERATION DEBUG:")
            print(f"{'='*80}")
            print(f"CSV File Path: {csv_file}")
            print(f"CSV File Exists: {os.path.exists(csv_file)}")
            print(f"Monthly File Path: {monthly_file}")
            print(f"Monthly File Exists: {os.path.exists(monthly_file)}")
            print(f"Validation File Path: {validation_file}")
            print(f"Validation File Exists: {os.path.exists(validation_file)}")

            if os.path.exists(csv_file):
                print(f"✅ Reading CSV file: {csv_file}")
                df = pd.read_csv(csv_file)
                print(f"✅ CSV loaded successfully: {len(df)} rows, {len(df.columns)} columns")
                print(f"📋 CSV Columns: {list(df.columns)[:20]}")

                summary["statistik_data"] = {
                    "total_hari": len(df) if not df.empty else 0,
                    "curah_rainfall": {
                        "rata_rata": self.safe_get_value(df, 'rainfall', 'mean', format_str="{:.2f} mm/hari"),
                        "maximum": self.safe_get_value(df, 'rainfall', 'max', format_str="{:.2f} mm"),
                        "minimum": self.safe_get_value(df, 'rainfall', 'min', format_str="{:.2f} mm"),
                        "total": self.safe_get_value(df, 'rainfall', 'sum', format_str="{:.2f} mm")
                    },
                    "volume_reservoir": {
                        "rata_rata": self.safe_get_value(df, 'reservoir', 'mean', format_str="{:.2f} mm"),
                        "maximum": self.safe_get_value(df, 'reservoir', 'max', format_str="{:.2f} mm"),
                        "minimum": self.safe_get_value(df, 'reservoir', 'min', format_str="{:.2f} mm"),
                        "akhir_periode": self.safe_get_value(df, 'reservoir', 'last', format_str="{:.2f} mm")
                    },
                    "reliability_sistem": {
                        "rata_rata": self.safe_get_value(df, 'reliability', 'mean', format_str="{:.1f}%", default=na_text) if 'reliability' not in df.columns else f"{df['reliability'].mean() * 100:.1f}%",
                        "status": self.get_reliability_status(df['reliability'].mean() * 100 if 'reliability' in df.columns and not df['reliability'].isna().all() else 0)
                    }
                }

                summary["analysis_results"] = {
                    "supply_air": {
                        "total_supply": f"{df['total_supply'].mean():.2f} mm/hari" if 'total_supply' in df.columns else na_text,
                        "total_demand": f"{df['total_demand'].mean():.2f} mm/hari" if 'total_demand' in df.columns else na_text,
                        "defisit": f"{df['defisit_total'].mean():.2f} mm/hari" if 'defisit_total' in df.columns else na_text,
                        "status_supply": self.get_supply_status(df) if len(df) > 0 else na_text
                    },
                    "risiko": {
                        "banjir": f"{df['flood_risk'].mean() * 100:.1f}%" if 'flood_risk' in df.columns else na_text,
                        "kekeringan": f"{df['drought_risk'].mean() * 100:.1f}%" if 'drought_risk' in df.columns else na_text,
                        "kategori_risiko": self.get_risk_category(df) if len(df) > 0 else na_text
                    }
                }

                if 'WQI' in df.columns:
                    summary["analysis_results"]["water_quality"] = {
                        "WQI_rata_rata": f"{df['WQI'].mean():.1f}/100" if 'WQI' in df.columns else na_text,
                        "status": self.get_water_quality_index_status(df['WQI'].mean() if 'WQI' in df.columns else 0),
                        "pH": f"{df['pH'].mean():.2f}" if 'pH' in df.columns else na_text,
                        "DO": f"{df['DO'].mean():.2f} mg/L" if 'DO' in df.columns else na_text,
                        "TDS": f"{df['TDS'].mean():.2f} mg/L" if 'TDS' in df.columns else na_text
                    }

                if 'ecosystem_health' in df.columns:
                    eco_health_value = df['ecosystem_health'].mean() * 100
                    summary["analysis_results"]["ecosystem_health"] = {
                        "index": f"{eco_health_value:.1f}%",
                        "status": self.get_ecosystem_status(eco_health_value),
                        "habitat_fish": f"{df['fish_HSI'].mean():.2f}" if 'fish_HSI' in df.columns else na_text,
                        "habitat_vegetation": f"{df['vegetation_HSI'].mean():.2f}" if 'vegetation_HSI' in df.columns else na_text
                    }
                else:
                    summary["analysis_results"]["ecosystem_health"] = {
                        "index": na_text,
                        "status": get_text('data_not_available', lang),
                        "habitat_fish": na_text,
                        "habitat_vegetation": na_text
                    }
            else:
                print(f"❌ CSV file not found: {csv_file}")
                summary["error_detail"] = f"CSV file not found: {os.path.basename(csv_file)}"
                summary["statistik_data"]["total_hari"] = f"N/A - {get_text('file_not_available', lang)}"

            # Water Balance
            if os.path.exists(validation_file):
                print(f"✅ Reading validation file: {validation_file}")
                with open(validation_file, 'r') as f:
                    wb_data = json.load(f)
                    print(f"✅ Validation data loaded: {len(wb_data)} keys")
                    summary["water_balance"] = {
                        "total_input": f"{wb_data.get('total_input_mm', 0):.2f} mm",
                        "total_output": f"{wb_data.get('total_output_mm', 0):.2f} mm",
                        "residual": f"{wb_data.get('residual_mm', 0):.2f} mm",
                        "error_persen": f"{wb_data.get('error_percentage', 0):.2f}%",
                        "status": self.get_balance_status(wb_data.get('error_percentage', 0)),
                        "komponen_input": wb_data.get('input_components', {}),
                        "komponen_output": wb_data.get('output_components', {}),
                        "monthly_summary": wb_data.get('monthly_summary', [])
                    }
            else:
                print(f"❌ Validation file not found: {validation_file}")
                summary["water_balance"]["error_detail"] = get_text('validation_file_not_available', lang)

            print(f"{'='*80}\n")

            job_dir = os.path.dirname(csv_file) if csv_file else ""

            # 1. Model Validation JSON
            model_validation_file = os.path.join(job_dir, 'RIVANA_Model_Validation_Complete.json')
            if os.path.exists(model_validation_file):
                try:
                    with open(model_validation_file, 'r') as f:
                        val_data = json.load(f)
                        summary["model_validation"] = {
                            "NSE": f"{val_data.get('NSE', 0):.3f}",
                            "R2": f"{val_data.get('R2', 0):.3f}",
                            "PBIAS": f"{val_data.get('PBIAS', 0):.2f}%",
                            "RMSE": f"{val_data.get('RMSE', 0):.3f}",
                            "MAE": f"{val_data.get('MAE', 0):.3f}",
                            "status": val_data.get('status', na_text),
                            "interpretasi": val_data.get('interpretation', na_text)
                        }
                except Exception as e:
                    print(f"Warning: Could not read model validation file: {e}")

            # 2. Baseline Comparison JSON
            baseline_file = os.path.join(job_dir, 'RIVANA_Baseline_Comparison.json')
            if os.path.exists(baseline_file):
                try:
                    with open(baseline_file, 'r') as f:
                        baseline_data = json.load(f)
                        comp_results = baseline_data.get('comparison_results', {}).get('runoff', {})

                        avg_improvement = comp_results.get('average_improvement')
                        improvement_str = f"{avg_improvement:.1f}%" if avg_improvement is not None else na_text

                        summary["baseline_comparison"] = {
                            "ml_performance": comp_results.get('ML_Model', {}),
                            "traditional_methods": {
                                "Rational": comp_results.get('Rational Method', {}),
                                "Curve Number": comp_results.get('Curve Number', {}),
                                "Simple Balance": comp_results.get('Simple Balance', {}),
                                "Persistence": comp_results.get('Persistence', {}),
                                "Moving Average": comp_results.get('Moving Average', {})
                            },
                            "improvement": improvement_str,
                            "conclusion": baseline_data.get('conclusion', {})
                        }
                        print(f"✅ Baseline comparison data loaded successfully")
                except Exception as e:
                    print(f"⚠️ Warning: Could not read baseline comparison file: {e}")
                    summary["baseline_comparison"] = {"status": "File exists but could not be read"}
            else:
                print(f"ℹ️ Info: Baseline comparison file not found (optional): {baseline_file}")
                summary["baseline_comparison"] = {"status": get_text('not_available_for_job', lang)}

            # 3. TWI Analysis JSON
            twi_file = os.path.join(job_dir, 'RIVANA_TWI_Analysis.json')
            print(f"\n{'='*80}")
            print(f"🔍 DEBUG: Checking TWI Analysis JSON")
            print(f"{'='*80}")
            print(f"Job directory: {job_dir}")
            print(f"Expected file path: {twi_file}")
            print(f"File exists: {os.path.exists(twi_file)}")
            if os.path.exists(job_dir):
                print(f"Job dir exists, listing files:")
                for fname in os.listdir(job_dir):
                    if 'TWI' in fname:
                        print(f"  - {fname}")

            if os.path.exists(twi_file):
                try:
                    print(f"✅ TWI file found, loading data...")
                    file_size = os.path.getsize(twi_file)
                    print(f"File size: {file_size} bytes ({file_size/1024:.2f} KB)")

                    with open(twi_file, 'r', encoding='utf-8') as f:
                        twi_analysis = json.load(f)

                    print(f"✅ TWI JSON loaded successfully")
                    print(f"Keys in TWI data: {list(twi_analysis.keys())}")

                    twi_data = twi_analysis.get('twi_data', {})
                    flood_zones_raw = twi_analysis.get('flood_zones', [])
                    rtho_recs_raw = twi_analysis.get('rtho_recommendations', [])
                    drainage_recs_raw = twi_analysis.get('drainage_recommendations', [])
                    twi_summary = twi_analysis.get('summary', {})

                    print(f"📊 Extracted data:")
                    print(f"   - twi_data keys: {list(twi_data.keys()) if twi_data else 'None'}")
                    print(f"   - flood_zones count: {len(flood_zones_raw)}")
                    print(f"   - rtho_recommendations count: {len(rtho_recs_raw)}")
                    print(f"   - drainage_recommendations count: {len(drainage_recs_raw)}")
                    print(f"   - summary keys: {list(twi_summary.keys()) if twi_summary else 'None'}")

                    flood_zones_mapped = []
                    for zone in flood_zones_raw:
                        coords = zone.get('coordinates', {})
                        flood_zones_mapped.append({
                            'risk': zone.get('risk_level', na_text),
                            'twi_value': zone.get('twi_enhanced', 0),
                            'area_ha': zone.get('area_affected_ha', 0),
                            'lat': coords.get('latitude', 0),
                            'lon': coords.get('longitude', 0)
                        })

                    rtho_recs_mapped = []
                    for rec in rtho_recs_raw:
                        coords = rec.get('coordinates', {})
                        rtho_recs_mapped.append({
                            'priority': rec.get('priority', na_text),
                            'estimated_area_ha': rec.get('area_recommended_ha', 0),
                            'lat': coords.get('latitude', 0),
                            'lon': coords.get('longitude', 0),
                            'reason': rec.get('location_purpose', ' '.join(rec.get('reasons', [])) if isinstance(rec.get('reasons'), list) else 'Strategic location for flood mitigation')
                        })

                    drainage_recs_mapped = []
                    for drain in drainage_recs_raw:
                        coords = drain.get('coordinates', {})
                        specs = drain.get('specifications', {})
                        capacity = drain.get('capacity', {})
                        benefits = drain.get('expected_benefits', {})
                        maintenance = drain.get('maintenance_requirements', {})

                        drainage_recs_mapped.append({
                            'location_id': drain.get('location_id', na_text),
                            'priority': drain.get('priority', na_text),
                            'drainage_type': drain.get('drainage_type', na_text),
                            'necessity_score': drain.get('necessity_score', 0),
                            'coordinates': {
                                'lat': coords.get('latitude', 0),
                                'lon': coords.get('longitude', 0)
                            },
                            'specifications': {
                                'channel_width_m': specs.get('channel_width_m', 0),
                                'channel_depth_m': specs.get('channel_depth_m', 0),
                                'channel_slope_percent': specs.get('channel_slope_percent', 0),
                                'lining_type': specs.get('lining_type', na_text),
                                'length_estimated_m': specs.get('length_estimated_m', 0)
                            },
                            'capacity': {
                                'design_capacity_m3_per_hour': capacity.get('design_capacity_m3_per_hour', 0),
                                'peak_flow_m3_per_second': capacity.get('peak_flow_m3_per_second', 0),
                                'catchment_area_ha': capacity.get('catchment_area_ha', 0)
                            },
                            'expected_benefits': {
                                'flood_reduction_percent': benefits.get('flood_reduction_percent', 0),
                                'ponding_time_reduction_hours': benefits.get('ponding_time_reduction_hours', 0),
                                'affected_area_ha': benefits.get('affected_area_ha', 0)
                            },
                            'maintenance_requirements': {
                                'cleaning_frequency': maintenance.get('cleaning_frequency', na_text),
                                'inspection_frequency': maintenance.get('inspection_frequency', na_text),
                                'estimated_annual_cost_million_idr': maintenance.get('estimated_annual_cost_million_idr', 0)
                            },
                            'reasons': drain.get('reasons', [])
                        })

                    twi_enhanced_val = twi_data.get('twi_enhanced', 0)
                    summary["twi_analysis"] = {
                        "twi_physics": f"{twi_data.get('twi_physics', 0):.2f}",
                        "ml_correction_factor": f"{twi_data.get('correction_factor', 1.0):.2f}x",
                        "twi_enhanced": f"{twi_enhanced_val:.2f}",
                        "risk_level": translate_status(twi_data.get('risk_level', na_text), lang),
                        "flood_zones": {
                            "total": len(flood_zones_mapped),
                            "high_risk": sum(1 for z in flood_zones_mapped if z.get('risk') == 'HIGH'),
                            "moderate_risk": sum(1 for z in flood_zones_mapped if z.get('risk') == 'MODERATE'),
                            "low_risk": sum(1 for z in flood_zones_mapped if z.get('risk') == 'LOW'),
                            "zones_detail": flood_zones_mapped
                        },
                        "rtho_recommendations": {
                            "total": len(rtho_recs_mapped),
                            "high_priority": sum(1 for r in rtho_recs_mapped if r.get('priority') == 'HIGH'),
                            "moderate_priority": sum(1 for r in rtho_recs_mapped if r.get('priority') in ('MODERATE', 'MEDIUM')),
                            "total_area_ha": sum(r.get('estimated_area_ha', 0) for r in rtho_recs_mapped),
                            "recommendations_detail": rtho_recs_mapped
                        },
                        "drainage_recommendations": {
                            "total": len(drainage_recs_mapped),
                            "high_priority": sum(1 for d in drainage_recs_mapped if d.get('priority') == 'HIGH'),
                            "medium_priority": sum(1 for d in drainage_recs_mapped if d.get('priority') == 'MEDIUM'),
                            "total_capacity_m3_per_hour": sum(d['capacity'].get('design_capacity_m3_per_hour', 0) for d in drainage_recs_mapped),
                            "total_length_m": sum(d['specifications'].get('length_estimated_m', 0) for d in drainage_recs_mapped),
                            "recommendations_detail": drainage_recs_mapped
                        },
                        "summary": twi_summary,
                        "interpretation": {
                            "risk": get_text('twi_high_flood_risk', lang) if twi_enhanced_val >= 15 else get_text('twi_good_drainage', lang),
                            "action": get_text('twi_action_mitigate', lang) if len(flood_zones_mapped) > 0 else get_text('twi_action_monitor', lang)
                        }
                    }
                    print(f"✅ TWI analysis data successfully added to summary")
                    print(f"   - TWI Enhanced value: {twi_enhanced_val}")
                    print(f"   - Risk Level: {twi_data.get('risk_level', na_text)}")
                    print(f"   - Flood zones: {len(flood_zones_mapped)}")
                    print(f"   - RTH recommendations: {len(rtho_recs_mapped)}")
                    print(f"   - Drainage recommendations: {len(drainage_recs_mapped)}")
                    print(f"   - summary['twi_analysis'] keys: {list(summary['twi_analysis'].keys())}")
                except Exception as e:
                    print(f"❌ CRITICAL ERROR reading TWI file: {e}")
                    print(f"Error type: {type(e).__name__}")
                    import traceback
                    print("Full traceback:")
                    traceback.print_exc()
                    summary["twi_analysis"] = {
                        "status": "error",
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "file_path": twi_file,
                        "file_exists": os.path.exists(twi_file),
                        "file_size": os.path.getsize(twi_file) if os.path.exists(twi_file) else 0
                    }
            else:
                print(f"⚠️ TWI analysis file not found: {twi_file}")
                twi_dashboard = os.path.join(job_dir, 'RIVANA_TWI_Dashboard.png')
                print(f"TWI Dashboard PNG: {os.path.exists(twi_dashboard)}")

                if os.path.exists(twi_dashboard):
                    print(f"⚠️ ISSUE: TWI Dashboard generated but JSON not found!")
                    summary["twi_analysis"] = {
                        "status": "TWI Dashboard generated but JSON file missing",
                        "error": "Check logs for TWI JSON export errors",
                        "expected_file": "RIVANA_TWI_Analysis.json",
                        "note": "Dashboard PNG exists, indicating TWI ran but JSON save failed"
                    }
                else:
                    print(f"ℹ️ TWI analysis not available for this job")
                    summary["twi_analysis"] = {
                        "status": get_text('not_available_for_job', lang),
                        "note": "TWI analysis was not generated during processing"
                    }

            print(f"{'='*80}\n")

            # 4. River Network Metadata JSON
            river_map_file = os.path.join(job_dir, 'RIVANA_River_Network_Metadata.json')
            if os.path.exists(river_map_file):
                try:
                    with open(river_map_file, 'r') as f:
                        river_data = json.load(f)
                        summary["river_network"] = {
                            "location": river_data.get('location', {}),
                            "flow_characteristics": river_data.get('flow_characteristics', {}),
                            "water_occurrence": river_data.get('water_occurrence_stats', {}),
                            "analysis_buffer_km": river_data.get('analysis_buffer_m', 0) / 1000,
                            "map_files": river_data.get('output_files', {})
                        }
                        print(f"✅ River network data loaded successfully")
                except Exception as e:
                    print(f"⚠️ Warning: Could not read river network file: {e}")
            else:
                print(f"ℹ️ Info: River network file not found (optional): {river_map_file}")

            # 5. Additional data extraction from CSV
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)

                if 'total_supply' in df.columns:
                    summary["supply_demand"] = {
                        "supply_rata_rata": f"{df['total_supply'].mean():.2f} mm/hari",
                        "demand_rata_rata": f"{df['total_demand'].mean():.2f} mm/hari",
                        "ratio": f"{(df['total_supply'].mean() / df['total_demand'].mean() * 100):.1f}%" if df['total_demand'].mean() > 0 else na_text,
                        "defisit_maksimal": f"{df['defisit_total'].max():.2f} mm" if 'defisit_total' in df.columns else na_text,
                        "status": self.get_supply_status(df)
                    }

                sektor_cols = ['supply_Domestic', 'supply_Agriculture', 'supply_Industry', 'supply_Environmental']
                if all(col in df.columns for col in sektor_cols):
                    try:
                        summary["alokasi_sektor"] = {
                            get_text('sector_domestic', lang): {
                                "rata_rata": f"{df['supply_Domestic'].mean():.2f} mm/hari",
                                "total": f"{df['supply_Domestic'].sum():.2f} mm",
                                "pemenuhan": f"{(df['supply_Domestic'].mean() / 0.4 * 100):.1f}%" if df['supply_Domestic'].mean() > 0 else "0%"
                            },
                            get_text('sector_agriculture', lang): {
                                "rata_rata": f"{df['supply_Agriculture'].mean():.2f} mm/hari",
                                "total": f"{df['supply_Agriculture'].sum():.2f} mm",
                                "pemenuhan": f"{(df['supply_Agriculture'].mean() / 0.8 * 100):.1f}%" if df['supply_Agriculture'].mean() > 0 else "0%"
                            },
                            get_text('sector_industry', lang): {
                                "rata_rata": f"{df['supply_Industry'].mean():.2f} mm/hari",
                                "total": f"{df['supply_Industry'].sum():.2f} mm",
                                "pemenuhan": f"{(df['supply_Industry'].mean() / 0.2 * 100):.1f}%" if df['supply_Industry'].mean() > 0 else "0%"
                            },
                            get_text('sector_environmental', lang): {
                                "rata_rata": f"{df['supply_Environmental'].mean():.2f} mm/hari",
                                "total": f"{df['supply_Environmental'].sum():.2f} mm",
                                "pemenuhan": f"{(df['supply_Environmental'].mean() / 0.3 * 100):.1f}%" if df['supply_Environmental'].mean() > 0 else "0%"
                            }
                        }

                        summary["analysis_results"]["water_supply_per_sector"] = {
                            get_text('sector_domestic', lang): {
                                "quota": "0.4 mm/hari",
                                "alokasi": f"{df['supply_Domestic'].mean():.2f} mm/hari" if 'supply_Domestic' in df.columns else na_text,
                                "prioritas": "10 (Highest)",
                                "pemenuhan": f"{(df['supply_Domestic'].mean() / 0.4 * 100):.1f}%" if 'supply_Domestic' in df.columns else na_text
                            },
                            get_text('sector_agriculture', lang): {
                                "quota": "0.8 mm/hari",
                                "alokasi": f"{df['supply_Agriculture'].mean():.2f} mm/hari" if 'supply_Agriculture' in df.columns else na_text,
                                "prioritas": "7",
                                "pemenuhan": f"{(df['supply_Agriculture'].mean() / 0.8 * 100):.1f}%" if 'supply_Agriculture' in df.columns else na_text
                            },
                            get_text('sector_industry', lang): {
                                "quota": "0.2 mm/hari",
                                "alokasi": f"{df['supply_Industry'].mean():.2f} mm/hari" if 'supply_Industry' in df.columns else na_text,
                                "prioritas": "5",
                                "pemenuhan": f"{(df['supply_Industry'].mean() / 0.2 * 100):.1f}%" if 'supply_Industry' in df.columns else na_text
                            },
                            get_text('sector_environmental', lang): {
                                "quota": "0.3 mm/hari",
                                "alokasi": f"{df['supply_Environmental'].mean():.2f} mm/hari" if 'supply_Environmental' in df.columns else na_text,
                                "prioritas": "9",
                                "pemenuhan": f"{(df['supply_Environmental'].mean() / 0.3 * 100):.1f}%" if 'supply_Environmental' in df.columns else na_text
                            }
                        }
                    except Exception as e:
                        print(f"⚠️ Warning: Could not generate sector allocation data: {e}")
                        summary["alokasi_sektor"] = {"error": "Data not available"}
                        summary["analysis_results"]["water_supply_per_sector"] = {"error": "Data not available"}

                if 'total_supply' in df.columns:
                    total_supply = df['total_supply'].mean()
                    summary["analysis_results"]["water_sources"] = {
                        get_text('source_river', lang): {
                            "supply": f"{total_supply * 0.60:.2f} mm/hari",
                            "biaya": "Rp 150/m³",
                            "kontribusi": "60%"
                        },
                        get_text('source_diversion', lang): {
                            "supply": f"{total_supply * 0.25:.2f} mm/hari",
                            "biaya": "Rp 200/m³",
                            "kontribusi": "25%"
                        },
                        get_text('source_groundwater', lang): {
                            "supply": f"{total_supply * 0.15:.2f} mm/hari",
                            "biaya": "Rp 350/m³",
                            "kontribusi": "15%"
                        }
                    }

                if 'total_supply' in df.columns and 'total_demand' in df.columns:
                    try:
                        total_vol = df['total_supply'].sum()
                        biaya_operasi = total_vol * 150
                        biaya_pemeliharaan = biaya_operasi * 0.15
                        biaya_energi = biaya_operasi * 0.20
                        total_biaya = biaya_operasi + biaya_pemeliharaan + biaya_energi

                        manfaat_pertanian = df['supply_Agriculture'].sum() * 500 if 'supply_Agriculture' in df.columns else 0
                        manfaat_domestik = df['supply_Domestic'].sum() * 800 if 'supply_Domestic' in df.columns else 0
                        manfaat_industri = df['supply_Industry'].sum() * 1200 if 'supply_Industry' in df.columns else 0
                        total_manfaat = manfaat_pertanian + manfaat_domestik + manfaat_industri

                        net_benefit = total_manfaat - total_biaya
                        efisiensi = (total_manfaat / total_biaya * 100) if total_biaya > 0 else 0

                        summary["analysis_results"]["economics"] = {
                            "total_biaya": f"Rp {total_biaya:,.0f}",
                            "total_manfaat": f"Rp {total_manfaat:,.0f}",
                            "net_benefit": f"Rp {net_benefit:,.0f}",
                            "efisiensi": f"{efisiensi:.1f}%",
                            "breakdown": {
                                get_text('econ_operational_cost', lang): f"Rp {biaya_operasi:,.0f}",
                                get_text('econ_maintenance_cost', lang): f"Rp {biaya_pemeliharaan:,.0f}",
                                get_text('econ_energy_cost', lang): f"Rp {biaya_energi:,.0f}",
                                get_text('econ_benefit_agriculture', lang): f"Rp {manfaat_pertanian:,.0f}",
                                get_text('econ_benefit_domestic', lang): f"Rp {manfaat_domestik:,.0f}",
                                get_text('econ_benefit_industry', lang): f"Rp {manfaat_industri:,.0f}"
                            }
                        }
                    except Exception as e:
                        print(f"⚠️ Warning: Could not generate economics data: {e}")
                        summary["analysis_results"]["economics"] = {"error": "Economic data not available"}

                summary["prediksi_30_hari"] = {
                    "rainfall": {
                        "rata_rata": f"{df['forecast_rainfall'].mean():.2f} mm/hari" if 'forecast_rainfall' in df.columns else get_text('forecast_data_not_yet_available', lang),
                        "minimum": f"{df['forecast_rainfall'].min():.2f} mm" if 'forecast_rainfall' in df.columns else na_text,
                        "maximum": f"{df['forecast_rainfall'].max():.2f} mm" if 'forecast_rainfall' in df.columns else na_text,
                        "total": f"{df['forecast_rainfall'].sum():.2f} mm" if 'forecast_rainfall' in df.columns else na_text
                    },
                    "reservoir": {
                        "kondisi_saat_ini": f"{df['reservoir'].iloc[-1]:.2f} mm" if len(df) > 0 and 'reservoir' in df.columns else na_text,
                        "prediksi_30_hari": f"{df['forecast_reservoir'].iloc[-1]:.2f} mm" if 'forecast_reservoir' in df.columns and len(df) > 0 else na_text,
                        "persentase_capacity": f"{(df['reservoir'].iloc[-1] / 100.0 * 100):.1f}%" if len(df) > 0 and 'reservoir' in df.columns else na_text
                    },
                    "reliability": {
                        "saat_ini": f"{df['reliability'].mean() * 100:.1f}%" if 'reliability' in df.columns else na_text,
                        "prediksi_30_hari": f"{df['reliability'].iloc[-30:].mean() * 100:.1f}%" if 'reliability' in df.columns and len(df) >= 30 else na_text,
                        "tren": get_text('trend_stable', lang) if 'reliability' in df.columns and len(df) >= 30 and abs(df['reliability'].mean() - df['reliability'].iloc[-30:].mean()) < 0.05 else get_text('trend_changing', lang)
                    },
                    "rekomendasi_forecast": self._generate_forecast_recommendation(df) if len(df) > 0 else get_text('forecast_insufficient_data', lang)
                }

                if 'Retention Pond_action' in df.columns:
                    action_raw = df['Retention Pond_action'].iloc[-1] if len(df) > 0 and 'Retention Pond_action' in df.columns else "MAINTAIN"
                    summary["operasi_reservoir"] = {
                        "volume_saat_ini": f"{df['reservoir'].iloc[-1]:.2f} mm" if len(df) > 0 else na_text,
                        "persentase_capacity": f"{(df['reservoir'].iloc[-1] / 100.0 * 100):.1f}%" if len(df) > 0 else na_text,
                        "rekomendasi_aksi": translate_status(action_raw, lang),
                        "tren_30_hari": get_text('trend_rising', lang) if len(df) >= 30 and df['reservoir'].iloc[-1] > df['reservoir'].iloc[-30] else get_text('trend_falling', lang) if len(df) >= 30 else get_text('trend_stable', lang)
                    }

                morph_cols = ['channel_width', 'slope', 'total_sedimentt']
                if any(col in df.columns for col in morph_cols):
                    slope_value = df['slope'].mean() if 'slope' in df.columns else None
                    slope_str = f"{slope_value:.2f}°" if slope_value is not None else na_text

                    morfologi_data = {
                        "lebar_sungai": f"{df['channel_width'].mean():.2f} m" if 'channel_width' in df.columns else na_text,
                        "kemiringan": slope_str,
                        "beban_sediment": f"{df['total_sedimentt'].mean():.2f} ton/hari" if 'total_sedimentt' in df.columns else na_text,
                        "erosion_rata_rata": f"{df['erosion_rate'].mean():.2f} mm/tahun" if 'erosion_rate' in df.columns else na_text
                    }
                    summary["morfologi"] = morfologi_data
                    summary["analysis_results"]["morfologi"] = morfologi_data

                if 'ecosystem_health' in df.columns:
                    eco_health = df['ecosystem_health'].mean() * 100
                    summary["ekologi"] = {
                        "ecosystem_health": f"{eco_health:.1f}%",
                        "habitat_ikan": f"{df['fish_HSI'].mean():.2f}" if 'fish_HSI' in df.columns else na_text,
                        "habitat_vegetasi": f"{df['vegetation_HSI'].mean():.2f}" if 'vegetation_HSI' in df.columns else na_text,
                        "temperature_air": f"{df['temperature'].mean():.1f}°C" if 'temperature' in df.columns else na_text,
                        "status": self.get_ecosystem_status(eco_health)
                    }

                    if 'kesehatan_ekosistem' not in summary["analysis_results"]:
                        summary["analysis_results"]["ecosystem_health"] = {}

                    summary["analysis_results"]["ecosystem_health"]["habitat"] = {
                        get_text('habitat_fish', lang): {
                            "HSI": f"{df['fish_HSI'].mean():.2f}" if 'fish_HSI' in df.columns else na_text,
                            "status": get_text('status_good', lang) if 'fish_HSI' in df.columns and df['fish_HSI'].mean() > 0.6 else get_text('status_fair', lang)
                        },
                        get_text('habitat_vegetation', lang): {
                            "HSI": f"{df['vegetation_HSI'].mean():.2f}" if 'vegetation_HSI' in df.columns else na_text,
                            "status": get_text('status_good', lang) if 'vegetation_HSI' in df.columns and df['vegetation_HSI'].mean() > 0.6 else get_text('status_fair', lang)
                        }
                    }

                if 'reliability' in df.columns and 'total_supply' in df.columns and 'total_demand' in df.columns:
                    reliability = df['reliability'].mean() * 100
                    supply_demand_ratio = (df['total_supply'].mean() / df['total_demand'].mean() * 100) if df['total_demand'].mean() > 0 else 0
                    flood_risk = df['flood_risk'].mean() * 100 if 'flood_risk' in df.columns else 0
                    drought_risk = df['drought_risk'].mean() * 100 if 'drought_risk' in df.columns else 0

                    summary["analysis_results"]["summary_sistem"] = {
                        "reliability": {
                            "nilai": f"{reliability:.1f}%",
                            "status": self.get_reliability_status(reliability),
                            "warna": "green" if reliability >= 90 else "yellow" if reliability >= 75 else "orange" if reliability >= 60 else "red"
                        },
                        "supply_demand": {
                            "nilai": f"{supply_demand_ratio:.1f}%",
                            "status": get_text('supply_surplus', lang) if supply_demand_ratio >= 110 else get_text('supply_balanced', lang) if supply_demand_ratio >= 90 else get_text('supply_deficit', lang),
                            "warna": "green" if supply_demand_ratio >= 110 else "yellow" if supply_demand_ratio >= 90 else "red"
                        },
                        "risiko_banjir": {
                            "nilai": f"{flood_risk:.1f}%",
                            "status": get_text('risk_high', lang) if flood_risk > 30 else get_text('risk_medium', lang) if flood_risk > 15 else get_text('risk_low', lang),
                            "warna": "red" if flood_risk > 30 else "yellow" if flood_risk > 15 else "green"
                        },
                        "risiko_kekeringan": {
                            "nilai": f"{drought_risk:.1f}%",
                            "status": get_text('risk_high', lang) if drought_risk > 30 else get_text('risk_medium', lang) if drought_risk > 15 else get_text('risk_low', lang),
                            "warna": "red" if drought_risk > 30 else "yellow" if drought_risk > 15 else "green"
                        }
                    }

            # WATER BALANCE ANALYSIS
            if os.path.exists(validation_file):
                try:
                    with open(validation_file, 'r') as f:
                        validation_data = json.load(f)

                    err_pct = validation_data.get('mass_balance_error_pct', 0)
                    summary["analisis_keseimbangan_air"] = {
                        "validasi_massa": {
                            "error_persentase": f"{err_pct:.4f}%",
                            "status": f"✅ {get_text('validation_valid', lang)}" if abs(err_pct) < 1.0 else f"⚠️ {get_text('validation_needs_review', lang)}",
                            "residual": f"{validation_data.get('residual', 0):.2f} mm",
                            "keterangan": get_text('mass_balance_maintained', lang) if abs(err_pct) < 1.0 else get_text('needs_recalibration', lang)
                        },
                        "komponen_input": {
                            "rainfall": f"{validation_data.get('input_rainfall', 0):.2f} mm",
                            "inflow_sungai": f"{validation_data.get('input_inflow', 0):.2f} mm" if 'input_inflow' in validation_data else na_text,
                            "groundwater_recharge": f"{validation_data.get('input_groundwater', 0):.2f} mm" if 'input_groundwater' in validation_data else na_text,
                            "total_input": f"{validation_data.get('total_input', 0):.2f} mm"
                        },
                        "komponen_output": {
                            "evapotranspirasi": f"{validation_data.get('output_ET', 0):.2f} mm",
                            "runoff": f"{validation_data.get('output_runoff', 0):.2f} mm",
                            "total_supply": f"{validation_data.get('output_supply', 0):.2f} mm",
                            "percolation": f"{validation_data.get('output_percolation', 0):.2f} mm" if 'output_percolation' in validation_data else na_text,
                            "total_output": f"{validation_data.get('total_output', 0):.2f} mm"
                        },
                        "perubahan_storage": {
                            "Retention Pond": f"{validation_data.get('storage_Retention Pond', 0):.2f} mm",
                            "soil_moisture": f"{validation_data.get('storage_soil', 0):.2f} mm" if 'storage_soil' in validation_data else na_text,
                            "total_storage_change": f"{validation_data.get('total_storage_change', 0):.2f} mm"
                        },
                        "kesimpulan": validation_data.get('conclusion', get_text('water_balance_complete', lang))
                    }
                except Exception as e:
                    summary["analisis_keseimbangan_air"] = {"error": f"Cannot read validation file: {str(e)}"}
            else:
                summary["analisis_keseimbangan_air"] = {"status": get_text('validation_file_not_available', lang)}

            # RIVER & SOIL CONDITION ANALYSIS
            if os.path.exists(csv_file):
                kondisi_sungai_soil_storage = {}

                if 'channel_width' in df.columns or 'slope_degree' in df.columns or 'total_sedimentt' in df.columns:
                    morfologi_sungai = {}

                    if 'channel_width' in df.columns:
                        morfologi_sungai["lebar_sungai"] = {
                            "rata_rata": f"{df['channel_width'].mean():.2f} m",
                            "min": f"{df['channel_width'].min():.2f} m",
                            "max": f"{df['channel_width'].max():.2f} m",
                            "status": get_text('status_normal', lang) if 10 <= df['channel_width'].mean() <= 50 else get_text('needs_monitoring', lang)
                        }

                    if 'slope_degree' in df.columns:
                        slope_mean = df['slope_degree'].mean()
                        morfologi_sungai["kemiringan"] = {
                            "rata_rata": f"{slope_mean:.2f}°",
                            "kategori": get_text('slope_gentle', lang) if slope_mean < 2 else get_text('slope_moderate', lang) if slope_mean < 8 else get_text('slope_steep', lang)
                        }

                    if 'total_sedimentt' in df.columns:
                        morfologi_sungai["beban_sediment"] = {
                            "rata_rata": f"{df['total_sedimentt'].mean():.2f} ton/hari",
                            "total": f"{df['total_sedimentt'].sum():.2f} ton",
                            "status": get_text('status_normal', lang) if df['total_sedimentt'].mean() < 100 else get_text('sediment_high_dredging', lang)
                        }

                    if morfologi_sungai:
                        kondisi_sungai_soil_storage["morfologi_sungai"] = morfologi_sungai

                if all(col in df.columns for col in ['soil_moisture', 'infiltration', 'percolation']):
                    sm_mean = df['soil_moisture'].mean()
                    kondisi_sungai_soil_storage["kondisi_soil_storage"] = {
                        "soil_moisture": {
                            "rata_rata": f"{sm_mean:.2f} mm",
                            "min": f"{df['soil_moisture'].min():.2f} mm",
                            "max": f"{df['soil_moisture'].max():.2f} mm",
                            "status": get_text('soil_optimal', lang) if 20 <= sm_mean <= 40 else get_text('soil_dry', lang) if sm_mean < 20 else get_text('soil_saturated', lang)
                        },
                        "infiltration": {
                            "rata_rata": f"{df['infiltration'].mean():.2f} mm/hari",
                            "total": f"{df['infiltration'].sum():.2f} mm",
                            "capacity": get_text('status_good', lang) if df['infiltration'].mean() > 2 else get_text('status_low', lang)
                        },
                        "percolation": {
                            "rata_rata": f"{df['percolation'].mean():.2f} mm/hari",
                            "total": f"{df['percolation'].sum():.2f} mm",
                            "ke_groundwater": f"{df['percolation'].sum() * 0.7:.2f} mm (est. 70%)"
                        }
                    }

                if 'erosion_rate' in df.columns and 'sediment_transport' in df.columns:
                    er_mean = df['erosion_rate'].mean()
                    kondisi_sungai_soil_storage["erosion_sediment"] = {
                        "laju_erosion": {
                            "nilai": f"{er_mean:.2f} ton/ha/tahun",
                            "status": get_text('status_low', lang) if er_mean < 10 else get_text('status_medium', lang) if er_mean < 50 else get_text('status_high', lang),
                            "total_tahunan": f"{df['erosion_rate'].sum():.2f} ton/ha"
                        },
                        "transport_sediment": {
                            "capacity": f"{df['sediment_transport'].mean():.2f} kg/s",
                            "efisiensi": f"{(df['sediment_transport'].mean() / df['total_sedimentt'].mean() * 100):.1f}%" if 'total_sedimentt' in df.columns and df['total_sedimentt'].mean() > 0 else na_text
                        }
                    }

                summary["analisis_kondisi_sungai_soil_storage"] = kondisi_sungai_soil_storage
            else:
                summary["analisis_kondisi_sungai_soil_storage"] = {"status": get_text('csv_data_not_available', lang)}

            # MANAGEMENT ADVICE
            validation_data = {}
            if os.path.exists(validation_file):
                try:
                    with open(validation_file, 'r') as f:
                        validation_data = json.load(f)
                except:
                    pass

            summary["saran_pengelolaan"] = self._generate_management_advice(df, validation_data)

            # IMPROVEMENT RECOMMENDATIONS
            saran_perbaikan = []

            if os.path.exists(csv_file):
                if 'total_sedimentt' in df.columns and df['total_sedimentt'].mean() > 100:
                    saran_perbaikan.append({
                        "kategori": get_text('rec_category_sedimentation', lang),
                        "prioritas": get_text('priority_high', lang),
                        "masalah": f"{get_text('rec_problem_high_sediment', lang)} ({df['total_sedimentt'].mean():.2f} kg/s)",
                        "solusi": [
                            get_text('rec_solution_dredging', lang),
                            get_text('rec_solution_sediment_trap', lang),
                            get_text('rec_solution_reforestation', lang)
                        ],
                        "estimasi_biaya": "Rp 500 juta - 2 miliar",
                        "timeline": "3-6 bulan"
                    })

                if 'soil_moisture' in df.columns and df['soil_moisture'].mean() < 20:
                    saran_perbaikan.append({
                        "kategori": get_text('rec_category_soil_conservation', lang),
                        "prioritas": get_text('priority_medium', lang),
                        "masalah": f"{get_text('rec_problem_low_soil_moisture', lang)} ({df['soil_moisture'].mean():.2f} mm)",
                        "solusi": [
                            get_text('rec_solution_infiltration_wells', lang),
                            get_text('rec_solution_rainwater_harvesting', lang),
                            get_text('rec_solution_mulching', lang)
                        ],
                        "estimasi_biaya": "Rp 100-300 juta",
                        "timeline": "2-4 bulan"
                    })

                if 'reservoir' in df.columns and df['reservoir'].mean() < 30:
                    saran_perbaikan.append({
                        "kategori": get_text('rec_category_reservoir_capacity', lang),
                        "prioritas": get_text('priority_high', lang),
                        "masalah": f"{get_text('rec_problem_reservoir_critical', lang)} ({df['reservoir'].mean():.2f} mm)",
                        "solusi": [
                            get_text('rec_solution_evaluate_reservoir', lang),
                            get_text('rec_solution_optimize_distribution', lang),
                            get_text('rec_solution_demand_management', lang),
                            get_text('rec_solution_alternative_sources', lang)
                        ],
                        "estimasi_biaya": "Rp 5-20 miliar",
                        "timeline": "12-24 bulan"
                    })

                if 'flood_risk' in df.columns and df['flood_risk'].mean() > 0.3:
                    saran_perbaikan.append({
                        "kategori": get_text('rec_category_flood_mitigation', lang),
                        "prioritas": get_text('priority_high', lang),
                        "masalah": f"{get_text('rec_problem_high_flood_risk', lang)} ({df['flood_risk'].mean()*100:.1f}%)",
                        "solusi": [
                            get_text('rec_solution_drainage_system', lang),
                            get_text('rec_solution_detention_pond', lang),
                            get_text('rec_solution_spillway', lang),
                            get_text('rec_solution_early_warning', lang)
                        ],
                        "estimasi_biaya": "Rp 1-5 miliar",
                        "timeline": "6-12 bulan"
                    })

                if 'drought_risk' in df.columns and df['drought_risk'].mean() > 0.3:
                    saran_perbaikan.append({
                        "kategori": get_text('rec_category_drought_mitigation', lang),
                        "prioritas": get_text('priority_high', lang),
                        "masalah": f"{get_text('rec_problem_high_drought_risk', lang)} ({df['drought_risk'].mean()*100:.1f}%)",
                        "solusi": [
                            get_text('rec_solution_additional_reservoir', lang),
                            get_text('rec_solution_groundwater', lang),
                            get_text('rec_solution_efficient_irrigation', lang),
                            get_text('rec_solution_water_conservation', lang)
                        ],
                        "estimasi_biaya": "Rp 2-10 miliar",
                        "timeline": "12-18 bulan"
                    })

                if 'reliability' in df.columns and df['reliability'].mean() < 0.7:
                    saran_perbaikan.append({
                        "kategori": get_text('rec_category_infrastructure', lang),
                        "prioritas": get_text('priority_medium', lang),
                        "masalah": f"{get_text('rec_problem_low_reliability', lang)} ({df['reliability'].mean()*100:.1f}%)",
                        "solusi": [
                            get_text('rec_solution_infrastructure_audit', lang),
                            get_text('rec_solution_leak_repair', lang),
                            get_text('rec_solution_pump_upgrade', lang),
                            get_text('rec_solution_scada', lang)
                        ],
                        "estimasi_biaya": "Rp 500 juta - 3 miliar",
                        "timeline": "6-9 bulan"
                    })

            if not saran_perbaikan:
                saran_perbaikan.append({
                    "kategori": get_text('rec_category_routine_maintenance', lang),
                    "prioritas": get_text('priority_normal', lang),
                    "masalah": get_text('rec_system_running_well', lang),
                    "solusi": [
                        get_text('rec_solution_continue_monitoring', lang),
                        get_text('rec_solution_predictive_maintenance', lang),
                        get_text('rec_solution_update_database', lang),
                        get_text('rec_solution_staff_training', lang)
                    ],
                    "estimasi_biaya": "Rp 50-200 juta/tahun",
                    "timeline": get_text('timeline_ongoing', lang)
                })

            summary["saran_perbaikan_kondisi"] = saran_perbaikan

            summary["kualitas_data"] = {
                "kelengkapan_data": "100%" if os.path.exists(csv_file) else get_text('data_incomplete', lang),
                "periode_valid": get_text('yes', lang) if os.path.exists(csv_file) else get_text('no', lang),
                "file_tersedia": {
                    "visualisasi": f"{job_data.get('files_generated', {}).get('png', 0)} PNG files",
                    "data_csv": f"{job_data.get('files_generated', {}).get('csv', 0)} CSV files",
                    "metadata": f"{job_data.get('files_generated', {}).get('json', 0)} JSON files"
                }
            }

            summary["rekomendasi"] = self.generate_recommendations(summary)

        except KeyError as e:
            summary["error"] = f"Missing column in data: {str(e)}"
            summary["error_detail"] = f"Column '{str(e)}' not found in CSV. Data may not be fully generated."
            print(f"❌ KeyError in generate_summary_text: {e}")
            traceback.print_exc()
        except Exception as e:
            summary["error"] = f"Error generating summary: {str(e)}"
            summary["error_detail"] = "An error occurred while creating the summary."
            print(f"❌ Exception in generate_summary_text: {e}")
            traceback.print_exc()

        return summary

    def get_reliability_status(self, reliability):
        lang = self._get_lang()
        if reliability >= 90:
            return get_text('reliability_excellent', lang)
        elif reliability >= 75:
            return get_text('reliability_good', lang)
        elif reliability >= 60:
            return get_text('reliability_fair', lang)
        else:
            return get_text('reliability_poor', lang)

    def get_supply_status(self, df):
        lang = self._get_lang()
        if 'total_supply' in df.columns and 'total_demand' in df.columns:
            ratio = df['total_supply'].mean() / df['total_demand'].mean() if df['total_demand'].mean() > 0 else 0
            if ratio >= 1.1:
                return get_text('supply_surplus_detail', lang)
            elif ratio >= 0.9:
                return get_text('supply_balanced_detail', lang)
            else:
                return get_text('supply_deficit_detail', lang)
        return get_text('data_not_available', lang)

    def get_risk_category(self, df):
        lang = self._get_lang()
        flood_risk = df['flood_risk'].mean() * 100 if 'flood_risk' in df.columns else 0
        drought_risk = df['drought_risk'].mean() * 100 if 'drought_risk' in df.columns else 0

        if flood_risk > 30:
            return get_text('risk_high_flood', lang)
        elif drought_risk > 30:
            return get_text('risk_high_drought', lang)
        elif flood_risk > 15 or drought_risk > 15:
            return get_text('risk_medium_monitor', lang)
        else:
            return get_text('risk_low_normal', lang)

    def get_water_quality_index_status(self, water_quality_index):
        lang = self._get_lang()
        if water_quality_index >= 90:
            return get_text('wqi_excellent', lang)
        elif water_quality_index >= 70:
            return get_text('wqi_good', lang)
        elif water_quality_index >= 50:
            return get_text('wqi_fair', lang)
        elif water_quality_index >= 30:
            return get_text('wqi_poor', lang)
        else:
            return get_text('wqi_very_poor', lang)

    def get_ecosystem_status(self, health):
        lang = self._get_lang()
        if health >= 80:
            return get_text('ecosystem_very_healthy', lang)
        elif health >= 60:
            return get_text('ecosystem_healthy', lang)
        elif health >= 40:
            return get_text('ecosystem_fair', lang)
        else:
            return get_text('ecosystem_poor', lang)

    def get_balance_status(self, error_pct):
        lang = self._get_lang()
        if abs(error_pct) < 5:
            return get_text('balance_excellent', lang)
        elif abs(error_pct) < 10:
            return get_text('balance_good', lang)
        elif abs(error_pct) < 20:
            return get_text('balance_fair', lang)
        else:
            return get_text('balance_poor', lang)

    def get_file_display_order(self, filename):
        """Get display order priority for sorting files"""
        if filename.endswith('.html'):
            if 'Interactive_River_Map' in filename or 'River_Map' in filename:
                return (0, 0, filename)
            else:
                return (0, 1, filename)
        elif filename.endswith('.png'):
            png_priority = {
                'RIVANA_Dashboard.png': 1,
                'RIVANA_Enhanced_Dashboard.png': 2,
                'RIVANA_TWI_Dashboard.png': 3,
                'RIVANA_Water_Balance_Dashboard.png': 4,
                'RIVANA_Morphometry_Summary.png': 5,
                'RIVANA_Morphology_Ecology_Dashboard.png': 6,
                'RIVANA_Baseline_Comparison.png': 7,
                'RIVANA_River_Network_Map.png': 8
            }
            return (1, png_priority.get(filename, 99), filename)
        elif filename.endswith('.csv'):
            csv_priority = {
                'RIVANA_Hasil_Complete.csv': 1,
                'RIVANA_Monthly_WaterBalance.csv': 2,
                'RIVANA_Prediksi_30Hari.csv': 3,
                'GEE_Raw_Data.csv': 4,
                'RIVANA_WaterBalance_Indices.csv': 5
            }
            return (2, csv_priority.get(filename, 99), filename)
        elif filename.endswith('.json'):
            json_priority = {
                'RIVANA_Model_Validation_Complete.json': 1,
                'RIVANA_Baseline_Comparison.json': 2,
                'RIVANA_TWI_Analysis.json': 3,
                'RIVANA_WaterBalance_Validation.json': 4,
                'RIVANA_Model_Validation_Report.json': 5,
                'RIVANA_River_Network_Metadata.json': 6,
                'GEE_Data_Metadata.json': 7
            }
            return (3, json_priority.get(filename, 99), filename)
        else:
            return (99, 99, filename)

    def generate_recommendations(self, summary):
        """Generate recommendations based on analysis results"""
        lang = self._get_lang()
        recommendations = []

        try:
            reliability_text = summary.get("statistik_data", {}).get("reliability_sistem", {}).get("rata_rata", "")
            if reliability_text and isinstance(reliability_text, str):
                if reliability_text not in [get_text('data_not_available', lang), "N/A", ""]:
                    try:
                        reliability_value = reliability_text.replace("%", "").strip()
                        if reliability_value.replace(".", "", 1).isdigit():
                            reliability = float(reliability_value)
                            if reliability < 75:
                                recommendations.append({
                                    "kategori": get_text('rec_cat_reliability', lang),
                                    "prioritas": get_text('priority_high', lang),
                                    "rekomendasi": get_text('rec_improve_reliability', lang)
                                })
                    except (ValueError, AttributeError) as e:
                        print(f"⚠️ Warning: Could not convert reliability '{reliability_text}' to float: {e}")

            supply_status = summary.get("analysis_results", {}).get("supply_air", {}).get("status_supply", "")
            if get_text('supply_deficit', lang).lower() in supply_status.lower() or "Defisit" in supply_status:
                recommendations.append({
                    "kategori": get_text('rec_cat_water_supply', lang),
                    "prioritas": get_text('priority_high', lang),
                    "rekomendasi": get_text('rec_water_conservation', lang)
                })

            risk_category = summary.get("analysis_results", {}).get("risiko", {}).get("kategori_risiko", "")
            if get_text('risk_high', lang).lower() in risk_category.lower() or "High" in risk_category:
                if "flood" in risk_category.lower() or "banjir" in risk_category.lower():
                    recommendations.append({
                        "kategori": get_text('rec_cat_flood_mitigation', lang),
                        "prioritas": get_text('priority_high', lang),
                        "rekomendasi": get_text('rec_flood_early_warning', lang)
                    })
                elif "drought" in risk_category.lower() or "kekeringan" in risk_category.lower():
                    recommendations.append({
                        "kategori": get_text('rec_cat_drought_mitigation', lang),
                        "prioritas": get_text('priority_high', lang),
                        "rekomendasi": get_text('rec_drought_strategy', lang)
                    })

            if "water_quality" in summary.get("analysis_results", {}):
                wqi_status = summary["analysis_results"]["water_quality"].get("status", "")
                poor_keys = [get_text('wqi_poor', lang), get_text('wqi_very_poor', lang), "Bad", "Sangat Bad"]
                if wqi_status in poor_keys:
                    recommendations.append({
                        "kategori": get_text('rec_cat_water_quality', lang),
                        "prioritas": get_text('priority_high', lang),
                        "rekomendasi": get_text('rec_improve_water_quality', lang)
                    })

            if "ecosystem_health" in summary.get("analysis_results", {}):
                eco_status = summary["analysis_results"]["ecosystem_health"].get("status", "")
                if eco_status in [get_text('ecosystem_poor', lang), "Poor Sehat"]:
                    recommendations.append({
                        "kategori": get_text('rec_cat_ecosystem', lang),
                        "prioritas": get_text('priority_medium', lang),
                        "rekomendasi": get_text('rec_ecosystem_restoration', lang)
                    })

            if len(recommendations) == 0:
                recommendations.append({
                    "kategori": get_text('rec_cat_routine', lang),
                    "prioritas": get_text('priority_normal', lang),
                    "rekomendasi": get_text('rec_routine_monitoring', lang)
                })

        except Exception as e:
            recommendations.append({
                "kategori": "System",
                "prioritas": "Info",
                "rekomendasi": f"Error generating recommendations: {str(e)}"
            })

        return recommendations

    def do_GET(self):
        if not self._check_auth():
            self._send_auth_error()
            return

        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path.startswith('/status/'):
            job_id = path.split('/')[-1]
            if job_id in RESULTS:
                self._set_response()
                result_data = dict(RESULTS[job_id])
                result_data["progress"] = result_data.get("progress", 0)
                self.wfile.write(json.dumps(result_data).encode('utf-8'))
            else:
                self._set_response(404)
                self.wfile.write(json.dumps({"error": "Job not found"}).encode('utf-8'))

        elif path.startswith('/logs/'):
            job_id = path.split('/')[-1]
            if job_id in RESULTS:
                result_dir = get_job_result_path(job_id)
                log_file_path = os.path.join(result_dir, "process.log")

                if os.path.exists(log_file_path):
                    try:
                        with open(log_file_path, 'r', encoding='utf-8') as log_file:
                            log_content = log_file.read()

                        self._set_response()
                        self.wfile.write(json.dumps({
                            "success": True,
                            "job_id": job_id,
                            "log_content": log_content,
                            "log_lines": log_content.split('\n'),
                            "status": RESULTS[job_id]["status"]
                        }).encode('utf-8'))
                    except Exception as e:
                        self._set_response(500)
                        self.wfile.write(json.dumps({
                            "success": False,
                            "error": f"Error reading log file: {str(e)}"
                        }).encode('utf-8'))
                else:
                    self._set_response(404)
                    self.wfile.write(json.dumps({
                        "success": False,
                        "error": "Log file not found"
                    }).encode('utf-8'))
            else:
                self._set_response(404)
                self.wfile.write(json.dumps({
                    "success": False,
                    "error": "Job not found"
                }).encode('utf-8'))

        elif path == '/jobs':
            self._set_response()
            jobs_list = []
            for job_id, data in RESULTS.items():
                jobs_list.append({
                    "job_id": job_id,
                    "status": data["status"],
                    "created_at": data.get("created_at", ""),
                    "params": data.get("params", {})
                })
            self.wfile.write(json.dumps({"jobs": jobs_list}).encode('utf-8'))

        elif path == '/storage/info':
            self._set_response()
            results_dir = get_results_dir()

            total_size = 0
            job_count = 0
            old_jobs = []
            current_time = datetime.now()

            if os.path.exists(results_dir):
                for job_id in os.listdir(results_dir):
                    job_dir = os.path.join(results_dir, job_id)
                    if not os.path.isdir(job_dir):
                        continue

                    job_count += 1
                    dir_size = 0

                    for dirpath, dirnames, filenames in os.walk(job_dir):
                        for filename in filenames:
                            filepath = os.path.join(dirpath, filename)
                            if os.path.exists(filepath):
                                dir_size += os.path.getsize(filepath)

                    total_size += dir_size

                    dir_mtime = os.path.getmtime(job_dir)
                    dir_datetime = datetime.fromtimestamp(dir_mtime)
                    age_days = (current_time - dir_datetime).days

                    if age_days > 30:
                        old_jobs.append({
                            "job_id": job_id,
                            "age_days": age_days,
                            "size_mb": dir_size / (1024 * 1024)
                        })

            self.wfile.write(json.dumps({
                "storage": {
                    "total_size_mb": total_size / (1024 * 1024),
                    "total_size_gb": total_size / (1024 * 1024 * 1024),
                    "job_count": job_count,
                    "results_directory": results_dir
                },
                "retention_policy": {
                    "max_age_days": 30,
                    "cleanup_on_startup": True
                },
                "old_jobs": {
                    "count": len(old_jobs),
                    "total_size_mb": sum(j["size_mb"] for j in old_jobs),
                    "jobs": old_jobs
                }
            }).encode('utf-8'))

        elif path == '/storage/cleanup':
            self._set_response()
            try:
                deleted_count, freed_space = cleanup_old_jobs(max_age_days=30)
                self.wfile.write(json.dumps({
                    "success": True,
                    "deleted_jobs": deleted_count,
                    "freed_space_mb": freed_space,
                    "message": f"Successfully deleted {deleted_count} old jobs, freed {freed_space:.2f} MB"
                }).encode('utf-8'))
            except Exception as e:
                self._set_response(500)
                self.wfile.write(json.dumps({
                    "success": False,
                    "error": str(e)
                }).encode('utf-8'))

        elif path.startswith('/result/'):
            job_id = path.split('/')[-1]
            if job_id in RESULTS and RESULTS[job_id]["status"] in ["completed", "completed_with_warning"]:
                result_dir = get_job_result_path(job_id)
                result_files = []
                png_files = []
                csv_files = []
                json_files = []
                other_files = []

                if os.path.exists(result_dir):
                    for file_name in os.listdir(result_dir):
                        file_path = os.path.join(result_dir, file_name)
                        if os.path.isfile(file_path):
                            file_info = {
                                "name": file_name,
                                "path": f"/download/{job_id}/{file_name}",
                                "size": os.path.getsize(file_path)
                            }
                            result_files.append(file_info)

                            if file_name.endswith('.png'):
                                png_files.append(file_info)
                            elif file_name.endswith('.csv'):
                                csv_files.append(file_info)
                            elif file_name.endswith('.json'):
                                json_files.append(file_info)
                            else:
                                other_files.append(file_info)

                self._set_response()
                response_data = {
                    "job_id": job_id,
                    "status": RESULTS[job_id]["status"],
                    "files": result_files,
                    "files_by_type": {
                        "png": png_files,
                        "csv": csv_files,
                        "json": json_files,
                        "other": other_files
                    },
                    "summary": {
                        "total": len(result_files),
                        "png_count": len(png_files),
                        "csv_count": len(csv_files),
                        "json_count": len(json_files)
                    }
                }

                if "warning" in RESULTS[job_id]:
                    response_data["warning"] = RESULTS[job_id]["warning"]

                self.wfile.write(json.dumps(response_data).encode('utf-8'))
            else:
                self._set_response(404)
                self.wfile.write(json.dumps({"error": "Results not found or job not completed"}).encode('utf-8'))

        elif path.startswith('/summary/'):
            job_id = path.split('/')[-1]
            if job_id in RESULTS and RESULTS[job_id]["status"] in ["completed", "completed_with_warning"]:
                try:
                    result_dir = get_job_result_path(job_id)

                    print(f"\n{'='*80}")
                    print(f"ENDPOINT /summary/{job_id} - START")
                    print(f"{'='*80}")
                    print(f"Result directory: {result_dir}")

                    csv_file = os.path.join(result_dir, 'RIVANA_Hasil_Complete.csv')
                    monthly_file = os.path.join(result_dir, 'RIVANA_Monthly_WaterBalance.csv')
                    validation_file = os.path.join(result_dir, 'RIVANA_WaterBalance_Validation.json')

                    print(f"CSV file: {csv_file} (exists: {os.path.exists(csv_file)})")
                    print(f"Monthly file: {monthly_file} (exists: {os.path.exists(monthly_file)})")
                    print(f"Validation file: {validation_file} (exists: {os.path.exists(validation_file)})")

                    summary_text = self.generate_summary_text(csv_file, monthly_file, validation_file, RESULTS[job_id])

                    has_twi = 'twi_analysis' in summary_text
                    print(f"\n{'='*80}")
                    print(f"ENDPOINT /summary/{job_id} - SUMMARY GENERATED")
                    print(f"{'='*80}")
                    print(f"Has TWI Analysis: {has_twi}")
                    if has_twi:
                        twi_keys = list(summary_text['twi_analysis'].keys()) if isinstance(summary_text.get('twi_analysis'), dict) else []
                        print(f"TWI Analysis keys: {twi_keys}")
                    else:
                        print(f"❌ TWI Analysis NOT found in summary")
                        print(f"Summary top-level keys: {list(summary_text.keys())}")
                    print(f"{'='*80}\n")

                    self._set_response()
                    self.wfile.write(json.dumps({
                        "success": True,
                        "job_id": job_id,
                        "summary": summary_text
                    }).encode('utf-8'))
                except Exception as e:
                    self._set_response(500)
                    self.wfile.write(json.dumps({
                        "success": False,
                        "error": f"Error generating summary: {str(e)}"
                    }).encode('utf-8'))
            else:
                self._set_response(404)
                self.wfile.write(json.dumps({
                    "success": False,
                    "error": "Results not found or job not completed"
                }).encode('utf-8'))

        elif path.startswith('/images/'):
            job_id = path.split('/')[-1]
            if job_id in RESULTS and RESULTS[job_id]["status"] in ["completed", "completed_with_warning"]:
                result_dir = get_job_result_path(job_id)
                png_files = []

                if os.path.exists(result_dir):
                    for file_name in os.listdir(result_dir):
                        if file_name.endswith('.png'):
                            file_path = os.path.join(result_dir, file_name)
                            if os.path.isfile(file_path):
                                file_size = os.path.getsize(file_path)
                                if file_size > 0:
                                    png_files.append({
                                        "name": file_name,
                                        "download_url": f"/download/{job_id}/{file_name}",
                                        "preview_url": f"/preview/{job_id}/{file_name}",
                                        "size": file_size,
                                        "size_mb": round(file_size / (1024 * 1024), 2)
                                    })

                self._set_response()
                self.wfile.write(json.dumps({
                    "job_id": job_id,
                    "total_images": len(png_files),
                    "images": png_files
                }).encode('utf-8'))
            else:
                self._set_response(404)
                self.wfile.write(json.dumps({"error": "Results not found or job not completed"}).encode('utf-8'))

        elif path.startswith('/files/'):
            job_id = path.split('/')[-1]
            if job_id in RESULTS and RESULTS[job_id]["status"] in ["completed", "completed_with_warning"]:
                result_dir = get_job_result_path(job_id)
                all_files = []

                if os.path.exists(result_dir):
                    for file_name in os.listdir(result_dir):
                        if file_name in ['process.log', 'error.log', 'error_outer.log', 'params.json']:
                            continue

                        file_path = os.path.join(result_dir, file_name)
                        if os.path.isfile(file_path):
                            file_size = os.path.getsize(file_path)
                            if file_size > 0:
                                file_ext = os.path.splitext(file_name)[1].lower().replace('.', '')

                                file_info = {
                                    "name": file_name,
                                    "type": file_ext,
                                    "download_url": f"/download/{job_id}/{file_name}",
                                    "preview_url": f"/preview/{job_id}/{file_name}",
                                    "size": file_size,
                                    "size_kb": round(file_size / 1024, 2),
                                    "size_mb": round(file_size / (1024 * 1024), 2),
                                    "display_order": self.get_file_display_order(file_name)
                                }
                                all_files.append(file_info)

                all_files.sort(key=lambda x: (x['display_order'], x['name']))

                self._set_response()
                self.wfile.write(json.dumps({
                    "job_id": job_id,
                    "total_files": len(all_files),
                    "files": all_files,
                    "files_by_type": {
                        "png": [f for f in all_files if f['type'] == 'png'],
                        "csv": [f for f in all_files if f['type'] == 'csv'],
                        "json": [f for f in all_files if f['type'] == 'json'],
                        "html": [f for f in all_files if f['type'] == 'html']
                    },
                    "summary": {
                        "png_count": len([f for f in all_files if f['type'] == 'png']),
                        "csv_count": len([f for f in all_files if f['type'] == 'csv']),
                        "json_count": len([f for f in all_files if f['type'] == 'json']),
                        "html_count": len([f for f in all_files if f['type'] == 'html']),
                        "other_count": len([f for f in all_files if f['type'] not in ['png', 'csv', 'json', 'html']])
                    }
                }).encode('utf-8'))
            else:
                self._set_response(404)
                self.wfile.write(json.dumps({"error": "Results not found or job not completed"}).encode('utf-8'))

        # River Network Map endpoint
        elif path.startswith('/river-map/'):
            job_id = path.split('/')[-1]
            if job_id in RESULTS and RESULTS[job_id]["status"] in ["completed", "completed_with_warning"]:
                result_dir = get_job_result_path(job_id)

                river_map_data = {
                    "job_id": job_id,
                    "available": False,
                    "files": {},
                    "metadata": None
                }

                html_map = os.path.join(result_dir, 'RIVANA_Interactive_River_Map.html')
                if os.path.exists(html_map) and os.path.getsize(html_map) > 0:
                    river_map_data["available"] = True
                    river_map_data["files"]["interactive_html"] = {
                        "name": "RIVANA_Interactive_River_Map.html",
                        "type": "html",
                        "size": os.path.getsize(html_map),
                        "size_kb": round(os.path.getsize(html_map) / 1024, 2),
                        "download_url": f"/download/{job_id}/RIVANA_Interactive_River_Map.html",
                        "preview_url": f"/preview/{job_id}/RIVANA_Interactive_River_Map.html",
                        "description": "Interactive map with zoom, layers, and markers"
                    }

                png_map = os.path.join(result_dir, 'RIVANA_River_Network_Map.png')
                if os.path.exists(png_map) and os.path.getsize(png_map) > 0:
                    river_map_data["available"] = True
                    river_map_data["files"]["static_png"] = {
                        "name": "RIVANA_River_Network_Map.png",
                        "type": "png",
                        "size": os.path.getsize(png_map),
                        "size_kb": round(os.path.getsize(png_map) / 1024, 2),
                        "download_url": f"/download/{job_id}/RIVANA_River_Network_Map.png",
                        "preview_url": f"/preview/{job_id}/RIVANA_River_Network_Map.png",
                        "description": "Static map for presentations and reports"
                    }

                metadata_file = os.path.join(result_dir, 'RIVANA_River_Network_Metadata.json')
                if os.path.exists(metadata_file) and os.path.getsize(metadata_file) > 0:
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            river_map_data["metadata"] = json.load(f)

                        river_map_data["files"]["metadata_json"] = {
                            "name": "RIVANA_River_Network_Metadata.json",
                            "type": "json",
                            "size": os.path.getsize(metadata_file),
                            "size_kb": round(os.path.getsize(metadata_file) / 1024, 2),
                            "download_url": f"/download/{job_id}/RIVANA_River_Network_Metadata.json",
                            "preview_url": f"/preview/{job_id}/RIVANA_River_Network_Metadata.json",
                            "description": "River characteristics and statistics"
                        }
                    except Exception as e:
                        print(f"Error loading river map metadata: {e}")

                if river_map_data["available"]:
                    river_map_data["summary"] = {
                        "status": "Available",
                        "files_count": len(river_map_data["files"]),
                        "has_interactive": "interactive_html" in river_map_data["files"],
                        "has_static": "static_png" in river_map_data["files"],
                        "has_metadata": "metadata_json" in river_map_data["files"]
                    }

                    if river_map_data["metadata"]:
                        river_map_data["quick_info"] = {
                            "location": river_map_data["metadata"].get("location", {}),
                            "flow_accumulation_mean": river_map_data["metadata"].get("flow_characteristics", {}).get("mean_accumulation", "N/A"),
                            "water_occurrence_mean": river_map_data["metadata"].get("water_occurrence", {}).get("mean_percentage", "N/A"),
                            "data_sources": list(river_map_data["metadata"].get("data_sources", {}).keys())
                        }
                else:
                    river_map_data["summary"] = {
                        "status": "Not Available",
                        "message": "River map was not generated for this job"
                    }

                self._set_response()
                self.wfile.write(json.dumps(river_map_data).encode('utf-8'))
            else:
                self._set_response(404)
                self.wfile.write(json.dumps({
                    "error": "Job not found or not completed",
                    "job_id": job_id
                }).encode('utf-8'))

        elif path.startswith('/preview/'):
            parts = path.split('/')
            if len(parts) >= 4:
                job_id = parts[2]
                file_name = urllib.parse.unquote(parts[3])
                file_path = os.path.join(get_job_result_path(job_id), file_name)

                if os.path.exists(file_path) and os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)

                    if file_size == 0:
                        self._set_response(500)
                        self.wfile.write(json.dumps({
                            "error": "File is empty or corrupt",
                            "file_name": file_name
                        }).encode('utf-8'))
                        return

                    content_type = 'application/octet-stream'
                    cache_duration = 3600

                    if file_name.endswith('.png'):
                        content_type = 'image/png'
                        cache_duration = 86400
                    elif file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
                        content_type = 'image/jpeg'
                        cache_duration = 86400
                    elif file_name.endswith('.gif'):
                        content_type = 'image/gif'
                        cache_duration = 86400
                    elif file_name.endswith('.svg'):
                        content_type = 'image/svg+xml'
                        cache_duration = 86400
                    elif file_name.endswith('.csv'):
                        content_type = 'text/csv; charset=utf-8'
                    elif file_name.endswith('.json'):
                        content_type = 'application/json; charset=utf-8'
                    elif file_name.endswith('.txt') or file_name.endswith('.log'):
                        content_type = 'text/plain; charset=utf-8'
                    elif file_name.endswith('.pdf'):
                        content_type = 'application/pdf'
                        cache_duration = 86400
                    elif file_name.endswith('.html'):
                        content_type = 'text/html; charset=utf-8'
                    elif file_name.endswith('.xml'):
                        content_type = 'application/xml; charset=utf-8'

                    self.send_response(200)
                    self.send_header('Content-type', content_type)
                    self.send_header('Content-Length', str(file_size))
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('Cache-Control', f'public, max-age={cache_duration}')
                    self.end_headers()

                    try:
                        with open(file_path, 'rb') as file:
                            chunk_size = 8192
                            while True:
                                chunk = file.read(chunk_size)
                                if not chunk:
                                    break
                                self.wfile.write(chunk)
                    except Exception as e:
                        print(f"Error sending file {file_name}: {str(e)}")
                else:
                    self._set_response(404)
                    self.wfile.write(json.dumps({
                        "error": "File not found",
                        "file_name": file_name,
                        "file_path": file_path
                    }).encode('utf-8'))
            else:
                self._set_response(400)
                self.wfile.write(json.dumps({"error": "Invalid URL"}).encode('utf-8'))

        elif path.startswith('/download/'):
            parts = path.split('/')
            if len(parts) >= 4:
                job_id = parts[2]
                file_name = urllib.parse.unquote(parts[3])
                file_path = os.path.join(get_job_result_path(job_id), file_name)

                if os.path.exists(file_path) and os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)

                    if file_name.endswith('.png') and file_size == 0:
                        self._set_response(500)
                        self.wfile.write(json.dumps({
                            "error": "PNG file is empty or corrupt",
                            "file_name": file_name,
                            "file_size": 0
                        }).encode('utf-8'))
                        return

                    content_type = 'application/octet-stream'
                    if file_name.endswith('.csv'):
                        content_type = 'text/csv'
                    elif file_name.endswith('.json'):
                        content_type = 'application/json'
                    elif file_name.endswith('.png'):
                        content_type = 'image/png'
                    elif file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
                        content_type = 'image/jpeg'
                    elif file_name.endswith('.html'):
                        content_type = 'text/html'

                    self.send_response(200)
                    self.send_header('Content-type', content_type)
                    self.send_header('Content-Length', str(file_size))
                    self.send_header('Content-Disposition', f'attachment; filename="{file_name}"')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()

                    try:
                        with open(file_path, 'rb') as file:
                            chunk_size = 8192
                            while True:
                                chunk = file.read(chunk_size)
                                if not chunk:
                                    break
                                self.wfile.write(chunk)
                    except Exception as e:
                        print(f"Error sending file {file_name}: {str(e)}")
                else:
                    self._set_response(404)
                    self.wfile.write(json.dumps({
                        "error": "File not found",
                        "file_path": file_path,
                        "exists": os.path.exists(file_path)
                    }).encode('utf-8'))
            else:
                self._set_response(400)
                self.wfile.write(json.dumps({"error": "Invalid URL"}).encode('utf-8'))

        else:
            self._set_response(404)
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode('utf-8'))

    def do_POST(self):
        if not self._check_auth():
            self._send_auth_error()
            return

        if self.path == '/generate':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                params = json.loads(post_data.decode('utf-8'))
                required_params = ['longitude', 'latitude', 'start', 'end']
                missing_params = [param for param in required_params if param not in params]
                if missing_params:
                    self._set_response(400)
                    self.wfile.write(json.dumps({
                        "error": f"Missing parameters: {', '.join(missing_params)}"
                    }).encode('utf-8'))
                    return

                job_id = str(uuid.uuid4())
                RESULTS[job_id] = {
                    "job_id": job_id,
                    "status": "processing",
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "params": params
                }

                result_dir = get_job_result_path(job_id)
                os.makedirs(result_dir, exist_ok=True)

                thread = threading.Thread(
                    target=run_hidrologi_process,
                    args=(job_id, params, result_dir)
                )
                thread.daemon = True
                thread.start()
                PROCESSES[job_id] = thread

                self._set_response()
                self.wfile.write(json.dumps({
                    "job_id": job_id,
                    "status": "processing",
                    "message": "Hydrological calculation process started"
                }).encode('utf-8'))
            except json.JSONDecodeError:
                self._set_response(400)
                self.wfile.write(json.dumps({"error": "Invalid JSON format"}).encode('utf-8'))
        else:
            self._set_response(404)
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode('utf-8'))

def run_hidrologi_process(job_id, params, result_dir):
    """Run hydrological calculation process"""
    try:
        RESULTS[job_id]["progress"] = 0

        with open(os.path.join(result_dir, "params.json"), "w") as f:
            json.dump(params, f)

        RESULTS[job_id]["progress"] = 10

        longitude = params['longitude']
        latitude = params['latitude']
        start = params['start']
        end = params['end']

        RESULTS[job_id]["progress"] = 20

        log_file_path = os.path.join(result_dir, "process.log")
        with open(log_file_path, 'w', encoding='utf-8', buffering=1) as log_file:
            original_stdout = sys.stdout
            original_stderr = sys.stderr

            class TeeWriter:
                def __init__(self, file_handle, original_stream):
                    self.file = file_handle
                    self.original_stream = original_stream

                def write(self, data):
                    try:
                        self.file.write(data)
                        self.file.flush()
                    except Exception as e:
                        if self.original_stream:
                            self.original_stream.write(f"[LOG ERROR: {str(e)}]\n")

                def flush(self):
                    try:
                        self.file.flush()
                    except:
                        pass

                def close(self):
                    try:
                        self.flush()
                    except:
                        pass

            sys.stdout = TeeWriter(log_file, original_stdout)
            sys.stderr = TeeWriter(log_file, original_stderr)

            try:
                RESULTS[job_id]["progress"] = 30
                RESULTS[job_id]["status"] = "processing"
                RESULTS[job_id]["message"] = "Initializing ML engine..."

                print(f"{'='*80}")
                print(f"Starting WEAP-ML Analysis")
                print(f"Job ID: {job_id}")
                print(f"Output Directory: {result_dir}")
                print(f"Parameters: lon={longitude}, lat={latitude}, start={start}, end={end}")
                print(f"{'='*80}\n")

                print("Loading main_weap_ml module...")
                sys.stdout.flush()

                try:
                    import main_weap_ml
                    import importlib
                    importlib.reload(main_weap_ml)
                    print("✓ Module loaded successfully (with force reload)!\n")
                    sys.stdout.flush()
                except Exception as import_error:
                    print(f"✗ ERROR loading main_weap_ml: {str(import_error)}")
                    sys.stdout.flush()
                    raise

                RESULTS[job_id]["progress"] = 35
                RESULTS[job_id]["message"] = "Fetching satellite data from Google Earth Engine..."
                print("\n" + "="*80)
                print("⚠️  CRITICAL SECTION: Google Earth Engine Data Fetching")
                print("   This may take 2-5 minutes depending on date range")
                print("   Progress will update after data is fetched")
                print("="*80 + "\n")
                sys.stdout.flush()

                try:
                    print("Starting main_weap_ml.main()...")
                    sys.stdout.flush()

                    main_weap_ml.main(
                        lon=longitude,
                        lat=latitude,
                        start=start,
                        end=end,
                        output_dir=result_dir,
                        lang='en'  # Force English for API calls
                    )

                except Exception as main_error:
                    print(f"\n✗ ERROR in main_weap_ml.main(): {str(main_error)}")
                    print(f"Error type: {type(main_error).__name__}")
                    print("Traceback:")
                    traceback.print_exc()
                    sys.stdout.flush()
                    raise

                print(f"\n{'='*80}")
                print(f"main_weap_ml.main() completed successfully")
                print(f"{'='*80}")

                sys.stdout.flush()
                sys.stderr.flush()

                RESULTS[job_id]["progress"] = 85

                print("Waiting for file write completion...")
                sys.stdout.flush()
                time.sleep(2)

                print(f"Checking files in directory: {result_dir}")
                print(f"Directory exists: {os.path.exists(result_dir)}")

                RESULTS[job_id]["progress"] = 90

                expected_files = {
                    'png': [
                        'RIVANA_Dashboard.png',
                        'RIVANA_Enhanced_Dashboard.png',
                        'RIVANA_Water_Balance_Dashboard.png',
                        'RIVANA_Morphometry_Summary.png',
                        'RIVANA_Morphology_Ecology_Dashboard.png',
                        'RIVANA_Baseline_Comparison.png'
                    ],
                    'csv': [
                        'RIVANA_Hasil_Complete.csv',
                        'RIVANA_Monthly_WaterBalance.csv',
                        'RIVANA_Prediksi_30Hari.csv',
                        'GEE_Raw_Data.csv'
                    ],
                    'json': [
                        'RIVANA_WaterBalance_Validation.json',
                        'RIVANA_Model_Validation_Complete.json',
                        'RIVANA_Baseline_Comparison.json',
                        'RIVANA_Model_Validation_Report.json',
                        'GEE_Data_Metadata.json'
                    ]
                }

                all_files = os.listdir(result_dir) if os.path.exists(result_dir) else []
                print(f"All files in directory: {all_files}")

                png_files = []
                csv_files = []
                json_files = []

                for file_name in all_files:
                    file_path = os.path.join(result_dir, file_name)
                    if os.path.isfile(file_path):
                        file_size = os.path.getsize(file_path)

                        if file_name.endswith('.png'):
                            if file_size > 0:
                                png_files.append(file_name)
                                print(f"  ✅ PNG: {file_name} ({file_size:,} bytes)")
                            else:
                                print(f"  ⚠️  WARNING: {file_name} is empty (0 bytes)")
                        elif file_name.endswith('.csv'):
                            if file_size > 0:
                                csv_files.append(file_name)
                                print(f"  ✅ CSV: {file_name} ({file_size:,} bytes)")
                            else:
                                print(f"  ⚠️  WARNING: {file_name} is empty (0 bytes)")
                        elif file_name.endswith('.json'):
                            if file_size > 0:
                                json_files.append(file_name)
                                print(f"  ✅ JSON: {file_name} ({file_size:,} bytes)")
                            else:
                                print(f"  ⚠️  WARNING: {file_name} is empty (0 bytes)")

                print(f"\n{'='*80}")
                print("FILE COMPLETENESS CHECK:")
                print(f"{'='*80}")

                missing_png = set(expected_files['png']) - set(png_files)
                missing_csv = set(expected_files['csv']) - set(csv_files)
                missing_json = set(expected_files['json']) - set(json_files)

                if missing_png:
                    print(f"⚠️  Missing PNG files: {', '.join(missing_png)}")
                if missing_csv:
                    print(f"⚠️  Missing CSV files: {', '.join(missing_csv)}")
                if missing_json:
                    print(f"⚠️  Missing JSON files: {', '.join(missing_json)}")

                print(f"\n{'='*80}")
                print(f"Summary - Files generated:")
                print(f"  PNG: {len(png_files)}/{len(expected_files['png'])} files")
                print(f"  CSV: {len(csv_files)}/{len(expected_files['csv'])} files")
                print(f"  JSON: {len(json_files)}/{len(expected_files['json'])} files")
                print(f"{'='*80}")

                RESULTS[job_id]["progress"] = 95

                missing_info = []
                if missing_png:
                    missing_info.append(f"PNG: {', '.join(missing_png)}")
                if missing_csv:
                    missing_info.append(f"CSV: {', '.join(missing_csv)}")
                if missing_json:
                    missing_info.append(f"JSON: {', '.join(missing_json)}")

                if not png_files and not csv_files:
                    RESULTS[job_id]["status"] = "failed"
                    RESULTS[job_id]["error"] = "No output files generated"
                    RESULTS[job_id]["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    RESULTS[job_id]["progress"] = 100
                    print("❌ ERROR: No files generated")
                elif not png_files or missing_info:
                    RESULTS[job_id]["status"] = "completed_with_warning"
                    RESULTS[job_id]["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    RESULTS[job_id]["result_path"] = result_dir

                    warning_msg = "Some files were not generated. "
                    if not png_files:
                        warning_msg += "No PNG files found. "
                    if missing_info:
                        warning_msg += f"Missing files: {'; '.join(missing_info)}"

                    RESULTS[job_id]["warning"] = warning_msg
                    RESULTS[job_id]["missing_files"] = {
                        "png": list(missing_png),
                        "csv": list(missing_csv),
                        "json": list(missing_json)
                    }
                    RESULTS[job_id]["files_generated"] = {
                        "png": len(png_files),
                        "csv": len(csv_files),
                        "json": len(json_files),
                        "png_files": png_files,
                        "csv_files": csv_files,
                        "json_files": json_files,
                        "expected_png": len(expected_files['png']),
                        "expected_csv": len(expected_files['csv']),
                        "expected_json": len(expected_files['json'])
                    }
                    RESULTS[job_id]["progress"] = 100
                    print(f"⚠️  WARNING: {warning_msg}")
                else:
                    RESULTS[job_id]["status"] = "completed"
                    RESULTS[job_id]["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    RESULTS[job_id]["result_path"] = result_dir
                    RESULTS[job_id]["files_generated"] = {
                        "png": len(png_files),
                        "csv": len(csv_files),
                        "json": len(json_files),
                        "png_files": png_files,
                        "csv_files": csv_files,
                        "json_files": json_files,
                        "expected_png": len(expected_files['png']),
                        "expected_csv": len(expected_files['csv']),
                        "expected_json": len(expected_files['json'])
                    }
                    RESULTS[job_id]["progress"] = 100
                    print(f"✅ SUCCESS: All expected files generated!")
                    print(f"   PNG: {len(png_files)}/{len(expected_files['png'])}")
                    print(f"   CSV: {len(csv_files)}/{len(expected_files['csv'])}")
                    print(f"   JSON: {len(json_files)}/{len(expected_files['json'])}")

            except TimeoutError as te:
                print(f"\n{'='*80}")
                print(f"⏱️  TIMEOUT ERROR: Process exceeded maximum time limit")
                print(f"{'='*80}\n")
                print(f"Error: {str(te)}")
                print(f"\nThis usually means:")
                print(f"  1. Google Earth Engine is taking too long to respond")
                print(f"  2. Date range is too large (try smaller range)")
                print(f"  3. Network connectivity issues")
                print(f"  4. Server resource constraints")

                sys.stdout.flush()
                sys.stderr.flush()

                RESULTS[job_id]["status"] = "failed"
                RESULTS[job_id]["error"] = f"Timeout: {str(te)}"
                RESULTS[job_id]["error_type"] = "timeout"
                RESULTS[job_id]["progress"] = 100
                RESULTS[job_id]["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                error_log_path = os.path.join(result_dir, "error.log")
                with open(error_log_path, 'w', encoding='utf-8') as error_file:
                    error_file.write(f"TIMEOUT ERROR\n{'='*80}\n")
                    error_file.write(f"Error: {str(te)}\n\nTimestamp: {datetime.now()}\n")
                    error_file.write(f"Parameters: {json.dumps(params, indent=2)}\n")
                    traceback.print_exc(file=error_file)

                print(f"Error details saved to: {error_log_path}")
                traceback.print_exc()
                sys.stdout.flush()
                sys.stderr.flush()

            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__

                print(f"\n{'='*80}")
                print(f"❌ ERROR running process")
                print(f"{'='*80}\n")
                print(f"Error Type: {error_type}")
                print(f"Error Message: {error_msg}")

                if "ee.data.authenticateViaPrivateKey" in error_msg or "credentials" in error_msg.lower():
                    print(f"\n🔐 AUTHENTICATION ERROR DETECTED")
                    print(f"  1. Check if gee-credentials.json exists and is valid")
                    print(f"  2. Verify service account email in .env.production")
                    print(f"  3. Ensure GEE project ID is correct")
                    print(f"  4. Check if service account has Earth Engine API enabled")
                elif "429" in error_msg or "quota" in error_msg.lower():
                    print(f"\n⚠️  QUOTA ERROR DETECTED")
                    print(f"  1. Wait a few minutes before trying again")
                    print(f"  2. Reduce date range")
                    print(f"  3. Check GEE project quotas")
                elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                    print(f"\n⏱️  TIMEOUT ERROR DETECTED")
                    print(f"  1. Try reducing date range")
                    print(f"  2. Check internet connectivity")
                    print(f"  3. Try again later")
                elif "memory" in error_msg.lower() or "memoryerror" in error_type.lower():
                    print(f"\n💾 MEMORY ERROR DETECTED")
                    print(f"  1. Reduce date range")
                    print(f"  2. Check server memory availability")
                    print(f"  3. Restart the API service")
                else:
                    print(f"\n❓ UNKNOWN ERROR - check full error log for details")

                print(f"{'='*80}\n")

                sys.stdout.flush()
                sys.stderr.flush()

                RESULTS[job_id]["status"] = "failed"
                RESULTS[job_id]["error"] = error_msg
                RESULTS[job_id]["error_type"] = error_type
                RESULTS[job_id]["progress"] = 100
                RESULTS[job_id]["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                error_log_path = os.path.join(result_dir, "error.log")
                with open(error_log_path, 'w', encoding='utf-8') as error_file:
                    error_file.write(f"ERROR: {error_type}\n{'='*80}\n")
                    error_file.write(f"Message: {error_msg}\n\nTimestamp: {datetime.now()}\n")
                    error_file.write(f"Parameters: {json.dumps(params, indent=2)}\n\n")
                    error_file.write(f"Full Traceback:\n{'='*80}\n")
                    traceback.print_exc(file=error_file)

                print(f"Error details saved to: {error_log_path}")
                traceback.print_exc()

                sys.stdout.flush()
                sys.stderr.flush()

            finally:
                try:
                    sys.stdout.flush()
                    sys.stderr.flush()
                except:
                    pass

                sys.stdout = original_stdout
                sys.stderr = original_stderr

                try:
                    log_file.flush()
                except:
                    pass

        time.sleep(1)

        try:
            original_stdout.write(f"[INFO] Log file closed and flushed for job {job_id}\n")
            original_stdout.write(f"[INFO] Log file path: {log_file_path}\n")
            original_stdout.write(f"[INFO] Log file exists: {os.path.exists(log_file_path)}\n")
            if os.path.exists(log_file_path):
                original_stdout.write(f"[INFO] Log file size: {os.path.getsize(log_file_path)} bytes\n")
            original_stdout.flush()
        except:
            pass

    except Exception as e:
        RESULTS[job_id]["status"] = "failed"
        RESULTS[job_id]["error"] = str(e)
        RESULTS[job_id]["progress"] = 100

        try:
            error_log_path = os.path.join(result_dir, "error_outer.log")
            with open(error_log_path, 'w', encoding='utf-8') as error_file:
                traceback.print_exc(file=error_file)
        except:
            pass

def run_server(port=8000, host='127.0.0.1'):
    """Run HTTP server on specified port"""
    handler = HidrologiRequestHandler

    results_dir = config.RESULTS_DIR if CONFIG_LOADED else "results"
    os.makedirs(results_dir, exist_ok=True)

    cleanup_old_jobs(max_age_days=30)

    print("="*80)
    print("📄 LOADING EXISTING JOBS FROM DISK")
    print("="*80)
    load_existing_jobs()

    print("="*80)
    print("🌊 API SERVER HIDROLOGI ML (RIVANA)")
    print("="*80)
    print(f"🔍 Starting server on port {port}...")
    print(f"Server running at: http://localhost:{port}")
    print(f"\n🔑 AUTHENTICATION:")
    print(f"  Bearer Token: {'Enabled' if CONFIG_LOADED else 'Enabled (using default)'}")
    print(f"  Token: {'*' * 20}...{config.API_TOKEN[-4:] if CONFIG_LOADED and len(config.API_TOKEN) > 4 else 'change_this'}")
    print(f"  ⚠️  All requests MUST include: Authorization: Bearer YOUR_TOKEN")
    print(f"\n🌐 Translations: {'✅ Loaded' if TRANSLATIONS_LOADED else '⚠️  Not loaded (using fallback)'}")
    print(f"\n📋 AVAILABLE ENDPOINTS:")
    print(f"  POST   /generate                    - Start new calculation")
    print(f"  GET    /status/<job_id>             - Check job status")
    print(f"  GET    /jobs                        - List all jobs")
    print(f"  GET    /result/<job_id>             - List result files (legacy)")
    print(f"  GET    /summary/<job_id>            - Structured analysis summary")
    print(f"  GET    /logs/<job_id>               - Full process log")
    print(f"  GET    /images/<job_id>             - List PNG images only")
    print(f"  GET    /files/<job_id>              - List ALL files (PNG, CSV, JSON)")
    print(f"  GET    /river-map/<job_id>          - River network map info")
    print(f"  GET    /preview/<job_id>/<file>     - Preview file")
    print(f"  GET    /download/<job_id>/<file>    - Download file")
    print(f"  GET    /storage/info                - Storage info & old jobs")
    print(f"  GET    /storage/cleanup             - Manual cleanup old jobs")
    print("="*80)
    print("\nPress Ctrl+C to stop the server...\n")

    socketserver.TCPServer.allow_reuse_address = True

    try:
        with socketserver.TCPServer((host, port), handler) as httpd:
            print(f"✅ Server started at http://{host}:{port}")
            if CONFIG_LOADED:
                print(f"🌍 Environment: {config.environment.upper()}")
                print(f"📊 Debug Mode: {config.DEBUG}")
                print(f"📁 Results Dir: {config.RESULTS_DIR}")
            print(f"📡 Listening for requests...\n")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n\n🛑 Server stopped by user")
                pass

            httpd.server_close()
            print("✅ Server stopped successfully")
    except OSError as e:
        if e.errno == 98 or (hasattr(e, 'winerror') and e.winerror == 10048):
            print(f"\n❌ ERROR: Port {port} is already in use!")
            print(f"\n💡 SOLUTIONS:")
            print(f"   1. Stop the process using port {port}")
            if os.name == 'nt':
                print(f"      Check with: netstat -ano | findstr :{port}")
                print(f"      Kill process: taskkill /PID <PID> /F")
            else:
                print(f"      Check with: sudo lsof -i :{port}")
                print(f"      Kill process: sudo kill -9 <PID>")
                print(f"      Or: sudo pkill -f api_server.py")
            print(f"\n   2. Or use a different port:")
            print(f"      python api_server.py 8001")
            print(f"\n   3. Or wait a few seconds and try again")
        else:
            print(f"\n❌ ERROR: {str(e)}")
            raise

if __name__ == "__main__":
    if CONFIG_LOADED:
        config.print_config()
        port = config.API_PORT
        host = config.API_HOST
    else:
        port = int(os.getenv('API_PORT', '8000'))
        host = os.getenv('API_HOST', '127.0.0.1')

    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
            print(f"✅ Using port from command line: {port}")
        except ValueError:
            print(f"⚠️  Invalid port: {sys.argv[1]}, using port {port}")

    if len(sys.argv) > 2:
        host = sys.argv[2]
        print(f"✅ Using host from command line: {host}")

    run_server(port, host)