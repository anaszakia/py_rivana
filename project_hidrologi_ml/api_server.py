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
    print("âš ï¸  Warning: config.py not found, using defaults")

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
  - params.json - Input parameterers
  - process.log - Complete execution log

River Network Map Features:
---------------------------
🗺️ Interactive HTML Map:
  - Multiple layers (DEM, Flow Accumulation, Water Occurrence)
  - Zoom in/out functionality
  - Toggle layer visibility
  - Multiple basemaps (OpenStreetMap, Terrain, Light)
  - Location marker and buffer circle
  - Legend and title overlay

📊 Static PNG Map:
  - Flow accumulation visualization
  - Analysis point marker
  - Suitable for presentations and reports

📋 Metadata JSON:
  - Location coordinates
  - Flow characteristics (mean, max, min accumulation)
  - Water occurrence statistics
  - Data sources information
  - File paths
"""

# Import pandas untuk summary (optional - akan dicek saat digunakan)
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("âš ï¸  Warning: pandas not available, fitur summary akan terbatas")

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
    print(f"ðŸ§¹ AUTO-CLEANUP: Checking for jobs older than {max_age_days} days")
    print(f"{'='*80}")
    
    for job_id in os.listdir(results_dir):
        job_dir = os.path.join(results_dir, job_id)
        if not os.path.isdir(job_dir):
            continue
        
        try:
            # Get directory creation/modification time
            dir_mtime = os.path.getmtime(job_dir)
            dir_datetime = datetime.fromtimestamp(dir_mtime)
            age_days = (current_time - dir_datetime).days
            
            if age_days > max_age_days:
                # Calculate directory size before deletion
                dir_size = 0
                for dirpath, dirnames, filenames in os.walk(job_dir):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        if os.path.exists(filepath):
                            dir_size += os.path.getsize(filepath)
                
                dir_size_mb = dir_size / (1024 * 1024)
                
                # Delete directory
                import shutil
                shutil.rmtree(job_dir)
                
                deleted_count += 1
                freed_space += dir_size_mb
                
                print(f"  âœ… Deleted job {job_id} (Age: {age_days} days, Size: {dir_size_mb:.2f} MB)")
                
                # Remove from RESULTS if exists
                if job_id in RESULTS:
                    del RESULTS[job_id]
                    
        except Exception as e:
            print(f"  âš ï¸  Error deleting job {job_id}: {e}")
    
    if deleted_count > 0:
        print(f"\nðŸ“Š Cleanup Summary:")
        print(f"  Jobs Deleted: {deleted_count}")
        print(f"  Space Freed: {freed_space:.2f} MB")
    else:
        print(f"  âœ… No jobs older than {max_age_days} days found")
    
    print(f"{'='*80}\n")
    
    return deleted_count, freed_space

def load_existing_jobs():
    """Load existing jobs from results directory"""
    results_dir = get_results_dir()
    if not os.path.exists(results_dir):
        print(f"âš ï¸  Results directory not found: {results_dir}")
        return
    
    job_count = 0
    for job_id in os.listdir(results_dir):
        job_dir = os.path.join(results_dir, job_id)
        if not os.path.isdir(job_dir):
            continue
        
        # Load params.json if exists
        params_file = os.path.join(job_dir, "params.json")
        params = {}
        if os.path.exists(params_file):
            try:
                with open(params_file, 'r') as f:
                    params = json.load(f)
            except Exception as e:
                print(f"âš ï¸  Failed to load params for job {job_id}: {e}")
        
        # Check if files exist to determine status
        png_files = [f for f in os.listdir(job_dir) if f.endswith('.png')]
        csv_files = [f for f in os.listdir(job_dir) if f.endswith('.csv')]
        json_files = [f for f in os.listdir(job_dir) if f.endswith('.json') and f != 'params.json']
        html_files = [f for f in os.listdir(job_dir) if f.endswith('.html')]  # 🌊 NEW: Detect HTML files
        
        # 🌊 Check for river map files
        has_river_map = any('River_Map' in f or 'River_Network' in f for f in png_files + html_files + json_files)
        has_twi = any('TWI' in f for f in png_files + json_files)
        
        # Determine status based on files
        if len(png_files) > 0 or len(csv_files) > 0:
            status = "completed"
            if len(png_files) == 0:
                status = "completed_with_warning"
        else:
            status = "failed"
        
        # Add to RESULTS
        RESULTS[job_id] = {
            "job_id": job_id,
            "status": status,
            "params": params,
            "created_at": "N/A",  # Could parse from process.log if needed
            "completed_at": "N/A",
            "result_path": job_dir,
            "files_generated": {
                "png": len(png_files),
                "csv": len(csv_files),
                "json": len(json_files),
                "html": len(html_files),  # 🌊 NEW: HTML count
                "png_files": png_files,
                "csv_files": csv_files,
                "json_files": json_files,
                "html_files": html_files  # 🌊 NEW: HTML files list
            },
            "river_map": {  # 🌊 NEW: River map info
                "available": has_river_map,
                "interactive_html": any('RIVANA_Interactive_River_Map.html' in f for f in html_files),
                "static_png": any('RIVANA_River_Network_Map.png' in f for f in png_files),
                "metadata_json": any('RIVANA_River_Network_Metadata.json' in f for f in json_files)
            },
            "twi_analysis": {  # 🌊 NEW: TWI analysis info
                "available": has_twi,
                "dashboard_png": any('RIVANA_TWI_Dashboard.png' in f for f in png_files),
                "analysis_json": any('RIVANA_TWI_Analysis.json' in f for f in json_files)
            },
            "progress": 100
        }
        job_count += 1
    
    print(f"âœ… Loaded {job_count} existing jobs from disk\n")

class HidrologiRequestHandler(http.server.BaseHTTPRequestHandler):
    def _check_auth(self):
        """Check Bearer Token authentication"""
        # Get Authorization header
        auth_header = self.headers.get('Authorization', '')
    
        # Extract token from header (handle both "Bearer token" and "token" format)
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]  # Remove "Bearer " prefix
        else:
            token = auth_header  # Use as-is if no prefix
    
        # Get expected token
        if CONFIG_LOADED:
            expected_token = config.API_TOKEN
        else:
            expected_token = "rivana_ml_2024_secure_token_change_this"
    
        # Compare tokens (case-sensitive)
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
        self.send_header('Access-Control-Allow-Origin', '*')  # Untuk CORS
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_OPTIONS(self):
        # Handle preflight requests untuk CORS
        self._set_response()
    
    def _generate_forecast_recommendation(self, df):
        """Generate rekomendasi berdasarkan forecast"""
        try:
            if 'forecast_rainfall' not in df.columns or len(df) < 30:
                return "Data prediksi tidak mencukupi"
            
            recent_forecast = df['forecast_rainfall'].iloc[-30:].mean()
            
            if recent_forecast > 10:
                return "âš ï¸ Potensi HUJAN TINGGI. Siapkan mitigasi banjir, pastikan drainase optimal."
            elif recent_forecast > 5:
                return "âœ… Kondisi NORMAL. Ketersediaan air mencukupi."
            else:
                return "ðŸ”´ Potensi KEKERINGAN. Hemat air, pertimbangkan irigasi alternatif."
        except Exception as e:
            return f"Failed generate rekomendasi: {str(e)}"
    
    def _generate_management_advice(self, df, validation_data):
        """Generate saran pengelolaan berdasarkan kondisi aktual"""
        advice = []
        
        try:
            # Check water balance - dengan backward compatibility untuk 'reservoir'
            kolam_col = 'reservoir' if 'reservoir' in df.columns else 'reservoir' if 'reservoir' in df.columns else None
            if kolam_col:
                avg_reservoir = df[kolam_col].mean()
                if avg_reservoir < 20:
                    advice.append("ðŸ”´ PRIORITAS TINGGI: Retention Pond kritis (<20mm). Terapkan rationing air segera.")
                elif avg_reservoir < 50:
                    advice.append("âš ï¸ PRIORITAS SEDANG: Retention Pond low. Monitor ketat, siapkan contingency plan.")
                else:
                    advice.append("âœ… Kapasitas Retention Pond baik. Lanjutkan operasi normal.")
            
            # Check reliability
            if 'reliability' in df.columns:
                avg_reliability = df['reliability'].mean()
                if avg_reliability < 0.7:
                    advice.append("ðŸ”´ Keandalan sistem <70%. Audit infrastruktur, kurangi kebocoran.")
                elif avg_reliability < 0.85:
                    advice.append("âš ï¸ Keandalan moderat. Tingkatkan maintenance preventif.")
                else:
                    advice.append("âœ… Keandalan sistem excellent (>85%). Maintain standar operasi.")
            
            # Check supply vs demand
            demand_cols = ['demand_Domestic', 'demand_Agriculture', 'demand_Industry']
            supply_cols = ['supply_Domestic', 'supply_Agriculture', 'supply_Industry']
            
            if all(col in df.columns for col in demand_cols + supply_cols):
                total_demand = sum(df[col].sum() for col in demand_cols)
                total_supply = sum(df[col].sum() for col in supply_cols)
                ratio = total_supply / total_demand if total_demand > 0 else 0
                
                if ratio < 0.9:
                    advice.append("ðŸ”´ KRITIS: Supply < 90% demand. Terapkan demand management, cari sumber alternatif.")
                elif ratio < 1.0:
                    advice.append("âš ï¸ Supply mendekati limit. Optimalkan distribusi, reduce losses.")
                else:
                    advice.append("âœ… Supply mencukupi. Fokus pada efisiensi dan konservasi jangka panjang.")
            
            return advice if advice else ["âœ… Sistem berjalan normal, lanjutkan monitoring rutin."]
        
        except Exception as e:
            return [f"âš ï¸ Tidak dapat generate saran: {str(e)}"]

    def safe_get_value(self, df, column, agg_func='mean', default="Data tidak tersedia", format_str="{:.2f}"):
        """Safely get value from DataFrame with fallback"""
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
        """Generate summary text dari hasil analisis - COMPREHENSIVE VERSION"""
        
        # Check pandas availability
        if not PANDAS_AVAILABLE:
            return {
                "error": "Pandas library not available",
                "message": "Install pandas dengan: pip install pandas"
            }
        
        summary = {
            "title": "ðŸ“Š HYDROLOGICAL ANALYSIS SUMMARY",
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
                "job_id": job_data.get("job_id", "N/A"),
                "status": job_data.get("status", "N/A"),
                "created_at": job_data.get("created_at", "N/A"),
                "completed_at": job_data.get("completed_at", "N/A"),
                "files_generated": {
                    "png": job_data.get("files_generated", {}).get("png", 0),
                    "csv": job_data.get("files_generated", {}).get("csv", 0),
                    "json": job_data.get("files_generated", {}).get("json", 0)
                }
            }
            
            # Input Parameterers
            params = job_data.get("params", {})
            summary["input_parameterers"] = {
                "longitude": params.get("longitude", "N/A"),
                "latitude": params.get("latitude", "N/A"),
                "start_date": params.get("start", "N/A"),
                "end_date": params.get("end", "N/A"),
                "periode_analisis": f"{params.get('start', 'N/A')} s/d {params.get('end', 'N/A')}"
            }
            
            # Debug: Log file paths
            print(f"\n{'='*80}")
            print(f"SUMMARY GENERATION DEBUG:")
            print(f"{'='*80}")
            print(f"CSV File Path: {csv_file}")
            print(f"CSV File Exists: {os.path.exists(csv_file)}")
            print(f"Monthly File Path: {monthly_file}")
            print(f"Monthly File Exists: {os.path.exists(monthly_file)}")
            print(f"Validation File Path: {validation_file}")
            print(f"Validation File Exists: {os.path.exists(validation_file)}")
            
            # Baca data CSV jika ada
            if os.path.exists(csv_file):
                print(f"âœ… Reading CSV file: {csv_file}")
                df = pd.read_csv(csv_file)
                print(f"âœ… CSV loaded successfully: {len(df)} rows, {len(df.columns)} columns")
                print(f"ðŸ“‹ CSV Columns: {list(df.columns)[:20]}")  # Print first 20 columns
                
                # Statistik Data - WITH SAFE CHECKS USING HELPER FUNCTION
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
                        "rata_rata": self.safe_get_value(df, 'reliability', 'mean', format_str="{:.1f}%", default="Data tidak tersedia") if 'reliability' not in df.columns else f"{df['reliability'].mean() * 100:.1f}%",
                        "status": self.get_reliability_status(df['reliability'].mean() * 100 if 'reliability' in df.columns and not df['reliability'].isna().all() else 0)
                    }
                }
                
                # Hasil Analisis
                summary["analysis_results"] = {
                    "supply_air": {
                        "total_supply": f"{df['total_supply'].mean():.2f} mm/hari" if 'total_supply' in df.columns else "N/A",
                        "total_demand": f"{df['total_demand'].mean():.2f} mm/hari" if 'total_demand' in df.columns else "N/A",
                        "defisit": f"{df['defisit_total'].mean():.2f} mm/hari" if 'defisit_total' in df.columns else "N/A",
                        "status_supply": self.get_supply_status(df) if len(df) > 0 else "N/A"
                    },
                    "risiko": {
                        "banjir": f"{df['flood_risk'].mean() * 100:.1f}%" if 'flood_risk' in df.columns else "N/A",
                        "kekeringan": f"{df['drought_risk'].mean() * 100:.1f}%" if 'drought_risk' in df.columns else "N/A",
                        "kategori_risiko": self.get_risk_category(df) if len(df) > 0 else "N/A"
                    }
                }
                
                # Quality Air (jika ada)
                if 'WQI' in df.columns:
                    summary["analysis_results"]["water_quality"] = {
                        "WQI_rata_rata": f"{df['WQI'].mean():.1f}/100" if 'WQI' in df.columns else "N/A",
                        "status": self.get_water_quality_index_status(df['WQI'].mean() if 'WQI' in df.columns else 0),
                        "pH": f"{df['pH'].mean():.2f}" if 'pH' in df.columns else "N/A",
                        "DO": f"{df['DO'].mean():.2f} mg/L" if 'DO' in df.columns else "N/A",
                        "TDS": f"{df['TDS'].mean():.2f} mg/L" if 'TDS' in df.columns else "N/A"
                    }
                
                # Ekologi (jika ada)
                if 'ecosystem_health' in df.columns:
                    eco_health_value = df['ecosystem_health'].mean() * 100
                    summary["analysis_results"]["ecosystem_health"] = {
                        "index": f"{eco_health_value:.1f}%",
                        "status": self.get_ecosystem_status(eco_health_value),
                        "habitat_fish": f"{df['fish_HSI'].mean():.2f}" if 'fish_HSI' in df.columns else "N/A",
                        "habitat_vegetation": f"{df['vegetation_HSI'].mean():.2f}" if 'vegetation_HSI' in df.columns else "N/A"
                    }
                else:
                    # Initialize with default values if ecosystem data not available
                    summary["analysis_results"]["ecosystem_health"] = {
                        "index": "N/A",
                        "status": "Data not available",
                        "habitat_fish": "N/A",
                        "habitat_vegetation": "N/A"
                    }
            else:
                # CSV file tidak ditemukan
                print(f"âŒ CSV file not found: {csv_file}")
                summary["error_detail"] = f"CSV file tidak ditemukan: {os.path.basename(csv_file)}"
                summary["statistik_data"]["total_hari"] = "N/A - File not available"
            
            # Water Balance
            if os.path.exists(validation_file):
                print(f"âœ… Reading validation file: {validation_file}")
                with open(validation_file, 'r') as f:
                    wb_data = json.load(f)
                    print(f"âœ… Validation data loaded: {len(wb_data)} keys")
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
                print(f"âŒ Validation file not found: {validation_file}")
                summary["water_balance"]["error_detail"] = "File validasi not available"
            
            print(f"{'='*80}\n")
            
            # ========== BACA FILE JSON TAMBAHAN ==========
            job_dir = os.path.dirname(csv_file) if csv_file else ""
            
            # 1. Model Validation JSON (FIXED: Use correct filename)
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
                            "status": val_data.get('status', 'N/A'),
                            "interpretasi": val_data.get('interpretation', 'N/A')
                        }
                except Exception as e:
                    print(f"Warning: Could not read model validation file: {e}")
            
            # 2. Baseline Comparison JSON (FIXED: Use correct filename)
            baseline_file = os.path.join(job_dir, 'RIVANA_Baseline_Comparison.json')
            if os.path.exists(baseline_file):
                try:
                    with open(baseline_file, 'r') as f:
                        baseline_data = json.load(f)
                        comp_results = baseline_data.get('comparison_results', {}).get('runoff', {})
                        
                        # Safe get with None check
                        avg_improvement = comp_results.get('average_improvement')
                        improvement_str = f"{avg_improvement:.1f}%" if avg_improvement is not None else "N/A"
                        
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
                summary["baseline_comparison"] = {"status": "Not available for this job"}
            
            # 3. TWI Analysis JSON - 🌊 NEW!
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
                    
                    # Extract TWI data
                    twi_data = twi_analysis.get('twi_data', {})
                    flood_zones_raw = twi_analysis.get('flood_zones', [])
                    rtho_recs_raw = twi_analysis.get('rtho_recommendations', [])
                    drainage_recs_raw = twi_analysis.get('drainage_recommendations', [])  # 🚰 NEW: Drainage
                    twi_summary = twi_analysis.get('summary', {})
                    
                    print(f"📊 Extracted data:")
                    print(f"   - twi_data keys: {list(twi_data.keys()) if twi_data else 'None'}")
                    print(f"   - flood_zones count: {len(flood_zones_raw)}")
                    print(f"   - rtho_recommendations count: {len(rtho_recs_raw)}")
                    print(f"   - drainage_recommendations count: {len(drainage_recs_raw)}")
                    print(f"   - summary keys: {list(twi_summary.keys()) if twi_summary else 'None'}")
                    
                    # Map flood zones to expected format for blade template
                    flood_zones_mapped = []
                    for zone in flood_zones_raw:
                        coords = zone.get('coordinates', {})
                        flood_zones_mapped.append({
                            'risk': zone.get('risk_level', 'N/A'),
                            'twi_value': zone.get('twi_enhanced', 0),
                            'area_ha': zone.get('area_affected_ha', 0),
                            'lat': coords.get('latitude', 0),
                            'lon': coords.get('longitude', 0)
                        })
                    
                    # Map RTH recommendations to expected format
                    rtho_recs_mapped = []
                    for rec in rtho_recs_raw:
                        coords = rec.get('coordinates', {})
                        rtho_recs_mapped.append({
                            'priority': rec.get('priority', 'N/A'),
                            'estimated_area_ha': rec.get('area_recommended_ha', 0),
                            'lat': coords.get('latitude', 0),
                            'lon': coords.get('longitude', 0),
                            'reason': rec.get('location_purpose', ' '.join(rec.get('reasons', [])) if isinstance(rec.get('reasons'), list) else 'Strategic location for flood mitigation')
                        })
                    
                    # 🚰 Map Drainage recommendations to expected format
                    drainage_recs_mapped = []
                    for drain in drainage_recs_raw:
                        coords = drain.get('coordinates', {})
                        specs = drain.get('specifications', {})
                        capacity = drain.get('capacity', {})
                        benefits = drain.get('expected_benefits', {})
                        maintenance = drain.get('maintenance_requirements', {})
                        
                        drainage_recs_mapped.append({
                            'location_id': drain.get('location_id', 'N/A'),
                            'priority': drain.get('priority', 'N/A'),
                            'drainage_type': drain.get('drainage_type', 'N/A'),
                            'necessity_score': drain.get('necessity_score', 0),
                            'coordinates': {
                                'lat': coords.get('latitude', 0),
                                'lon': coords.get('longitude', 0)
                            },
                            'specifications': {
                                'channel_width_m': specs.get('channel_width_m', 0),
                                'channel_depth_m': specs.get('channel_depth_m', 0),
                                'channel_slope_percent': specs.get('channel_slope_percent', 0),
                                'lining_type': specs.get('lining_type', 'N/A'),
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
                                'cleaning_frequency': maintenance.get('cleaning_frequency', 'N/A'),
                                'inspection_frequency': maintenance.get('inspection_frequency', 'N/A'),
                                'estimated_annual_cost_million_idr': maintenance.get('estimated_annual_cost_million_idr', 0)
                            },
                            'reasons': drain.get('reasons', [])
                        })
                    
                    summary["twi_analysis"] = {
                        "twi_physics": f"{twi_data.get('twi_physics', 0):.2f}",
                        "ml_correction_factor": f"{twi_data.get('correction_factor', 1.0):.2f}x",
                        "twi_enhanced": f"{twi_data.get('twi_enhanced', 0):.2f}",
                        "risk_level": twi_data.get('risk_level', 'N/A'),
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
                            "moderate_priority": sum(1 for r in rtho_recs_mapped if r.get('priority') == 'MODERATE' or r.get('priority') == 'MEDIUM'),
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
                            "risk": "Area dengan risiko genangan tinggi" if twi_data.get('twi_enhanced', 0) >= 15 else "Area dengan drainase baik",
                            "action": "Perlu mitigasi banjir dan RTH" if len(flood_zones_mapped) > 0 else "Monitoring rutin"
                        }
                    }
                    print(f"✅ TWI analysis data successfully added to summary")
                    print(f"   - TWI Enhanced value: {twi_data.get('twi_enhanced', 0)}")
                    print(f"   - Risk Level: {twi_data.get('risk_level', 'N/A')}")
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
                print(f"Checking if TWI Dashboard PNG exists...")
                twi_dashboard = os.path.join(job_dir, 'RIVANA_TWI_Dashboard.png')
                print(f"TWI Dashboard PNG: {os.path.exists(twi_dashboard)}")
                
                if os.path.exists(twi_dashboard):
                    print(f"⚠️ ISSUE: TWI Dashboard generated but JSON not found!")
                    print(f"This indicates TWI calculation ran but JSON export failed")
                    summary["twi_analysis"] = {
                        "status": "TWI Dashboard generated but JSON file missing",
                        "error": "Check logs for TWI JSON export errors",
                        "expected_file": "RIVANA_TWI_Analysis.json",
                        "note": "Dashboard PNG exists, indicating TWI ran but JSON save failed"
                    }
                else:
                    print(f"ℹ️ TWI analysis not available for this job")
                    summary["twi_analysis"] = {
                        "status": "Not available for this job",
                        "note": "TWI analysis was not generated during processing"
                    }
            
            print(f"{'='*80}\n")
            
            # 4. River Network Metadata JSON - 🌊 NEW!
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
            
            # 5. Extract data tambahan dari CSV untuk morfologi, ekologi, dll
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                
                # Supply & Demand Detail
                if 'total_supply' in df.columns:
                    summary["supply_demand"] = {
                        "supply_rata_rata": f"{df['total_supply'].mean():.2f} mm/hari",
                        "demand_rata_rata": f"{df['total_demand'].mean():.2f} mm/hari",
                        "ratio": f"{(df['total_supply'].mean() / df['total_demand'].mean() * 100):.1f}%" if df['total_demand'].mean() > 0 else "N/A",
                        "defisit_maksimal": f"{df['defisit_total'].max():.2f} mm" if 'defisit_total' in df.columns else "N/A",
                        "status": self.get_supply_status(df)
                    }
                
                # Alokasi Per Sektor - WITH SAFE COLUMN CHECK
                sektor_cols = ['supply_Domestic', 'supply_Agriculture', 'supply_Industry', 'supply_Environmental']
                if all(col in df.columns for col in sektor_cols):
                    try:
                        summary["alokasi_sektor"] = {
                            "Domestic": {
                                "rata_rata": f"{df['supply_Domestic'].mean():.2f} mm/hari",
                                "total": f"{df['supply_Domestic'].sum():.2f} mm",
                                "pemenuhan": f"{(df['supply_Domestic'].mean() / 0.4 * 100):.1f}%" if df['supply_Domestic'].mean() > 0 else "0%"
                            },
                            "Agriculture": {
                                "rata_rata": f"{df['supply_Agriculture'].mean():.2f} mm/hari",
                                "total": f"{df['supply_Agriculture'].sum():.2f} mm",
                                "pemenuhan": f"{(df['supply_Agriculture'].mean() / 0.8 * 100):.1f}%" if df['supply_Agriculture'].mean() > 0 else "0%"
                            },
                            "Industry": {
                                "rata_rata": f"{df['supply_Industry'].mean():.2f} mm/hari",
                                "total": f"{df['supply_Industry'].sum():.2f} mm",
                                "pemenuhan": f"{(df['supply_Industry'].mean() / 0.2 * 100):.1f}%" if df['supply_Industry'].mean() > 0 else "0%"
                            },
                            "Environmental": {
                                "rata_rata": f"{df['supply_Environmental'].mean():.2f} mm/hari",
                                "total": f"{df['supply_Environmental'].sum():.2f} mm",
                                "pemenuhan": f"{(df['supply_Environmental'].mean() / 0.3 * 100):.1f}%" if df['supply_Environmental'].mean() > 0 else "0%"
                            }
                        }
                        
                        # â­ ADD for hasil_analisis.supply_air_per_sektor (for view SECTION 1)
                        summary["analysis_results"]["water_supply_per_sector"] = {
                            "Domestic": {
                                "quota": "0.4 mm/hari",
                                "alokasi": f"{df['supply_Domestic'].mean():.2f} mm/hari" if 'supply_Domestic' in df.columns else "N/A",
                                "prioritas": "10 (Highest)",
                                "pemenuhan": f"{(df['supply_Domestic'].mean() / 0.4 * 100):.1f}%" if 'supply_Domestic' in df.columns else "N/A"
                            },
                            "Agriculture": {
                                "quota": "0.8 mm/hari",
                                "alokasi": f"{df['supply_Agriculture'].mean():.2f} mm/hari" if 'supply_Agriculture' in df.columns else "N/A",
                                "prioritas": "7",
                                "pemenuhan": f"{(df['supply_Agriculture'].mean() / 0.8 * 100):.1f}%" if 'supply_Agriculture' in df.columns else "N/A"
                            },
                            "Industry": {
                                "quota": "0.2 mm/hari",
                                "alokasi": f"{df['supply_Industry'].mean():.2f} mm/hari" if 'supply_Industry' in df.columns else "N/A",
                                "prioritas": "5",
                                "pemenuhan": f"{(df['supply_Industry'].mean() / 0.2 * 100):.1f}%" if 'supply_Industry' in df.columns else "N/A"
                            },
                            "Environmental": {
                                "quota": "0.3 mm/hari",
                                "alokasi": f"{df['supply_Environmental'].mean():.2f} mm/hari" if 'supply_Environmental' in df.columns else "N/A",
                                "prioritas": "9",
                                "pemenuhan": f"{(df['supply_Environmental'].mean() / 0.3 * 100):.1f}%" if 'supply_Environmental' in df.columns else "N/A"
                            }
                        }
                    except Exception as e:
                        print(f"âš ï¸ Warning: Could not generate sektor allocation data: {e}")
                        summary["alokasi_sektor"] = {"error": "Data alokasi sektor not available"}
                        summary["analysis_results"]["water_supply_per_sector"] = {"error": "Data not available"}
                
                # â­ TAMBAHKAN sumber_air (for view SECTION 2)
                if 'total_supply' in df.columns:
                    total_supply = df['total_supply'].mean()
                    summary["analysis_results"]["water_sources"] = {
                        "Sungai": {
                            "supply": f"{total_supply * 0.60:.2f} mm/hari",
                            "biaya": "Rp 150/mÂ³",
                            "kontribusi": "60%"
                        },
                        "Diversi": {
                            "supply": f"{total_supply * 0.25:.2f} mm/hari",
                            "biaya": "Rp 200/mÂ³",
                            "kontribusi": "25%"
                        },
                        "Air Soil": {
                            "supply": f"{total_supply * 0.15:.2f} mm/hari",
                            "biaya": "Rp 350/mÂ³",
                            "kontribusi": "15%"
                        }
                    }
                
                # â­ TAMBAHKAN ekonomi detail (for view SECTION 3) - WITH SAFE COLUMN CHECK
                if 'total_supply' in df.columns and 'total_demand' in df.columns:
                    try:
                        total_vol = df['total_supply'].sum()
                        biaya_operasi = total_vol * 150  # Rp/mÂ³
                        biaya_pemeliharaan = biaya_operasi * 0.15
                        biaya_energi = biaya_operasi * 0.20
                        total_biaya = biaya_operasi + biaya_pemeliharaan + biaya_energi
                        
                        # Check if sektor supply columns exist
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
                                "Biaya Operasi": f"Rp {biaya_operasi:,.0f}",
                                "Biaya Pemeliharaan": f"Rp {biaya_pemeliharaan:,.0f}",
                                "Biaya Energi": f"Rp {biaya_energi:,.0f}",
                                "Manfaat Agriculture": f"Rp {manfaat_pertanian:,.0f}",
                                "Manfaat Domestic": f"Rp {manfaat_domestik:,.0f}",
                                "Manfaat Industry": f"Rp {manfaat_industri:,.0f}"
                            }
                        }
                    except Exception as e:
                        print(f"âš ï¸ Warning: Could not generate ekonomi data: {e}")
                        summary["analysis_results"]["economics"] = {"error": "Economic data not available"}
                
                # â­ Forecast Rainfall 30 Day & Forecast (LENGKAP)
                summary["prediksi_30_hari"] = {
                    "rainfall": {
                        "rata_rata": f"{df['forecast_rainfall'].mean():.2f} mm/hari" if 'forecast_rainfall' in df.columns else "Data belum tersedia",
                        "minimum": f"{df['forecast_rainfall'].min():.2f} mm" if 'forecast_rainfall' in df.columns else "N/A",
                        "maximum": f"{df['forecast_rainfall'].max():.2f} mm" if 'forecast_rainfall' in df.columns else "N/A",
                        "total": f"{df['forecast_rainfall'].sum():.2f} mm" if 'forecast_rainfall' in df.columns else "N/A"
                    },
                    "reservoir": {
                        "kondisi_saat_ini": f"{df['reservoir'].iloc[-1]:.2f} mm" if len(df) > 0 and 'reservoir' in df.columns else "N/A",
                        "prediksi_30_hari": f"{df['forecast_reservoir'].iloc[-1]:.2f} mm" if 'forecast_reservoir' in df.columns and len(df) > 0 else "N/A",
                        "persentase_capacity": f"{(df['reservoir'].iloc[-1] / 100.0 * 100):.1f}%" if len(df) > 0 and 'reservoir' in df.columns else "N/A"
                    },
                    "reliability": {
                        "saat_ini": f"{df['reliability'].mean() * 100:.1f}%" if 'reliability' in df.columns else "N/A",
                        "prediksi_30_hari": f"{df['reliability'].iloc[-30:].mean() * 100:.1f}%" if 'reliability' in df.columns and len(df) >= 30 else "N/A",
                        "tren": "Stable" if 'reliability' in df.columns and len(df) >= 30 and abs(df['reliability'].mean() - df['reliability'].iloc[-30:].mean()) < 0.05 else "Berubah"
                    },
                    "rekomendasi_forecast": self._generate_forecast_recommendation(df) if len(df) > 0 else "Data tidak cukup"
                }
                
                # Operasi Retention Pond
                if 'Retention Pond_action' in df.columns:
                    summary["operasi_reservoir"] = {
                        "volume_saat_ini": f"{df['reservoir'].iloc[-1]:.2f} mm" if len(df) > 0 else "N/A",
                        "persentase_capacity": f"{(df['reservoir'].iloc[-1] / 100.0 * 100):.1f}%" if len(df) > 0 else "N/A",
                        "rekomendasi_aksi": df['Retention Pond_action'].iloc[-1] if len(df) > 0 and 'Retention Pond_action' in df.columns else "Maintain",
                        "tren_30_hari": "Naik" if len(df) >= 30 and df['reservoir'].iloc[-1] > df['reservoir'].iloc[-30] else "Turun" if len(df) >= 30 else "Stable"
                    }
                
                # Morfologi - TAMBAHKAN ke hasil_analisis (for view SECTION 4)
                morph_cols = ['channel_width', 'slope', 'total_sedimentt']
                if any(col in df.columns for col in morph_cols):
                    # Safe slope formatting
                    slope_value = df['slope'].mean() if 'slope' in df.columns else None
                    if slope_value is not None:
                        slope_str = f"{slope_value:.2f}°"
                    else:
                        slope_str = "N/A"
                    
                    morfologi_data = {
                        "lebar_sungai": f"{df['channel_width'].mean():.2f} m" if 'channel_width' in df.columns else "N/A",
                        "kemiringan": slope_str,
                        "beban_sediment": f"{df['total_sedimentt'].mean():.2f} ton/hari" if 'total_sedimentt' in df.columns else "N/A",
                        "erosion_rata_rata": f"{df['erosion_rate'].mean():.2f} mm/tahun" if 'erosion_rate' in df.columns else "N/A"
                    }
                    summary["morfologi"] = morfologi_data
                    # PENTING: Tambahkan juga ke hasil_analisis untuk view
                    summary["analysis_results"]["morfologi"] = morfologi_data
                
                # Ekologi Detail
                if 'ecosystem_health' in df.columns:
                    eco_health = df['ecosystem_health'].mean() * 100
                    summary["ekologi"] = {
                        "ecosystem_health": f"{eco_health:.1f}%",
                        "habitat_ikan": f"{df['fish_HSI'].mean():.2f}" if 'fish_HSI' in df.columns else "N/A",
                        "habitat_vegevapotranspirationasi": f"{df['vegetation_HSI'].mean():.2f}" if 'vegetation_HSI' in df.columns else "N/A",
                        "temperature_air": f"{df['temperature'].mean():.1f}Â°C" if 'temperature' in df.columns else "N/A",
                        "status": self.get_ecosystem_status(eco_health)
                    }
                    
                    # â­ TAMBAHKAN habitat detail untuk kesehatan_ekosistem di hasil_analisis
                    if 'kesehatan_ekosistem' not in summary["analysis_results"]:
                        summary["analysis_results"]["ecosystem_health"] = {}
                    
                    summary["analysis_results"]["ecosystem_health"]["habitat"] = {
                        "fish": {
                            "HSI": f"{df['fish_HSI'].mean():.2f}" if 'fish_HSI' in df.columns else "N/A",
                            "status": "Good" if 'fish_HSI' in df.columns and df['fish_HSI'].mean() > 0.6 else "Fair"
                        },
                        "vegetation": {
                            "HSI": f"{df['vegetation_HSI'].mean():.2f}" if 'vegetation_HSI' in df.columns else "N/A",
                            "status": "Good" if 'vegetation_HSI' in df.columns and df['vegetation_HSI'].mean() > 0.6 else "Fair"
                        }
                    }
                
                # â­ TAMBAHKAN summary_sistem (for view SECTION 5)
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
                            "status": "Surplus" if supply_demand_ratio >= 110 else "Seimbang" if supply_demand_ratio >= 90 else "Defisit",
                            "warna": "green" if supply_demand_ratio >= 110 else "yellow" if supply_demand_ratio >= 90 else "red"
                        },
                        "risiko_banjir": {
                            "nilai": f"{flood_risk:.1f}%",
                            "status": "High" if flood_risk > 30 else "Medium" if flood_risk > 15 else "Low",
                            "warna": "red" if flood_risk > 30 else "yellow" if flood_risk > 15 else "green"
                        },
                        "risiko_kekeringan": {
                            "nilai": f"{drought_risk:.1f}%",
                            "status": "High" if drought_risk > 30 else "Medium" if drought_risk > 15 else "Low",
                            "warna": "red" if drought_risk > 30 else "yellow" if drought_risk > 15 else "green"
                        }
                    }
            
            # â­â­â­ COMPREHENSIVE SECTIONS - MATCH TERMINAL OUTPUT â­â­â­
            
            # 1. WATER BALANCE ANALYSIS (LENGKAP)
            if os.path.exists(validation_file):
                try:
                    with open(validation_file, 'r') as f:
                        validation_data = json.load(f)
                    
                    summary["analisis_keseimbangan_air"] = {
                        "validasi_massa": {
                            "error_persentase": f"{validation_data.get('mass_balance_error_pct', 0):.4f}%",
                            "status": "âœ… VALID" if abs(validation_data.get('mass_balance_error_pct', 100)) < 1.0 else "âš ï¸ PERLU REVIEW",
                            "residual": f"{validation_data.get('residual', 0):.2f} mm",
                            "keterangan": "Keseimbangan massa terjaga" if abs(validation_data.get('mass_balance_error_pct', 100)) < 1.0 else "Perlu kalibrasi ulang"
                        },
                        "komponen_input": {
                            "rainfall": f"{validation_data.get('input_rainfall', 0):.2f} mm",
                            "inflow_sungai": f"{validation_data.get('input_inflow', 0):.2f} mm" if 'input_inflow' in validation_data else "N/A",
                            "groundwater_recharge": f"{validation_data.get('input_groundwater', 0):.2f} mm" if 'input_groundwater' in validation_data else "N/A",
                            "total_input": f"{validation_data.get('total_input', 0):.2f} mm"
                        },
                        "komponen_output": {
                            "evapotranspirasi": f"{validation_data.get('output_ET', 0):.2f} mm",
                            "runoff": f"{validation_data.get('output_runoff', 0):.2f} mm",
                            "total_supply": f"{validation_data.get('output_supply', 0):.2f} mm",
                            "percolation": f"{validation_data.get('output_percolation', 0):.2f} mm" if 'output_percolation' in validation_data else "N/A",
                            "total_output": f"{validation_data.get('total_output', 0):.2f} mm"
                        },
                        "perubahan_storage": {
                            "Retention Pond": f"{validation_data.get('storage_Retention Pond', 0):.2f} mm",
                            "soil_moisture": f"{validation_data.get('storage_soil', 0):.2f} mm" if 'storage_soil' in validation_data else "N/A",
                            "total_storage_change": f"{validation_data.get('total_storage_change', 0):.2f} mm"
                        },
                        "kesimpulan": validation_data.get('conclusion', 'Analisis keseimbangan air successfully')
                    }
                except Exception as e:
                    summary["analisis_keseimbangan_air"] = {"error": f"Tidak dapat membaca validation file: {str(e)}"}
            else:
                summary["analisis_keseimbangan_air"] = {"status": "File validasi belum tersedia"}
            
            # 2. ANALISIS KONDISI SUNGAI DAN TANAH (LENGKAP)
            if os.path.exists(csv_file):
                kondisi_sungai_soil_storage = {}
                
                # Morfologi Sungai (gunakan any karena tidak semua kolom selalu ada)
                if 'channel_width' in df.columns or 'slope_degree' in df.columns or 'total_sedimentt' in df.columns:
                    morfologi_sungai = {}
                    
                    if 'channel_width' in df.columns:
                        morfologi_sungai["lebar_sungai"] = {
                            "rata_rata": f"{df['channel_width'].mean():.2f} m",
                            "min": f"{df['channel_width'].min():.2f} m",
                            "max": f"{df['channel_width'].max():.2f} m",
                            "status": "Normal" if 10 <= df['channel_width'].mean() <= 50 else "Perlu Monitoring"
                        }
                    
                    if 'slope_degree' in df.columns:
                        morfologi_sungai["kemiringan"] = {
                            "rata_rata": f"{df['slope_degree'].mean():.2f}°",
                            "kategori": "Landai (<2°)" if df['slope_degree'].mean() < 2 else "Sedang (2-8°)" if df['slope_degree'].mean() < 8 else "Curam (>8°)"
                        }
                    
                    if 'total_sedimentt' in df.columns:
                        morfologi_sungai["beban_sediment"] = {
                            "rata_rata": f"{df['total_sedimentt'].mean():.2f} ton/hari",
                            "total": f"{df['total_sedimentt'].sum():.2f} ton",
                            "status": "Normal" if df['total_sedimentt'].mean() < 100 else "Tinggi - Perlu Pengerukan"
                        }
                    
                    if morfologi_sungai:  # Only add if there's data
                        kondisi_sungai_soil_storage["morfologi_sungai"] = morfologi_sungai
                
                # Soil Condition
                if all(col in df.columns for col in ['soil_moisture', 'infiltration', 'percolation']):
                    kondisi_sungai_soil_storage["kondisi_soil_storage"] = {
                        "soil_moisture": {
                            "rata_rata": f"{df['soil_moisture'].mean():.2f} mm",
                            "min": f"{df['soil_moisture'].min():.2f} mm",
                            "max": f"{df['soil_moisture'].max():.2f} mm",
                            "status": "Optimal" if 20 <= df['soil_moisture'].mean() <= 40 else "Kering" if df['soil_moisture'].mean() < 20 else "Jenuh"
                        },
                        "infiltration": {
                            "rata_rata": f"{df['infiltration'].mean():.2f} mm/hari",
                            "total": f"{df['infiltration'].sum():.2f} mm",
                            "capacity": "Good" if df['infiltration'].mean() > 2 else "Low"
                        },
                        "percolation": {
                            "rata_rata": f"{df['percolation'].mean():.2f} mm/hari",
                            "total": f"{df['percolation'].sum():.2f} mm",
                            "ke_groundwater": f"{df['percolation'].sum() * 0.7:.2f} mm (estimasi 70%)"
                        }
                    }
                
                # Erosi & Transportasi Sedimen
                if 'erosion_rate' in df.columns and 'sediment_transport' in df.columns:
                    kondisi_sungai_soil_storage["erosion_sediment"] = {
                        "laju_erosion": {
                            "nilai": f"{df['erosion_rate'].mean():.2f} ton/ha/tahun",
                            "status": "Low" if df['erosion_rate'].mean() < 10 else "Medium" if df['erosion_rate'].mean() < 50 else "High",
                            "total_tahunan": f"{df['erosion_rate'].sum():.2f} ton/ha"
                        },
                        "transport_sediment": {
                            "capacity": f"{df['sediment_transport'].mean():.2f} kg/s",
                            "efisiensi": f"{(df['sediment_transport'].mean() / df['total_sedimentt'].mean() * 100):.1f}%" if 'total_sedimentt' in df.columns and df['total_sedimentt'].mean() > 0 else "N/A"
                        }
                    }
                
                summary["analisis_kondisi_sungai_soil_storage"] = kondisi_sungai_soil_storage
            else:
                summary["analisis_kondisi_sungai_soil_storage"] = {"status": "Data CSV belum tersedia"}
            
            # 3. SARAN PENGELOLAAN (COMPREHENSIVE)
            validation_data = {}
            if os.path.exists(validation_file):
                try:
                    with open(validation_file, 'r') as f:
                        validation_data = json.load(f)
                except:
                    pass
            
            summary["saran_pengelolaan"] = self._generate_management_advice(df, validation_data)
            
            # 4. SARAN PERBAIKAN KONDISI (SPECIFIC ACTIONS)
            saran_perbaikan = []
            
            if os.path.exists(csv_file):
                # Berdasarkan Morfologi
                if 'total_sedimentt' in df.columns and df['total_sedimentt'].mean() > 100:
                    saran_perbaikan.append({
                        "kategori": "Sedimentasi",
                        "prioritas": "TINGGI",
                        "masalah": f"Beban sediment high ({df['total_sedimentt'].mean():.2f} kg/s)",
                        "solusi": [
                            "Lakukan dredging/pengerukan sungai secara berkala",
                            "Bangun sedimentt trap di hulu",
                            "Reboisasi catchment area untuk mengurangi erosion"
                        ],
                        "estimasi_biaya": "Rp 500 juta - 2 miliar (tergantung skala)",
                        "timeline": "3-6 bulan"
                    })
                
                # Berdasarkan Humidity Soil
                if 'soil_moisture' in df.columns and df['soil_moisture'].mean() < 20:
                    saran_perbaikan.append({
                        "kategori": "Konservasi Air Soil",
                        "prioritas": "SEDANG",
                        "masalah": f"Humidity soil_storage low ({df['soil_moisture'].mean():.2f} mm)",
                        "solusi": [
                            "Buat sumur resapan untuk meningkatkan infiltration",
                            "Implementasi rainwater harvesting",
                            "Terapkan mulching untuk mengurangi evaporasi"
                        ],
                        "estimasi_biaya": "Rp 100-300 juta",
                        "timeline": "2-4 bulan"
                    })
                
                # Berdasarkan Retention Pond
                if 'reservoir' in df.columns and df['reservoir'].mean() < 30:
                    saran_perbaikan.append({
                        "kategori": "Kapasitas Retention Pond",
                        "prioritas": "TINGGI",
                        "masalah": f"Kapasitas Retention Pond kritis ({df['reservoir'].mean():.2f} mm)",
                        "solusi": [
                            "Evaluasi capacity Retention Pond, pertimbangkan peningkatan/penambahan",
                            "Optimalkan sistem distribusi untuk reduce losses",
                            "Implementasi demand management (rationing, pricing)",
                            "Explore alternative water sources (groundwater, recycled water)"
                        ],
                        "estimasi_biaya": "Rp 5-20 miliar (upgrading infrastruktur)",
                        "timeline": "12-24 bulan"
                    })
                
                # Berdasarkan Flood Risk
                if 'flood_risk' in df.columns and df['flood_risk'].mean() > 0.3:
                    saran_perbaikan.append({
                        "kategori": "Mitigasi Banjir",
                        "prioritas": "TINGGI",
                        "masalah": f"Risiko banjir high ({df['flood_risk'].mean()*100:.1f}%)",
                        "solusi": [
                            "Bangun/perbaiki sistem drainase dan kanal banjir",
                            "Buat detention pond/kolam retensi",
                            "Tingkatkan capacity spillway",
                            "Implementasi early warning system"
                        ],
                        "estimasi_biaya": "Rp 1-5 miliar",
                        "timeline": "6-12 bulan"
                    })
                
                # Berdasarkan Drought Risk
                if 'drought_risk' in df.columns and df['drought_risk'].mean() > 0.3:
                    saran_perbaikan.append({
                        "kategori": "Mitigasi Kekeringan",
                        "prioritas": "TINGGI",
                        "masalah": f"Risiko kekeringan high ({df['drought_risk'].mean()*100:.1f}%)",
                        "solusi": [
                            "Bangun tambahan reservoir/embung kecil",
                            "Kembangkan groundwater storage",
                            "Implementasi water-efficient irrigation (drip, sprinkler)",
                            "Promosikan konservasi air dan water-saving technologies"
                        ],
                        "estimasi_biaya": "Rp 2-10 miliar",
                        "timeline": "12-18 bulan"
                    })
                
                # Berdasarkan System Reliability
                if 'reliability' in df.columns and df['reliability'].mean() < 0.7:
                    saran_perbaikan.append({
                        "kategori": "Keandalan Infrastruktur",
                        "prioritas": "SEDANG",
                        "masalah": f"Keandalan sistem low ({df['reliability'].mean()*100:.1f}%)",
                        "solusi": [
                            "Audit menyeluruh infrastruktur distribusi",
                            "Perbaiki kebocoran pipa (leak detection & repair)",
                            "Upgrade pompa dan valve yang sudah tua",
                            "Implementasi SCADA system untuk monitoring real-time"
                        ],
                        "estimasi_biaya": "Rp 500 juta - 3 miliar",
                        "timeline": "6-9 bulan"
                    })
            
            # Tambahkan rekomendasi umum jika tidak ada masalah spesifik
            if not saran_perbaikan:
                saran_perbaikan.append({
                    "kategori": "Maintenance Rutin",
                    "prioritas": "NORMAL",
                    "masalah": "Sistem berjalan baik, fokus pada preventive maintenance",
                    "solusi": [
                        "Lanjutkan monitoring rutin semua parameterer",
                        "Terapkan predictive maintenance",
                        "Update database dan model secara berkala",
                        "Training staff untuk optimasi operasi"
                    ],
                    "estimasi_biaya": "Rp 50-200 juta/tahun (operational)",
                    "timeline": "Ongoing"
                })
            
            summary["saran_perbaikan_kondisi"] = saran_perbaikan
            
            # Quality Data
            summary["kualitas_data"] = {
                "kelengkapan_data": "100%" if os.path.exists(csv_file) else "Data tidak lengkap",
                "periode_valid": "Ya" if os.path.exists(csv_file) else "Tidak",
                "file_tersedia": {
                    "visualisasi": f"{job_data.get('files_generated', {}).get('png', 0)} file PNG",
                    "data_csv": f"{job_data.get('files_generated', {}).get('csv', 0)} file CSV",
                    "metadata": f"{job_data.get('files_generated', {}).get('json', 0)} file JSON"
                }
            }
            
            # Rekomendasi
            summary["rekomendasi"] = self.generate_recommendations(summary)
            
        except KeyError as e:
            summary["error"] = f"Missing column in data: {str(e)}"
            summary["error_detail"] = f"Kolom '{str(e)}' tidak ditemukan di CSV. Data mungkin belum lengkap ter-generate."
            print(f"âŒ KeyError in generate_summary_text: {e}")
            traceback.print_exc()
        except Exception as e:
            summary["error"] = f"Error generating summary: {str(e)}"
            summary["error_detail"] = "Terjadi kesalahan saat creating summary hasil"
            print(f"âŒ Exception in generate_summary_text: {e}")
            traceback.print_exc()
        
        return summary
    
    def get_reliability_status(self, reliability):
        """Get status reliability sistem"""
        if reliability >= 90:
            return "Excellent - Sistem sangat andal"
        elif reliability >= 75:
            return "Good - Sistem cukup andal"
        elif reliability >= 60:
            return "Fair - Perlu peningkatan"
        else:
            return "Poor - Perlu intervensi segera"
    
    def get_supply_status(self, df):
        """Get status supply air"""
        if 'total_supply' in df.columns and 'total_demand' in df.columns:
            ratio = df['total_supply'].mean() / df['total_demand'].mean() if df['total_demand'].mean() > 0 else 0
            if ratio >= 1.1:
                return "Surplus - Supply melebihi demand"
            elif ratio >= 0.9:
                return "Seimbang - Supply sesuai demand"
            else:
                return "Defisit - Supply kurang dari demand"
        return "N/A"
    
    def get_risk_category(self, df):
        """Get kategori risiko"""
        flood_risk = df['flood_risk'].mean() * 100 if 'flood_risk' in df.columns else 0
        drought_risk = df['drought_risk'].mean() * 100 if 'drought_risk' in df.columns else 0
        
        if flood_risk > 30:
            return "High - Risiko banjir high"
        elif drought_risk > 30:
            return "High - Risiko kekeringan high"
        elif flood_risk > 15 or drought_risk > 15:
            return "Medium - Perlu monitoring"
        else:
            return "Low - Kondisi normal"
    
    def get_water_quality_index_status(self, water_quality_index):
        """Get status Water Quality Index"""
        if water_quality_index >= 90:
            return "Excellent"
        elif water_quality_index >= 70:
            return "Good"
        elif water_quality_index >= 50:
            return "Fair"
        elif water_quality_index >= 30:
            return "Bad"
        else:
            return "Sangat Bad"
    
    def get_ecosystem_status(self, health):
        """Get status kesehatan ekosistem"""
        if health >= 80:
            return "Sangat Sehat"
        elif health >= 60:
            return "Sehat"
        elif health >= 40:
            return "Fair Sehat"
        else:
            return "Poor Sehat"
    
    def get_balance_status(self, error_pct):
        """Get status water balance"""
        if abs(error_pct) < 5:
            return "Excellent - Error minimal"
        elif abs(error_pct) < 10:
            return "Good - Error dalam batas wajar"
        elif abs(error_pct) < 20:
            return "Fair - Perlu verifikasi"
        else:
            return "Poor - Perlu review data"
    
    def get_file_display_order(self, filename):
        """Get display order priority untuk sorting files"""
        # Priority order: HTML/Map files first, PNG (dashboard priority), CSV, JSON, then others
        if filename.endswith('.html'):
            # Interactive map - highest priority
            if 'Interactive_River_Map' in filename or 'River_Map' in filename:
                return (0, 0, filename)
            else:
                return (0, 1, filename)
        elif filename.endswith('.png'):
            # PNG files - sort by importance
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
            # CSV files
            csv_priority = {
                'RIVANA_Hasil_Complete.csv': 1,
                'RIVANA_Monthly_WaterBalance.csv': 2,
                'RIVANA_Prediksi_30Hari.csv': 3,
                'GEE_Raw_Data.csv': 4,
                'RIVANA_WaterBalance_Indices.csv': 5
            }
            return (2, csv_priority.get(filename, 99), filename)
        elif filename.endswith('.json'):
            # JSON files
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
            return (99, 99, filename)  # Other files last
    
    def generate_recommendations(self, summary):
        """Generate rekomendasi berdasarkan hasil analisis"""
        recommendations = []
        
        try:
            # Rekomendasi berdasarkan reliability
            reliability_text = summary.get("statistik_data", {}).get("reliability_sistem", {}).get("rata_rata", "")
            if reliability_text and isinstance(reliability_text, str):
                # Validate before converting to float - handle N/A values
                if reliability_text not in ["N/A", "Data tidak tersedia", ""]:
                    try:
                        reliability_value = reliability_text.replace("%", "").strip()
                        # Check if it's a valid number
                        if reliability_value.replace(".", "", 1).isdigit():
                            reliability = float(reliability_value)
                            if reliability < 75:
                                recommendations.append({
                                    "kategori": "System Reliability",
                                    "prioritas": "High",
                                    "rekomendasi": "Tingkatkan reliability sistem dengan meningkatkan capacity Retention Pond atau menambah sumber air alternatif"
                                })
                    except (ValueError, AttributeError) as e:
                        print(f"⚠️ Warning: Could not convert reliability '{reliability_text}' to float: {e}")
            
            # Rekomendasi berdasarkan supply
            supply_status = summary.get("analysis_results", {}).get("supply_air", {}).get("status_supply", "")
            if "Defisit" in supply_status:
                recommendations.append({
                    "kategori": "Supply Air",
                    "prioritas": "High",
                    "rekomendasi": "Segera lakukan konservasi air dan cari sumber air tambahan untuk mengatasi defisit"
                })
            
            # Rekomendasi berdasarkan risiko
            risk_category = summary.get("analysis_results", {}).get("risiko", {}).get("kategori_risiko", "")
            if "High" in risk_category:
                if "banjir" in risk_category.lower():
                    recommendations.append({
                        "kategori": "Mitigasi Banjir",
                        "prioritas": "High",
                        "rekomendasi": "Persiapkan sistem early warning dan infrastruktur untuk menghadapi risiko banjir high"
                    })
                elif "kekeringan" in risk_category.lower():
                    recommendations.append({
                        "kategori": "Mitigasi Kekeringan",
                        "prioritas": "High",
                        "rekomendasi": "Implementasikan strategi penghematan air dan pengembangan sumber air alternatif"
                    })
            
            # Rekomendasi kualitas air
            if "water_quality" in summary.get("analysis_results", {}):
                water_quality_index_status = summary["analysis_results"]["water_quality"].get("status", "")
                if water_quality_index_status in ["Bad", "Sangat Bad"]:
                    recommendations.append({
                        "kategori": "Quality Air",
                        "prioritas": "High",
                        "rekomendasi": "Perbaiki kualitas air dengan pengolahan yang sesuai dan monitoring rutin parameterer kualitas air"
                    })
            
            # Rekomendasi ekosistem
            if "ecosystem_health" in summary.get("analysis_results", {}):
                eco_status = summary["analysis_results"]["ecosystem_health"].get("status", "")
                if eco_status in ["Poor Sehat"]:
                    recommendations.append({
                        "kategori": "Kesehatan Ekosistem",
                        "prioritas": "Medium",
                        "rekomendasi": "Lakukan restorasi ekosistem dan jaga environmental flow untuk kesehatan habitat"
                    })
            
            # Rekomendasi umum jika tidak ada masalah
            if len(recommendations) == 0:
                recommendations.append({
                    "kategori": "Pemeliharaan Rutin",
                    "prioritas": "Normal",
                    "rekomendasi": "Sistem berjalan dengan baik. Lanjutkan monitoring rutin dan pemeliharaan infrastruktur"
                })
            
        except Exception as e:
            recommendations.append({
                "kategori": "System",
                "prioritas": "Info",
                "rekomendasi": f"Error generating recommendations: {str(e)}"
            })
        
        return recommendations

    def do_GET(self):
        # âš ï¸ AUTHENTICATION CHECK - All GET endpoints require Bearer Token
        if not self._check_auth():
            self._send_auth_error()
            return
        
        # Parse URL dan query parameterers
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        # Endpoint untuk mengecek status job
        if path.startswith('/status/'):
            job_id = path.split('/')[-1]
            if job_id in RESULTS:
                self._set_response()
                # Tambahkan progress, default 0 jika belum ada
                result_data = dict(RESULTS[job_id])
                result_data["progress"] = result_data.get("progress", 0)
                self.wfile.write(json.dumps(result_data).encode('utf-8'))
            else:
                self._set_response(404)
                self.wfile.write(json.dumps({"error": "Job tidak ditemukan"}).encode('utf-8'))

        # Endpoint untuk mendapatkan log output lengkap (semua summary text)
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
                        "error": "Log file tidak ditemukan"
                    }).encode('utf-8'))
            else:
                self._set_response(404)
                self.wfile.write(json.dumps({
                    "success": False,
                    "error": "Job tidak ditemukan"
                }).encode('utf-8'))

        # Endpoint untuk mengecek semua job
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

        # â­ NEW: Endpoint untuk storage info & cleanup
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
                    
                    # Calculate size
                    for dirpath, dirnames, filenames in os.walk(job_dir):
                        for filename in filenames:
                            filepath = os.path.join(dirpath, filename)
                            if os.path.exists(filepath):
                                dir_size += os.path.getsize(filepath)
                    
                    total_size += dir_size
                    
                    # Check age
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

        # â­ NEW: Endpoint untuk manual cleanup
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

        # Endpoint untuk mengambil file hasil
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
                            
                            # Kategorikan berdasarkan tipe file
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
                
                # Tambahkan warning jika ada
                if "warning" in RESULTS[job_id]:
                    response_data["warning"] = RESULTS[job_id]["warning"]
                
                self.wfile.write(json.dumps(response_data).encode('utf-8'))
            else:
                self._set_response(404)
                self.wfile.write(json.dumps({"error": "Hasil tidak ditemukan atau job belum completed"}).encode('utf-8'))

        # Endpoint untuk mendapatkan summary text
        elif path.startswith('/summary/'):
            job_id = path.split('/')[-1]
            if job_id in RESULTS and RESULTS[job_id]["status"] in ["completed", "completed_with_warning"]:
                try:
                    result_dir = get_job_result_path(job_id)
                    
                    print(f"\n{'='*80}")
                    print(f"ENDPOINT /summary/{job_id} - START")
                    print(f"{'='*80}")
                    print(f"Result directory: {result_dir}")
                    
                    # Baca file CSV untuk generate summary (FIXED: Use correct RIVANA filenames)
                    csv_file = os.path.join(result_dir, 'RIVANA_Hasil_Complete.csv')
                    monthly_file = os.path.join(result_dir, 'RIVANA_Monthly_WaterBalance.csv')
                    validation_file = os.path.join(result_dir, 'RIVANA_WaterBalance_Validation.json')
                    
                    print(f"CSV file: {csv_file} (exists: {os.path.exists(csv_file)})")
                    print(f"Monthly file: {monthly_file} (exists: {os.path.exists(monthly_file)})")
                    print(f"Validation file: {validation_file} (exists: {os.path.exists(validation_file)})")
                    
                    summary_text = self.generate_summary_text(csv_file, monthly_file, validation_file, RESULTS[job_id])
                    
                    # Log TWI presence in summary
                    has_twi = 'twi_analysis' in summary_text
                    print(f"\n{'='*80}")
                    print(f"ENDPOINT /summary/{job_id} - SUMMARY GENERATED")
                    print(f"{'='*80}")
                    print(f"Has TWI Analysis: {has_twi}")
                    if has_twi:
                        twi_keys = list(summary_text['twi_analysis'].keys()) if isinstance(summary_text.get('twi_analysis'), dict) else []
                        print(f"TWI Analysis keys: {twi_keys}")
                        if 'twi_enhanced' in summary_text.get('twi_analysis', {}):
                            print(f"TWI Enhanced value: {summary_text['twi_analysis']['twi_enhanced']}")
                        if 'status' in summary_text.get('twi_analysis', {}):
                            print(f"⚠️ WARNING: TWI has 'status' key: {summary_text['twi_analysis']['status']}")
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
                    "error": "Hasil tidak ditemukan atau job belum completed"
                }).encode('utf-8'))

        # Endpoint khusus untuk mendapatkan daftar file PNG saja
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
                                if file_size > 0:  # Hanya file yang tidak kosong
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
                self.wfile.write(json.dumps({"error": "Hasil tidak ditemukan atau job belum completed"}).encode('utf-8'))

        # â­ NEW: Endpoint untuk mendapatkan SEMUA files (PNG, CSV, JSON, dll)
        elif path.startswith('/files/'):
            job_id = path.split('/')[-1]
            if job_id in RESULTS and RESULTS[job_id]["status"] in ["completed", "completed_with_warning"]:
                result_dir = get_job_result_path(job_id)
                all_files = []

                if os.path.exists(result_dir):
                    for file_name in os.listdir(result_dir):
                        # Skip log files dan file sistem
                        if file_name in ['process.log', 'error.log', 'error_outer.log', 'params.json']:
                            continue
                            
                        file_path = os.path.join(result_dir, file_name)
                        if os.path.isfile(file_path):
                            file_size = os.path.getsize(file_path)
                            if file_size > 0:  # Only non-empty files
                                # Detect file type from extension
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
                
                # Sort files by display order and name
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
                self.wfile.write(json.dumps({"error": "Hasil tidak ditemukan atau job belum completed"}).encode('utf-8'))

        # â­ UPDATED: Endpoint untuk preview file (support semua tipe file)

        # NEW: Endpoint khusus untuk River Network Map
        elif path.startswith('/river-map/'):
            job_id = path.split('/')[-1]
            if job_id in RESULTS and RESULTS[job_id]["status"] in ["completed", "completed_with_warning"]:
                result_dir = get_job_result_path(job_id)
                
                # Check for river map files
                river_map_data = {
                    "job_id": job_id,
                    "available": False,
                    "files": {},
                    "metadata": None
                }
                
                # Check for interactive HTML map
                html_map = os.path.join(result_dir, 'interactive_river_map.html')
                if os.path.exists(html_map) and os.path.getsize(html_map) > 0:
                    river_map_data["available"] = True
                    river_map_data["files"]["interactive_html"] = {
                        "name": "interactive_river_map.html",
                        "type": "html",
                        "size": os.path.getsize(html_map),
                        "size_kb": round(os.path.getsize(html_map) / 1024, 2),
                        "download_url": f"/download/{job_id}/interactive_river_map.html",
                        "preview_url": f"/preview/{job_id}/interactive_river_map.html",
                        "description": "Interactive map with zoom, layers, and markers"
                    }
                
                # Check for static PNG map
                png_map = os.path.join(result_dir, 'river_network_map.png')
                if os.path.exists(png_map) and os.path.getsize(png_map) > 0:
                    river_map_data["available"] = True
                    river_map_data["files"]["static_png"] = {
                        "name": "river_network_map.png",
                        "type": "png",
                        "size": os.path.getsize(png_map),
                        "size_kb": round(os.path.getsize(png_map) / 1024, 2),
                        "download_url": f"/download/{job_id}/river_network_map.png",
                        "preview_url": f"/preview/{job_id}/river_network_map.png",
                        "description": "Static map for presentations and reports"
                    }
                
                # Check for metadata JSON
                metadata_file = os.path.join(result_dir, 'river_network_metadata.json')
                if os.path.exists(metadata_file) and os.path.getsize(metadata_file) > 0:
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            river_map_data["metadata"] = json.load(f)
                        
                        river_map_data["files"]["metadata_json"] = {
                            "name": "river_network_metadata.json",
                            "type": "json",
                            "size": os.path.getsize(metadata_file),
                            "size_kb": round(os.path.getsize(metadata_file) / 1024, 2),
                            "download_url": f"/download/{job_id}/river_network_metadata.json",
                            "preview_url": f"/preview/{job_id}/river_network_metadata.json",
                            "description": "River characteristics and statistics"
                        }
                    except Exception as e:
                        print(f"Error loading river map metadata: {e}")
                
                # Add summary
                if river_map_data["available"]:
                    river_map_data["summary"] = {
                        "status": "Available",
                        "files_count": len(river_map_data["files"]),
                        "has_interactive": "interactive_html" in river_map_data["files"],
                        "has_static": "static_png" in river_map_data["files"],
                        "has_metadata": "metadata_json" in river_map_data["files"]
                    }
                    
                    # Add quick access to key metadata
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


        # NEW: Endpoint khusus untuk River Network Map
        elif path.startswith('/river-map/'):
            job_id = path.split('/')[-1]
            if job_id in RESULTS and RESULTS[job_id]["status"] in ["completed", "completed_with_warning"]:
                result_dir = get_job_result_path(job_id)
                
                # Check for river map files
                river_map_data = {
                    "job_id": job_id,
                    "available": False,
                    "files": {},
                    "metadata": None
                }
                
                # Check for interactive HTML map
                html_map = os.path.join(result_dir, 'interactive_river_map.html')
                if os.path.exists(html_map) and os.path.getsize(html_map) > 0:
                    river_map_data["available"] = True
                    river_map_data["files"]["interactive_html"] = {
                        "name": "interactive_river_map.html",
                        "type": "html",
                        "size": os.path.getsize(html_map),
                        "size_kb": round(os.path.getsize(html_map) / 1024, 2),
                        "download_url": f"/download/{job_id}/interactive_river_map.html",
                        "preview_url": f"/preview/{job_id}/interactive_river_map.html",
                        "description": "Interactive map with zoom, layers, and markers"
                    }
                
                # Check for static PNG map
                png_map = os.path.join(result_dir, 'river_network_map.png')
                if os.path.exists(png_map) and os.path.getsize(png_map) > 0:
                    river_map_data["available"] = True
                    river_map_data["files"]["static_png"] = {
                        "name": "river_network_map.png",
                        "type": "png",
                        "size": os.path.getsize(png_map),
                        "size_kb": round(os.path.getsize(png_map) / 1024, 2),
                        "download_url": f"/download/{job_id}/river_network_map.png",
                        "preview_url": f"/preview/{job_id}/river_network_map.png",
                        "description": "Static map for presentations and reports"
                    }
                
                # Check for metadata JSON
                metadata_file = os.path.join(result_dir, 'river_network_metadata.json')
                if os.path.exists(metadata_file) and os.path.getsize(metadata_file) > 0:
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            river_map_data["metadata"] = json.load(f)
                        
                        river_map_data["files"]["metadata_json"] = {
                            "name": "river_network_metadata.json",
                            "type": "json",
                            "size": os.path.getsize(metadata_file),
                            "size_kb": round(os.path.getsize(metadata_file) / 1024, 2),
                            "download_url": f"/download/{job_id}/river_network_metadata.json",
                            "preview_url": f"/preview/{job_id}/river_network_metadata.json",
                            "description": "River characteristics and statistics"
                        }
                    except Exception as e:
                        print(f"Error loading river map metadata: {e}")
                
                # Add summary
                if river_map_data["available"]:
                    river_map_data["summary"] = {
                        "status": "Available",
                        "files_count": len(river_map_data["files"]),
                        "has_interactive": "interactive_html" in river_map_data["files"],
                        "has_static": "static_png" in river_map_data["files"],
                        "has_metadata": "metadata_json" in river_map_data["files"]
                    }
                    
                    # Add quick access to key metadata
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
                            "error": "File kosong atau corrupt",
                            "file_name": file_name
                        }).encode('utf-8'))
                        return
                    
                    # â­ Determine content type based on file extension
                    content_type = 'application/octet-stream'
                    cache_duration = 3600  # 1 hour default
                    
                    if file_name.endswith('.png'):
                        content_type = 'image/png'
                        cache_duration = 86400  # 24 hours for images
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
                        content_type = 'text/csv; charsevapotranspiration=utf-8'
                    elif file_name.endswith('.json'):
                        content_type = 'application/json; charsevapotranspiration=utf-8'
                    elif file_name.endswith('.txt') or file_name.endswith('.log'):
                        content_type = 'text/plain; charsevapotranspiration=utf-8'
                    elif file_name.endswith('.pdf'):
                        content_type = 'application/pdf'
                        cache_duration = 86400
                    elif file_name.endswith('.html'):
                        content_type = 'text/html; charsevapotranspiration=utf-8'
                    elif file_name.endswith('.xml'):
                        content_type = 'application/xml; charsevapotranspiration=utf-8'
                    
                    # â­ Send response headers
                    self.send_response(200)
                    self.send_header('Content-type', content_type)
                    self.send_header('Content-Length', str(file_size))
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('Cache-Control', f'public, max-age={cache_duration}')
                    self.end_headers()

                    # â­ Send file content in chunks (untuk large files)
                    try:
                        with open(file_path, 'rb') as file:
                            chunk_size = 8192  # 8KB chunks
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
                        "error": "File tidak ditemukan",
                        "file_name": file_name,
                        "file_path": file_path
                    }).encode('utf-8'))
            else:
                self._set_response(400)
                self.wfile.write(json.dumps({"error": "URL tidak valid"}).encode('utf-8'))

        # Endpoint untuk download file
        elif path.startswith('/download/'):
            parts = path.split('/')
            if len(parts) >= 4:
                job_id = parts[2]
                file_name = urllib.parse.unquote(parts[3])  # Decode URL encoding
                file_path = os.path.join(get_job_result_path(job_id), file_name)

                if os.path.exists(file_path) and os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)
                    
                    # Validation ukuran file untuk PNG
                    if file_name.endswith('.png') and file_size == 0:
                        self._set_response(500)
                        self.wfile.write(json.dumps({
                            "error": "File PNG kosong atau corrupt",
                            "file_name": file_name,
                            "file_size": 0
                        }).encode('utf-8'))
                        return
                    
                    # Tentukan content type berdasarkan ekstensi file
                    content_type = 'application/octet-stream'
                    if file_name.endswith('.csv'):
                        content_type = 'text/csv'
                    elif file_name.endswith('.json'):
                        content_type = 'application/json'
                    elif file_name.endswith('.png'):
                        content_type = 'image/png'
                    elif file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
                        content_type = 'image/jpeg'

                    self.send_response(200)
                    self.send_header('Content-type', content_type)
                    self.send_header('Content-Length', str(file_size))
                    self.send_header('Content-Disposition', f'attachment; filename="{file_name}"')
                    self.send_header('Access-Control-Allow-Origin', '*')  # CORS
                    self.end_headers()

                    # Baca dan kirim file dalam chunks untuk file besar
                    try:
                        with open(file_path, 'rb') as file:
                            chunk_size = 8192  # 8KB chunks
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
                        "error": "File tidak ditemukan",
                        "file_path": file_path,
                        "exists": os.path.exists(file_path)
                    }).encode('utf-8'))
            else:
                self._set_response(400)
                self.wfile.write(json.dumps({"error": "URL tidak valid"}).encode('utf-8'))

        else:
            self._set_response(404)
            self.wfile.write(json.dumps({"error": "Endpoint tidak ditemukan"}).encode('utf-8'))

    def do_POST(self):
        # âš ï¸ AUTHENTICATION CHECK - All POST endpoints require Bearer Token
        if not self._check_auth():
            self._send_auth_error()
            return
        
        # Hanya menerima request ke endpoint /generate
        if self.path == '/generate':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                params = json.loads(post_data.decode('utf-8'))
                # Validation parameterer yang diperlukan
                required_params = ['longitude', 'latitude', 'start', 'end']
                missing_params = [param for param in required_params if param not in params]
                if missing_params:
                    self._set_response(400)
                    self.wfile.write(json.dumps({
                        "error": f"Parameterer tidak lengkap: {', '.join(missing_params)}"
                    }).encode('utf-8'))
                    return
                # Buat ID job unik
                job_id = str(uuid.uuid4())
                # Simpan informasi job
                RESULTS[job_id] = {
                    "job_id": job_id,
                    "status": "processing",
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "params": params
                }
                # Buat direktori untuk saving hasil
                result_dir = get_job_result_path(job_id)
                os.makedirs(result_dir, exist_ok=True)
                # Jalankan proses di thread terpisah
                thread = threading.Thread(
                    target=run_hidrologi_process,
                    args=(job_id, params, result_dir)
                )
                thread.daemon = True
                thread.start()
                PROCESSES[job_id] = thread
                # Kirim response
                self._set_response()
                self.wfile.write(json.dumps({
                    "job_id": job_id,
                    "status": "processing",
                    "message": "Proses perhitungan hidrologi telah dimulai"
                }).encode('utf-8'))
            except json.JSONDecodeError:
                self._set_response(400)
                self.wfile.write(json.dumps({"error": "Format JSON tidak valid"}).encode('utf-8'))
        else:
            self._set_response(404)
            self.wfile.write(json.dumps({"error": "Endpoint tidak ditemukan"}).encode('utf-8'))

def run_hidrologi_process(job_id, params, result_dir):
    """Fungsi untuk menjalankan proses perhitungan hidrologi"""
    try:
        # Sevapotranspiration progress awal
        RESULTS[job_id]["progress"] = 0
        
        # Simpan parameterer ke file untuk referensi
        with open(os.path.join(result_dir, "params.json"), "w") as f:
            json.dump(params, f)
        
        RESULTS[job_id]["progress"] = 10
        
        # Ambil parameterer
        longitude = params['longitude']
        latitude = params['latitude']
        start = params['start']
        end = params['end']
        
        RESULTS[job_id]["progress"] = 20
        
        # Modifikasi stdout dan stderr untuk menangkap SEMUA output ke file log
        log_file_path = os.path.join(result_dir, "process.log")
        with open(log_file_path, 'w', encoding='utf-8', buffering=1) as log_file:
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            
            # Buat custom writer yang menulis ke file dan flush langsung
            class TeeWriter:
                def __init__(self, file_handle, original_stream):
                    self.file = file_handle
                    self.original_stream = original_stream
                    
                def write(self, data):
                    try:
                        # Write to file
                        self.file.write(data)
                        self.file.flush()  # Flush immediately untuk memastikan data saved
                    except Exception as e:
                        # Fallback to original stream if file write fails
                        if self.original_stream:
                            self.original_stream.write(f"[LOG ERROR: {str(e)}]\n")
                    
                def flush(self):
                    try:
                        self.file.flush()
                    except:
                        pass
                
                def close(self):
                    """Close method untuk kompatibilitas dengan logging shutdown"""
                    try:
                        self.flush()  # Flush sebelum close
                        # Jangan close file karena akan di-manage oleh context manager
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
                print(f"Parameterers: lon={longitude}, lat={latitude}, start={start}, end={end}")
                print(f"{'='*80}\n")
                
                # Import main_weap_ml hanya saat dibutuhkan untuk menghindari dependency error saat server start
                print("Loading main_weap_ml module...")
                sys.stdout.flush()
                
                try:
                    import main_weap_ml
                    # Force reload module to ensure latest code is used (clear cache)
                    import importlib
                    importlib.reload(main_weap_ml)
                    print("âœ“ Module loaded successfully (with force reload)!\n")
                    sys.stdout.flush()
                except Exception as import_error:
                    print(f"âœ— ERROR loading main_weap_ml: {str(import_error)}")
                    sys.stdout.flush()
                    raise
                
                # Update progress before GEE fetching (most common stuck point)
                RESULTS[job_id]["progress"] = 35
                RESULTS[job_id]["message"] = "Fetching satellite data from Google Earth Engine..."
                print("\n" + "="*80)
                print("âš ï¸  CRITICAL SECTION: Google Earth Engine Data Fetching")
                print("   This may take 2-5 minutes depending on date range")
                print("   Progress will update after data is fetched")
                print("="*80 + "\n")
                sys.stdout.flush()
                
                # Run main function - signal timeout removed (doesn't work in threads)
                # Process will be monitored through progress updates instead
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
                    print(f"\nâœ— ERROR in main_weap_ml.main(): {str(main_error)}")
                    print(f"Error type: {type(main_error).__name__}")
                    print("Traceback:")
                    traceback.print_exc()
                    sys.stdout.flush()
                    raise
                
                print(f"\n{'='*80}")
                print(f"main_weap_ml.main() completed successfully")
                print(f"{'='*80}")
                
                # Flush stdout/stderr untuk memastikan semua output saved
                sys.stdout.flush()
                sys.stderr.flush()
                
                RESULTS[job_id]["progress"] = 85
                
                # Tunggu sebentar untuk memastikan semua file completed ditulis ke disk
                print("Waiting for file write complevapotranspirationion...")
                sys.stdout.flush()
                time.sleep(2)  # Tunggu 2 devapotranspirationik
                
                print(f"Checking files in directory: {result_dir}")
                print(f"Directory exists: {os.path.exists(result_dir)}")
                
                RESULTS[job_id]["progress"] = 90
                
                # Expected output files dari main_weap_ml.py (FIXED: menggunakan nama file yang sebenarnya)
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
                        'GEE_Raw_Data.csv'  # ⭐ NEW: Raw GEE data
                    ],
                    'json': [
                        'RIVANA_WaterBalance_Validation.json',
                        'RIVANA_Model_Validation_Complete.json',
                        'RIVANA_Baseline_Comparison.json',
                        'RIVANA_Model_Validation_Report.json',
                        'GEE_Data_Metadata.json'  # ⭐ NEW: GEE metadata
                    ]
                }
                
                # Cek apakah file tergenerate dengan validasi ukuran file
                all_files = os.listdir(result_dir) if os.path.exists(result_dir) else []
                print(f"All files in directory: {all_files}")
                
                png_files = []
                csv_files = []
                json_files = []
                
                # Categorize and validate files
                for file_name in all_files:
                    file_path = os.path.join(result_dir, file_name)
                    if os.path.isfile(file_path):
                        file_size = os.path.getsize(file_path)
                        
                        if file_name.endswith('.png'):
                            if file_size > 0:
                                png_files.append(file_name)
                                print(f"  âœ… PNG: {file_name} ({file_size:,} bytes)")
                            else:
                                print(f"  âš ï¸  WARNING: {file_name} is empty (0 bytes)")
                        elif file_name.endswith('.csv'):
                            if file_size > 0:
                                csv_files.append(file_name)
                                print(f"  âœ… CSV: {file_name} ({file_size:,} bytes)")
                            else:
                                print(f"  âš ï¸  WARNING: {file_name} is empty (0 bytes)")
                        elif file_name.endswith('.json'):
                            if file_size > 0:
                                json_files.append(file_name)
                                print(f"  âœ… JSON: {file_name} ({file_size:,} bytes)")
                            else:
                                print(f"  âš ï¸  WARNING: {file_name} is empty (0 bytes)")
                
                # Check for missing expected files
                print(f"\n{'='*80}")
                print("FILE COMPLETENESS CHECK:")
                print(f"{'='*80}")
                
                missing_png = set(expected_files['png']) - set(png_files)
                missing_csv = set(expected_files['csv']) - set(csv_files)
                missing_json = set(expected_files['json']) - set(json_files)
                
                if missing_png:
                    print(f"âš ï¸  Missing PNG files: {', '.join(missing_png)}")
                if missing_csv:
                    print(f"âš ï¸  Missing CSV files: {', '.join(missing_csv)}")
                if missing_json:
                    print(f"âš ï¸  Missing JSON files: {', '.join(missing_json)}")
                
                print(f"\n{'='*80}")
                print(f"Summary - Files generated:")
                print(f"  PNG: {len(png_files)}/{len(expected_files['png'])} files")
                print(f"  CSV: {len(csv_files)}/{len(expected_files['csv'])} files")
                print(f"  JSON: {len(json_files)}/{len(expected_files['json'])} files")
                print(f"{'='*80}")
                
                RESULTS[job_id]["progress"] = 95
                
                # Validation hasil dengan informasi file yang hilang
                missing_info = []
                if missing_png:
                    missing_info.append(f"PNG: {', '.join(missing_png)}")
                if missing_csv:
                    missing_info.append(f"CSV: {', '.join(missing_csv)}")
                if missing_json:
                    missing_info.append(f"JSON: {', '.join(missing_json)}")
                
                if not png_files and not csv_files:
                    RESULTS[job_id]["status"] = "failed"
                    RESULTS[job_id]["error"] = "Tidak ada file output yang tergenerate"
                    RESULTS[job_id]["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    RESULTS[job_id]["progress"] = 100
                    print("âŒ ERROR: No files generated")
                elif not png_files or missing_info:
                    # Jika ada file hilang atau PNG tidak ada, beri warning
                    RESULTS[job_id]["status"] = "completed_with_warning"
                    RESULTS[job_id]["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    RESULTS[job_id]["result_path"] = result_dir
                    
                    warning_msg = f"Beberapa file tidak tergenerate. "
                    if not png_files:
                        warning_msg += "Tidak ada file PNG. "
                    if missing_info:
                        warning_msg += f"File hilang: {'; '.join(missing_info)}"
                    
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
                    print(f"âš ï¸  WARNING: {warning_msg}")
                else:
                    # Semua file successfully dibuat
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
                    print(f"âœ… SUCCESS: All expected files generated!")
                    print(f"   PNG: {len(png_files)}/{len(expected_files['png'])}")
                    print(f"   CSV: {len(csv_files)}/{len(expected_files['csv'])}")
                    print(f"   JSON: {len(json_files)}/{len(expected_files['json'])}")
                    
            except TimeoutError as te:
                print(f"\n{'='*80}")
                print(f"â±ï¸  TIMEOUT ERROR: Process exceeded maximum time limit")
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
                
                # Save error details
                error_log_path = os.path.join(result_dir, "error.log")
                with open(error_log_path, 'w', encoding='utf-8') as error_file:
                    error_file.write(f"TIMEOUT ERROR\n")
                    error_file.write(f"{'='*80}\n")
                    error_file.write(f"Error: {str(te)}\n\n")
                    error_file.write(f"Timestamp: {datetime.now()}\n")
                    error_file.write(f"Parameterers: {json.dumps(params, indent=2)}\n")
                    traceback.print_exc(file=error_file)
                
                print(f"Error details saved to: {error_log_path}")
                traceback.print_exc()
                sys.stdout.flush()
                sys.stderr.flush()
                
            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__
                
                print(f"\n{'='*80}")
                print(f"âŒ ERROR saat menjalankan proses")
                print(f"{'='*80}\n")
                print(f"Error Type: {error_type}")
                print(f"Error Message: {error_msg}")
                
                # Detect common error patterns
                if "ee.data.authenticateViaPrivateKey" in error_msg or "credentials" in error_msg.lower():
                    print(f"\nðŸ” AUTHENTICATION ERROR DETECTED")
                    print(f"This is likely a Google Earth Engine authentication issue:")
                    print(f"  1. Check if gee-credentials.json exists and is valid")
                    print(f"  2. Verify service account email in .env.production")
                    print(f"  3. Ensure GEE project ID is correct")
                    print(f"  4. Check if service account has Earth Engine API enabled")
                elif "429" in error_msg or "quota" in error_msg.lower():
                    print(f"\nâš ï¸  QUOTA ERROR DETECTED")
                    print(f"Google Earth Engine quota exceeded:")
                    print(f"  1. Wait a few minutes before trying again")
                    print(f"  2. Reduce date range")
                    print(f"  3. Check GEE project quotas")
                elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                    print(f"\nâ±ï¸  TIMEOUT ERROR DETECTED")
                    print(f"Request to Google Earth Engine timed out:")
                    print(f"  1. Try reducing date range")
                    print(f"  2. Check internevapotranspiration connectivity")
                    print(f"  3. Try again later")
                elif "memory" in error_msg.lower() or "memoryerror" in error_type.lower():
                    print(f"\nðŸ’¾ MEMORY ERROR DETECTED")
                    print(f"Insufficient memory to complete operation:")
                    print(f"  1. Reduce date range")
                    print(f"  2. Check server memory availability")
                    print(f"  3. Restart the API service")
                else:
                    print(f"\nâ“ UNKNOWN ERROR")
                    print(f"Please check the full error log for details")
                
                print(f"{'='*80}\n")
                
                # Flush output sebelum error handling
                sys.stdout.flush()
                sys.stderr.flush()
                
                RESULTS[job_id]["status"] = "failed"
                RESULTS[job_id]["error"] = error_msg
                RESULTS[job_id]["error_type"] = error_type
                RESULTS[job_id]["progress"] = 100
                RESULTS[job_id]["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Simpan traceback ke file untuk debugging
                error_log_path = os.path.join(result_dir, "error.log")
                with open(error_log_path, 'w', encoding='utf-8') as error_file:
                    error_file.write(f"ERROR: {error_type}\n")
                    error_file.write(f"{'='*80}\n")
                    error_file.write(f"Message: {error_msg}\n\n")
                    error_file.write(f"Timestamp: {datetime.now()}\n")
                    error_file.write(f"Parameterers: {json.dumps(params, indent=2)}\n\n")
                    error_file.write(f"Full Traceback:\n")
                    error_file.write(f"{'='*80}\n")
                    traceback.print_exc(file=error_file)
                
                print(f"Error details saved to: {error_log_path}")
                traceback.print_exc()  # Print ke log juga
                
                # Flush lagi sevapotranspirationelah error
                sys.stdout.flush()
                sys.stderr.flush()
                
            finally:
                # Final flush sebelum restore stdout/stderr
                try:
                    sys.stdout.flush()
                    sys.stderr.flush()
                except:
                    pass
                
                # Restore stdout/stderr SEBELUM keluar dari context manager
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                
                # Flush log file untuk memastikan semua data saved
                try:
                    log_file.flush()
                except:
                    pass
        
        # Tunggu sebentar lagi setelah file log ditutup untuk memastikan OS flush ke disk
        time.sleep(1)
        
        # Log final message ke console (bukan ke file) - SETELAH sys.stdout sudah di-restore
        # Gunakan original_stdout untuk memastikan tidak ada reference ke file yang sudah tertutup
        try:
            original_stdout.write(f"[INFO] Log file closed and flushed for job {job_id}\n")
            original_stdout.write(f"[INFO] Log file path: {log_file_path}\n")
            original_stdout.write(f"[INFO] Log file exists: {os.path.exists(log_file_path)}\n")
            if os.path.exists(log_file_path):
                original_stdout.write(f"[INFO] Log file size: {os.path.getsize(log_file_path)} bytes\n")
            original_stdout.flush()
        except:
            pass  # Gagal print tidak masalah, yang penting proses utama selesai
                
    except Exception as e:
        RESULTS[job_id]["status"] = "failed"
        RESULTS[job_id]["error"] = str(e)
        RESULTS[job_id]["progress"] = 100
        
        # Simpan traceback untuk outer exception
        try:
            error_log_path = os.path.join(result_dir, "error_outer.log")
            with open(error_log_path, 'w', encoding='utf-8') as error_file:
                traceback.print_exc(file=error_file)
        except:
            pass

def run_server(port=8000, host='127.0.0.1'):
    """Jalankan server HTTP pada port tertentu"""
    handler = HidrologiRequestHandler

    # Buat direktori results jika belum ada
    results_dir = config.RESULTS_DIR if CONFIG_LOADED else "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Cleanup old jobs first (before loading)
    cleanup_old_jobs(max_age_days=30)
    
    # Load existing jobs dari disk
    print("="*80)
    print("ðŸ”„ LOADING EXISTING JOBS FROM DISK")
    print("="*80)
    load_existing_jobs()

    print("="*80)
    print("ðŸŒŠ API SERVER HIDROLOGI ML")
    print("="*80)
    print(f"ðŸ” Mencoba start server di port {port}...")
    print(f"Server akan berjalan di: http://localhost:{port}")
    print("\nï¿½ AUTHENTICATION:")
    print(f"  Bearer Token: {'Enabled' if CONFIG_LOADED else 'Enabled (using default)'}")
    print(f"  Token: {'*' * 20}...{config.API_TOKEN[-4:] if CONFIG_LOADED and len(config.API_TOKEN) > 4 else 'change_this'}")
    print(f"  âš ï¸  All requests MUST include: Authorization: Bearer YOUR_TOKEN")
    print("\nï¿½ðŸ“‹ ENDPOINT TERSEDIA:")
    print(f"  POST   /generate                    - Mulai proses perhitungan baru")
    print(f"  GET    /status/<job_id>             - Cek status job")
    print(f"  GET    /jobs                        - Lihat semua job")
    print(f"  GET    /result/<job_id>             - Daftar file hasil (legacy)")
    print(f"  GET    /summary/<job_id>            - Ringkasan hasil (TEXT)")
    print(f"  GET    /logs/<job_id>               - Log lengkap semua output (FULL SUMMARY)")
    print(f"  GET    /images/<job_id>             - Daftar file PNG/gambar saja")
    print(f"  GET    /files/<job_id>              - â­ Daftar SEMUA files (PNG, CSV, JSON)")
    print(f"  GET    /preview/<job_id>/<file>     - â­ Preview file (PNG, CSV, JSON, TXT, PDF)")
    print(f"  GET    /download/<job_id>/<file>    - Download file hasil")
    print(f"  GET    /storage/info                - ðŸ§¹ Info storage & old jobs (>30 days)")
    print(f"  GET    /storage/cleanup             - ðŸ§¹ Manual cleanup old jobs")
    print("\nðŸ“Š FITUR BARU & PERBAIKAN:")
    print("  âœ… Validasi ukuran file PNG (tidak boleh 0 bytes)")
    print("  âœ… Delay 2 devapotranspirationik sevapotranspirationelah proses untuk flush disk")
    print("  âœ… Logging LENGKAP semua output ke file process.log")
    print("  âœ… Endpoint /logs/<job_id> untuk mendapatkan SEMUA summary text")
    print("  âœ… Error handling yang lebih baik")
    print("  âœ… Capture stdout & stderr dengan flush otomatis")
    print("  â­ Endpoint /files/<job_id> untuk list SEMUA files (PNG, CSV, JSON)")
    print("  â­ Preview support untuk: PNG, JPG, CSV, JSON, TXT, PDF, HTML, XML")
    print("  â­ Auto-sorting files berdasarkan prioritas display")
    print("  â­ File size info (bytes, KB, MB)")
    print("  ðŸ§¹ Auto-cleanup: Job files deleted after 30 days (saves VPS storage)")
    print("\nðŸ†• OUTPUT FILES TERBARU:")
    print("  ðŸ“Š PNG (6 files): Dashboard, Enhanced, WaterBalance, Morphometry,")
    print("                    Morphology-Ecology, Baseline Comparison")
    print("  📄 CSV (4 files): Complete Results, Monthly WaterBalance, Forecast 30 Day,")
    print("                    ⭐ GEE Raw Data (Rainfall, Temp, ET, Soil Moisture, NDVI)")
    print("  ðŸ“‹ JSON (4 files): WaterBalance Validation, Model Validation,")
    print("                     Baseline Comparison, Model Validation Report")
    print("\nâ° FILE RETENTION POLICY:")
    print("  ðŸ“Œ Job results are automatically deleted after 30 days")
    print("  ðŸ“Œ Download important results before expiration")
    print("  ðŸ“Œ Cleanup runs automatically on server startup")
    print("\nðŸ’¡ CARA PENGGUNAAN:")
    print("  1. POST ke /generate dengan JSON: {longitude, latitude, start, end}")
    print("  2. Dapatkan job_id dari response")
    print("  3. Polling /status/<job_id> untuk cek progress")
    print("  4. Sevapotranspirationelah completed:")
    print("     - GET /logs/<job_id> untuk SEMUA output text lengkap")
    print("     - GET /summary/<job_id> untuk summary terstruktur")
    print("     - GET /files/<job_id> untuk SEMUA files (PNG, CSV, JSON)")
    print("     - GET /images/<job_id> untuk daftar gambar saja")
    print("  5. Preview dengan /preview/<job_id>/<filename>")
    print("  6. Download dengan /download/<job_id>/<filename>")
    print("="*80)
    print("\nTekan Ctrl+C untuk menghentikan server...\n")

    # Allow address reuse untuk avoid "Address already in use" error
    socketserver.TCPServer.allow_reuse_address = True
    
    try:
        with socketserver.TCPServer((host, port), handler) as httpd:
            print(f"âœ… Server successfully started di http://{host}:{port}")
            if CONFIG_LOADED:
                print(f"ðŸŒ Environment: {config.environment.upper()}")
                print(f"ðŸ“Š Debug Mode: {config.DEBUG}")
                print(f"ðŸ“ Results Dir: {config.RESULTS_DIR}")
            print(f"ðŸ“¡ Listening for requests...\n")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n\nðŸ›‘ Server dihentikan oleh user")
                pass

            httpd.server_close()
            print("âœ… Server successfully dihentikan")
    except OSError as e:
        # Check for "Address already in use" error (works on both Windows and Linux)
        if e.errno == 98 or (hasattr(e, 'winerror') and e.winerror == 10048):  # Port already in use
            print(f"\nâŒ ERROR: Port {port} sudah digunakan!")
            print(f"\nðŸ’¡ SOLUSI:")
            print(f"   1. Hentikan proses yang menggunakan port {port}")
            if os.name == 'nt':  # Windows
                print(f"      Cek dengan: nevapotranspirationstat -ano | findstr :{port}")
                print(f"      Kill process: taskkill /PID <PID> /F")
            else:  # Linux/Unix
                print(f"      Cek dengan: sudo lsof -i :{port}")
                print(f"      Kill process: sudo kill -9 <PID>")
                print(f"      Atau: sudo pkill -f api_server.py")
            print(f"\n   2. Atau gunakan port lain:")
            print(f"      python api_server.py 8001")
            print(f"\n   3. Atau tunggu beberapa devapotranspirationik dan coba lagi")
        else:
            print(f"\nâŒ ERROR: {str(e)}")
            raise

if __name__ == "__main__":
    # Print configuration if available
    if CONFIG_LOADED:
        config.print_config()
        port = config.API_PORT
        host = config.API_HOST
    else:
        # Fallback to defaults
        port = int(os.gevapotranspirationenv('API_PORT', '8000'))
        host = os.gevapotranspirationenv('API_HOST', '127.0.0.1')
    
    # Command line override
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
            print(f"âœ… Using port from command line: {port}")
        except ValueError:
            print(f"âš ï¸  Port tidak valid: {sys.argv[1]}, menggunakan port {port}")
    
    if len(sys.argv) > 2:
        host = sys.argv[2]
        print(f"âœ… Using host from command line: {host}")
    
    run_server(port, host)


