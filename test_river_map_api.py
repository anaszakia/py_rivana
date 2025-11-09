"""
Test API Endpoints untuk River Map HTML Files
==============================================

Script ini untuk test apakah file HTML peta aliran sungai
bisa di-fetch dari API server.

Usage:
    python test_river_map_api.py <job_id>
"""

import requests
import json
import sys
import os

# Configuration
API_BASE_URL = "http://localhost:8000"
API_TOKEN = "change_this_to_secure_token"  # Sesuaikan dengan config.py

def test_files_endpoint(job_id):
    """Test endpoint /files/<job_id> untuk cek HTML files"""
    print("\n" + "="*80)
    print(f"🧪 TEST 1: Endpoint /files/{job_id}")
    print("="*80)
    
    url = f"{API_BASE_URL}/files/{job_id}"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Response Status: {response.status_code}")
            print(f"📊 Total Files: {data.get('total_files', 0)}")
            print(f"\n📋 File Summary:")
            summary = data.get('summary', {})
            print(f"  - PNG: {summary.get('png_count', 0)}")
            print(f"  - CSV: {summary.get('csv_count', 0)}")
            print(f"  - JSON: {summary.get('json_count', 0)}")
            print(f"  - HTML: {summary.get('html_count', 0)} {'✅' if summary.get('html_count', 0) > 0 else '❌'}")
            
            # Check HTML files
            html_files = data.get('files_by_type', {}).get('html', [])
            if html_files:
                print(f"\n🗺️ HTML Files Found:")
                for file in html_files:
                    print(f"  ✅ {file['name']}")
                    print(f"     Size: {file['size_kb']:.2f} KB")
                    print(f"     Preview URL: {file['preview_url']}")
                    print(f"     Download URL: {file['download_url']}")
                return True
            else:
                print(f"\n❌ No HTML files found in response")
                print(f"   This might mean:")
                print(f"   1. Job hasn't generated river map yet")
                print(f"   2. HTML file doesn't exist in results folder")
                return False
        else:
            print(f"❌ Error: Status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False

def test_preview_endpoint(job_id, filename="peta_aliran_sungai_interaktif.html"):
    """Test endpoint /preview/<job_id>/<file>"""
    print("\n" + "="*80)
    print(f"🧪 TEST 2: Endpoint /preview/{job_id}/{filename}")
    print("="*80)
    
    url = f"{API_BASE_URL}/preview/{job_id}/{filename}"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    
    try:
        response = requests.get(url, headers=headers, stream=True)
        
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')
            content_length = response.headers.get('Content-Length', 0)
            
            print(f"✅ Response Status: {response.status_code}")
            print(f"📄 Content-Type: {content_type}")
            print(f"📦 Content-Length: {content_length} bytes ({int(content_length)/1024:.2f} KB)")
            
            # Read first 500 chars to verify HTML
            content = response.text[:500]
            
            if 'text/html' in content_type:
                print(f"✅ Content-Type is HTML")
            else:
                print(f"⚠️ Content-Type is not HTML: {content_type}")
            
            if '<html' in content.lower() or '<!doctype html' in content.lower():
                print(f"✅ Content contains HTML markup")
            else:
                print(f"❌ Content doesn't look like HTML")
                
            print(f"\n📄 Preview (first 300 chars):")
            print(f"{content[:300]}...")
            
            return True
        elif response.status_code == 404:
            print(f"❌ File not found: {filename}")
            print(f"   Make sure:")
            print(f"   1. Job ID is correct")
            print(f"   2. Analysis has generated river map")
            print(f"   3. File exists in results/{job_id}/")
            return False
        else:
            print(f"❌ Error: Status {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False

def test_river_map_endpoint(job_id):
    """Test endpoint /river-map/<job_id>"""
    print("\n" + "="*80)
    print(f"🧪 TEST 3: Endpoint /river-map/{job_id}")
    print("="*80)
    
    url = f"{API_BASE_URL}/river-map/{job_id}"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Response Status: {response.status_code}")
            print(f"🗺️ River Map Available: {data.get('available', False)}")
            
            if data.get('available'):
                files = data.get('files', {})
                print(f"\n📋 Available Files:")
                
                if 'interactive_html' in files:
                    html = files['interactive_html']
                    print(f"  ✅ Interactive HTML: {html['name']}")
                    print(f"     Size: {html['size_kb']:.2f} KB")
                    print(f"     Description: {html['description']}")
                
                if 'static_png' in files:
                    png = files['static_png']
                    print(f"  ✅ Static PNG: {png['name']}")
                    print(f"     Size: {png['size_kb']:.2f} KB")
                
                if 'metadata_json' in files:
                    meta = files['metadata_json']
                    print(f"  ✅ Metadata JSON: {meta['name']}")
                
                # Show quick info
                if data.get('quick_info'):
                    info = data['quick_info']
                    print(f"\n📊 Quick Info:")
                    location = info.get('location', {})
                    print(f"  📍 Location: ({location.get('longitude')}, {location.get('latitude')})")
                    print(f"  💧 Mean Flow: {info.get('flow_accumulation_mean', 'N/A')}")
                    print(f"  🌊 Water Occurrence: {info.get('water_occurrence_mean', 'N/A')}")
                
                return True
            else:
                print(f"❌ River map not available for this job")
                print(f"   Message: {data.get('summary', {}).get('message', 'N/A')}")
                return False
        else:
            print(f"❌ Error: Status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False

def check_local_files(job_id):
    """Check if HTML file exists locally"""
    print("\n" + "="*80)
    print(f"🔍 LOCAL FILE CHECK")
    print("="*80)
    
    result_dir = os.path.join("results", job_id)
    html_file = os.path.join(result_dir, "peta_aliran_sungai_interaktif.html")
    png_file = os.path.join(result_dir, "peta_aliran_sungai.png")
    json_file = os.path.join(result_dir, "peta_aliran_sungai_metadata.json")
    
    print(f"📂 Result Directory: {result_dir}")
    
    if os.path.exists(result_dir):
        print(f"✅ Result directory exists")
        
        files = os.listdir(result_dir)
        river_files = [f for f in files if 'peta_aliran_sungai' in f.lower()]
        
        if river_files:
            print(f"\n🗺️ River map files found:")
            for f in river_files:
                fpath = os.path.join(result_dir, f)
                size = os.path.getsize(fpath)
                print(f"  ✅ {f} ({size:,} bytes = {size/1024:.2f} KB)")
        else:
            print(f"\n❌ No river map files found in {result_dir}")
            print(f"   Files present: {len(files)}")
            
    else:
        print(f"❌ Result directory not found: {result_dir}")
        print(f"   Make sure:")
        print(f"   1. Job ID is correct")
        print(f"   2. Running from project_hidrologi_ml root directory")

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_river_map_api.py <job_id>")
        print("\nExample:")
        print("  python test_river_map_api.py 94722be4-b79d-46d1-9fe0-f8514b669309")
        sys.exit(1)
    
    job_id = sys.argv[1]
    
    print("\n" + "="*80)
    print("🧪 RIVER MAP API TESTING")
    print("="*80)
    print(f"Job ID: {job_id}")
    print(f"API URL: {API_BASE_URL}")
    print(f"Token: {'*' * 20}...{API_TOKEN[-4:]}")
    
    # Run tests
    test1 = test_files_endpoint(job_id)
    test2 = test_preview_endpoint(job_id)
    test3 = test_river_map_endpoint(job_id)
    check_local_files(job_id)
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    print(f"  Test 1 - /files endpoint: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"  Test 2 - /preview endpoint: {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"  Test 3 - /river-map endpoint: {'✅ PASS' if test3 else '❌ FAIL'}")
    
    if test1 and test2 and test3:
        print("\n🎉 ALL TESTS PASSED! API is working correctly.")
    else:
        print("\n⚠️ SOME TESTS FAILED. Check output above for details.")
        print("\nPossible issues:")
        print("  1. River map files not generated yet (run analysis first)")
        print("  2. API server not running (start with: python api_server.py)")
        print("  3. Wrong API token (check config.py)")
        print("  4. Job ID doesn't exist")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
