# API Authentication Guide - Bearer Token

## 🔐 Bearer Token Authentication

Semua endpoint API sekarang dilindungi dengan **Bearer Token authentication**. Setiap request harus menyertakan header `Authorization` dengan token yang valid.

---

## 📋 Setup untuk VPS/Production

### 1️⃣ Generate Secure Token

Gunakan salah satu cara berikut untuk generate secure token:

**Opsi A: Python (recommended)**
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

**Opsi B: OpenSSL**
```bash
openssl rand -base64 32
```

**Opsi C: Online Generator**
- https://randomkeygen.com/ (pilih "CodeIgniter Encryption Keys")

### 2️⃣ Setup di VPS

```bash
# SSH ke VPS
ssh root@103.127.136.132

# Buat file .env.production
cd /var/www/itriverdna.my.id/public_html/py_rivana
nano .env.production
```

Isi file `.env.production`:
```env
ENVIRONMENT=production
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False

# 🔐 IMPORTANT: Ganti dengan token yang Anda generate!
API_TOKEN=YOUR_SECURE_TOKEN_HERE

RESULTS_DIR=/var/www/itriverdna.my.id/public_html/py_rivana/results
TEMP_DIR=/var/www/itriverdna.my.id/public_html/py_rivana/temp
LOG_FILE=/var/log/py-rivana-api.log
```

**Save:** `Ctrl+O`, `Enter`  
**Exit:** `Ctrl+X`

### 3️⃣ Restart API Service

```bash
pkill -f api_server.py
source venv/bin/activate
nohup python project_hidrologi_ml/api_server.py > /var/log/py-rivana-api.log 2>&1 &
```

### 4️⃣ Test Authentication

**Tanpa Token (akan ditolak):**
```bash
curl http://localhost:8000/jobs
# Response: {"error": "Unauthorized", "message": "Valid Bearer Token required..."}
```

**Dengan Token (berhasil):**
```bash
curl -H "Authorization: Bearer YOUR_SECURE_TOKEN_HERE" http://localhost:8000/jobs
# Response: {"jobs": [...]}
```

---

## 💻 Setup untuk Laravel Controller

Update Laravel controller untuk include Bearer Token:

### File: `app/Http/Controllers/HidrologiController.php`

```php
<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;

class HidrologiController extends Controller
{
    private $apiUrl;
    private $apiToken;
    
    public function __construct()
    {
        // API URL dari .env
        $this->apiUrl = env('RIVANA_API_URL', 'http://103.127.136.132:8000');
        
        // 🔐 API Token dari .env
        $this->apiToken = env('RIVANA_API_TOKEN');
    }
    
    /**
     * Generate analisis baru
     */
    public function generate(Request $request)
    {
        $validated = $request->validate([
            'longitude' => 'required|numeric',
            'latitude' => 'required|numeric',
            'start' => 'required|date',
            'end' => 'required|date|after:start',
        ]);
        
        try {
            // POST ke API dengan Bearer Token
            $response = Http::withHeaders([
                'Authorization' => 'Bearer ' . $this->apiToken,
                'Content-Type' => 'application/json',
            ])->post($this->apiUrl . '/generate', $validated);
            
            if ($response->successful()) {
                $data = $response->json();
                return response()->json([
                    'success' => true,
                    'job_id' => $data['job_id'],
                    'message' => 'Analisis dimulai'
                ]);
            }
            
            return response()->json([
                'success' => false,
                'message' => 'API Error: ' . $response->body()
            ], $response->status());
            
        } catch (\Exception $e) {
            return response()->json([
                'success' => false,
                'message' => 'Connection Error: ' . $e->getMessage()
            ], 500);
        }
    }
    
    /**
     * Cek status job
     */
    public function status($jobId)
    {
        try {
            $response = Http::withHeaders([
                'Authorization' => 'Bearer ' . $this->apiToken,
            ])->get($this->apiUrl . '/status/' . $jobId);
            
            if ($response->successful()) {
                return response()->json($response->json());
            }
            
            return response()->json([
                'success' => false,
                'message' => 'Job not found'
            ], 404);
            
        } catch (\Exception $e) {
            return response()->json([
                'success' => false,
                'message' => $e->getMessage()
            ], 500);
        }
    }
    
    /**
     * Get summary hasil
     */
    public function summary($jobId)
    {
        try {
            $response = Http::withHeaders([
                'Authorization' => 'Bearer ' . $this->apiToken,
            ])->get($this->apiUrl . '/summary/' . $jobId);
            
            if ($response->successful()) {
                return response()->json($response->json());
            }
            
            return response()->json([
                'success' => false,
                'message' => 'Summary not found'
            ], 404);
            
        } catch (\Exception $e) {
            return response()->json([
                'success' => false,
                'message' => $e->getMessage()
            ], 500);
        }
    }
    
    /**
     * Get list files
     */
    public function files($jobId)
    {
        try {
            $response = Http::withHeaders([
                'Authorization' => 'Bearer ' . $this->apiToken,
            ])->get($this->apiUrl . '/files/' . $jobId);
            
            if ($response->successful()) {
                return response()->json($response->json());
            }
            
            return response()->json([
                'success' => false,
                'message' => 'Files not found'
            ], 404);
            
        } catch (\Exception $e) {
            return response()->json([
                'success' => false,
                'message' => $e->getMessage()
            ], 500);
        }
    }
    
    /**
     * Download file
     */
    public function download($jobId, $filename)
    {
        try {
            $response = Http::withHeaders([
                'Authorization' => 'Bearer ' . $this->apiToken,
            ])->get($this->apiUrl . '/download/' . $jobId . '/' . $filename);
            
            if ($response->successful()) {
                return response($response->body())
                    ->header('Content-Type', $response->header('Content-Type'))
                    ->header('Content-Disposition', $response->header('Content-Disposition'));
            }
            
            return response()->json([
                'success' => false,
                'message' => 'File not found'
            ], 404);
            
        } catch (\Exception $e) {
            return response()->json([
                'success' => false,
                'message' => $e->getMessage()
            ], 500);
        }
    }
}
```

### File: `.env` (Laravel)

Tambahkan ini ke file `.env` Laravel Anda:

```env
# Rivana ML API Configuration
RIVANA_API_URL=http://103.127.136.132:8000
RIVANA_API_TOKEN=YOUR_SECURE_TOKEN_HERE
```

**⚠️ IMPORTANT:** Token di Laravel harus sama dengan token di VPS!

---

## 🧪 Testing

### Test dari Terminal (curl)

```bash
# Set token sebagai variable
TOKEN="YOUR_SECURE_TOKEN_HERE"

# Test GET /jobs
curl -H "Authorization: Bearer $TOKEN" http://103.127.136.132:8000/jobs

# Test POST /generate
curl -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "longitude": 110.3695,
    "latitude": -7.7956,
    "start": "2024-01-01",
    "end": "2024-01-31"
  }' \
  http://103.127.136.132:8000/generate

# Test GET /status/{job_id}
curl -H "Authorization: Bearer $TOKEN" http://103.127.136.132:8000/status/YOUR_JOB_ID
```

### Test dari JavaScript (Fetch API)

```javascript
const API_URL = 'http://103.127.136.132:8000';
const API_TOKEN = 'YOUR_SECURE_TOKEN_HERE';

// Generate analisis
async function generateAnalysis(params) {
  try {
    const response = await fetch(`${API_URL}/generate`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${API_TOKEN}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(params)
    });
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error:', error);
  }
}

// Usage
generateAnalysis({
  longitude: 110.3695,
  latitude: -7.7956,
  start: '2024-01-01',
  end: '2024-01-31'
}).then(result => console.log(result));
```

---

## 🔒 Security Best Practices

### ✅ DO:
- Generate token yang random dan panjang (minimal 32 karakter)
- Simpan token di environment variable (`.env` file)
- Gunakan HTTPS di production (bukan HTTP)
- Rotate token secara berkala (setiap 3-6 bulan)
- Jangan commit token ke Git

### ❌ DON'T:
- Jangan hardcode token di source code
- Jangan share token di public
- Jangan gunakan token yang lemah (seperti "12345" atau "password")
- Jangan commit file `.env.production` ke Git

---

## 🐛 Troubleshooting

**Error: "Unauthorized"**
```json
{
  "error": "Unauthorized",
  "message": "Valid Bearer Token required. Use: Authorization: Bearer YOUR_TOKEN"
}
```
**Solusi:**
- Pastikan header `Authorization` ada
- Format harus: `Bearer YOUR_TOKEN` (perhatikan spasi setelah "Bearer")
- Token harus sama dengan yang di `.env.production`

**Error: "Connection refused"**
**Solusi:**
- Pastikan API server running: `ps aux | grep api_server`
- Cek port 8000 terbuka: `sudo netstat -tulpn | grep :8000`
- Restart API service

**Laravel tidak bisa connect**
**Solusi:**
- Cek `.env` Laravel apakah `RIVANA_API_URL` dan `RIVANA_API_TOKEN` sudah benar
- Clear config cache: `php artisan config:clear`
- Test dengan curl dulu untuk memastikan API bisa diakses

---

## 📊 Monitoring

Check API logs untuk melihat unauthorized access attempts:

```bash
# View logs
tail -f /var/log/py-rivana-api.log

# Search for auth errors
grep -i "unauthorized" /var/log/py-rivana-api.log
```

---

## 🔄 Rotate Token

Jika token ter-leak, segera ganti:

```bash
# 1. Generate token baru
python -c "import secrets; print(secrets.token_urlsafe(32))"

# 2. Update .env.production di VPS
nano /var/www/itriverdna.my.id/public_html/py_rivana/.env.production

# 3. Update .env di Laravel
nano /path/to/laravel/.env

# 4. Restart API service
pkill -f api_server.py
cd /var/www/itriverdna.my.id/public_html/py_rivana
source venv/bin/activate
nohup python project_hidrologi_ml/api_server.py > /var/log/py-rivana-api.log 2>&1 &

# 5. Clear Laravel config cache
cd /path/to/laravel
php artisan config:clear
```

---

**Token Anda adalah kunci akses ke API. Jaga kerahasiaannya!** 🔐
