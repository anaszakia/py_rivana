# Google Earth Engine Authentication Setup

## 🔐 Setup untuk VPS/Production

### 1️⃣ Buat Service Account di Google Cloud

1. Buka [Google Cloud Console](https://console.cloud.google.com/)
2. Pilih project: **fabled-era-474402-g2**
3. Pergi ke **IAM & Admin** → **Service Accounts**
4. Klik **Create Service Account**
5. Isi:
   - **Name**: `earth-engine-api`
   - **Description**: `Service account for Earth Engine API access`
6. Klik **Create and Continue**
7. Grant role: **Earth Engine Resource Admin**
8. Klik **Continue** → **Done**

### 2️⃣ Generate Key File

1. Klik service account yang baru dibuat
2. Tab **Keys** → **Add Key** → **Create New Key**
3. Pilih format: **JSON**
4. Download file JSON (akan otomatis terdownload)

### 3️⃣ Upload ke VPS

**Di local komputer:**
```bash
# Upload credentials ke VPS
scp path/to/downloaded-key.json root@103.127.136.132:/var/www/itriverdna.my.id/public_html/py_rivana/project_hidrologi_ml/gee-credentials.json
```

**Atau copy-paste manual:**
```bash
# SSH ke VPS
ssh root@103.127.136.132

# Buat file
nano /var/www/itriverdna.my.id/public_html/py_rivana/project_hidrologi_ml/gee-credentials.json

# Paste isi JSON yang didownload tadi
# Save: Ctrl+O, Enter, Exit: Ctrl+X

# Set permissions
chmod 600 /var/www/itriverdna.my.id/public_html/py_rivana/project_hidrologi_ml/gee-credentials.json
```

### 4️⃣ Enable Earth Engine API

1. Buka [Google Cloud Console](https://console.cloud.google.com/)
2. Pergi ke **APIs & Services** → **Library**
3. Search: **Earth Engine API**
4. Klik **Enable**

### 5️⃣ Test Authentication

```bash
cd /var/www/itriverdna.my.id/public_html/py_rivana
source venv/bin/activate
python -c "import ee; ee.Initialize(project='fabled-era-474402-g2'); print('✅ GEE Connected!')"
```

---

## 🖥️ Setup untuk Local Development

Untuk development di local (Windows/Mac), kode akan otomatis fallback ke default authentication.

**First time setup:**
```bash
# Install earthengine-api
pip install earthengine-api

# Authenticate (akan buka browser)
earthengine authenticate

# Test
python -c "import ee; ee.Initialize(project='fabled-era-474402-g2'); print('✅ GEE Connected!')"
```

---

## 📁 File Structure

```
project_hidrologi_ml/
├── main_weap_ml.py                    # Main script (updated with service account support)
├── gee-credentials.json               # Service account key (DO NOT COMMIT!)
└── gee-credentials.json.example       # Template file
```

---

## 🔒 Security Notes

⚠️ **IMPORTANT**: 
- **NEVER** commit `gee-credentials.json` to Git
- Add to `.gitignore`: `gee-credentials.json`
- Keep credentials file secure with proper permissions (chmod 600)
- Rotate keys regularly for security

---

## 🐛 Troubleshooting

**Error: "gcloud command not found"**
- ✅ Fixed! Now using service account authentication (no gcloud needed)

**Error: "Could not find default credentials"**
- Make sure `gee-credentials.json` exists in `project_hidrologi_ml/` directory
- Check file permissions: `ls -la gee-credentials.json`

**Error: "Earth Engine API has not been enabled"**
- Enable Earth Engine API in Google Cloud Console (see step 4 above)

**Error: "Permission denied"**
- Make sure service account has **Earth Engine Resource Admin** role
- Check project ID is correct: `fabled-era-474402-g2`
