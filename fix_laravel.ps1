# 🔧 Fix Laravel N/A Issues - PowerShell Script
# Run this script to apply fixes to Laravel blade files

$ErrorActionPreference = "Stop"
$laravelPath = "e:\laragon\www\it_river_dna"

Write-Host "====================================" -ForegroundColor Cyan
Write-Host "  FIX LARAVEL N/A ISSUES" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# Check if Laravel path exists
if (-not (Test-Path $laravelPath)) {
    Write-Host "❌ ERROR: Laravel path not found: $laravelPath" -ForegroundColor Red
    Write-Host "Please update the `$laravelPath variable in this script." -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ Laravel path found: $laravelPath" -ForegroundColor Green
Write-Host ""

# ============================================================
# FIX 1: Find translation key error
# ============================================================
Write-Host "[1/3] Searching for translation key error..." -ForegroundColor Yellow

$bladeFiles = Get-ChildItem -Path "$laravelPath\resources\views" -Filter *.blade.php -Recurse
$foundFiles = @()

foreach ($file in $bladeFiles) {
    $content = Get-Content $file.FullName -Raw
    if ($content -match "persentase_capacity:4" -or $content -match "messages\.persentase_capacity:4") {
        $foundFiles += $file
        Write-Host "   ⚠️  Found error in: $($file.FullName)" -ForegroundColor Yellow
    }
}

if ($foundFiles.Count -eq 0) {
    Write-Host "   ℹ️  No translation key error found. Searching for similar patterns..." -ForegroundColor Cyan
    
    # Alternative search
    foreach ($file in $bladeFiles) {
        $content = Get-Content $file.FullName -Raw
        if ($content -match ":4['\"]" -or $content -match "persentase_capacity") {
            $lines = $content -split "`n"
            for ($i = 0; $i -lt $lines.Count; $i++) {
                if ($lines[$i] -match ":4['\"]" -or $lines[$i] -match "persentase_capacity") {
                    Write-Host "   📄 $($file.Name) (line $($i+1)): $($lines[$i].Trim())" -ForegroundColor Cyan
                }
            }
        }
    }
} else {
    Write-Host "   ✅ Found $($foundFiles.Count) file(s) with translation error" -ForegroundColor Green
    Write-Host ""
    Write-Host "   TO FIX: Open the file(s) and replace:" -ForegroundColor Yellow
    Write-Host "   FROM: messages.persentase_capacity:4" -ForegroundColor Red
    Write-Host "   TO:   messages.persentase_capacity" -ForegroundColor Green
    Write-Host ""
}

# ============================================================
# FIX 2: Find show.blade.php for ecosystem health fix
# ============================================================
Write-Host "[2/3] Searching for show.blade.php..." -ForegroundColor Yellow

$showBlade = Get-ChildItem -Path "$laravelPath\resources\views" -Filter "show.blade.php" -Recurse | Where-Object { $_.FullName -like "*hidrologi*" }

if ($showBlade) {
    Write-Host "   ✅ Found: $($showBlade.FullName)" -ForegroundColor Green
    Write-Host ""
    
    # Check for ecosystem health section
    $content = Get-Content $showBlade.FullName -Raw
    if ($content -match "ecosystem_health") {
        Write-Host "   ℹ️  Ecosystem Health section found in blade file" -ForegroundColor Cyan
        Write-Host "   📝 Manual fix required: Add N/A check condition" -ForegroundColor Yellow
        Write-Host "   See: FIX_GUIDE_N_A.md (Fix 2 section)" -ForegroundColor Yellow
    }
} else {
    Write-Host "   ⚠️  show.blade.php not found in hidrologi folder" -ForegroundColor Yellow
    
    # List all show.blade.php files
    $allShowBlades = Get-ChildItem -Path "$laravelPath\resources\views" -Filter "show.blade.php" -Recurse
    if ($allShowBlades) {
        Write-Host "   Found show.blade.php files:" -ForegroundColor Cyan
        foreach ($f in $allShowBlades) {
            Write-Host "   - $($f.FullName)" -ForegroundColor Gray
        }
    }
}

Write-Host ""

# ============================================================
# FIX 3: Check Python API fix applied
# ============================================================
Write-Host "[3/3] Verifying Python API fix..." -ForegroundColor Yellow

$pythonApiPath = "e:\laragon\www\project_hidrologi_ml\project_hidrologi_ml\api_server.py"
if (Test-Path $pythonApiPath) {
    $apiContent = Get-Content $pythonApiPath -Raw
    if ($apiContent -match "slope_str.*Very Flat") {
        Write-Host "   ✅ Python API slope fix already applied!" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️  Python API slope fix NOT found" -ForegroundColor Yellow
        Write-Host "   Please check api_server.py line ~771" -ForegroundColor Yellow
    }
} else {
    Write-Host "   ⚠️  Python API file not found at: $pythonApiPath" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "====================================" -ForegroundColor Cyan
Write-Host "  SUMMARY" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "✅ Fix 1 (Translation Key): " -NoNewline
if ($foundFiles.Count -gt 0) {
    Write-Host "NEEDS FIX - Manual edit required" -ForegroundColor Yellow
} else {
    Write-Host "OK or needs search refinement" -ForegroundColor Green
}

Write-Host "✅ Fix 2 (Ecosystem Health): " -NoNewline
if ($showBlade) {
    Write-Host "NEEDS FIX - See FIX_GUIDE_N_A.md" -ForegroundColor Yellow
} else {
    Write-Host "File not found" -ForegroundColor Yellow
}

Write-Host "✅ Fix 3 (Slope Format): " -NoNewline
if (Test-Path $pythonApiPath) {
    $apiContent = Get-Content $pythonApiPath -Raw
    if ($apiContent -match "slope_str.*Very Flat") {
        Write-Host "ALREADY FIXED" -ForegroundColor Green
    } else {
        Write-Host "NEEDS FIX" -ForegroundColor Yellow
    }
} else {
    Write-Host "File not found" -ForegroundColor Red
}

Write-Host ""
Write-Host "📖 Full fix guide: FIX_GUIDE_N_A.md" -ForegroundColor Cyan
Write-Host ""

# Optional: Open files in VS Code
$openInCode = Read-Host "Open files in VS Code? (y/n)"
if ($openInCode -eq "y") {
    if ($showBlade) {
        code $showBlade.FullName
    }
    foreach ($f in $foundFiles) {
        code $f.FullName
    }
    Write-Host "✅ Files opened in VS Code" -ForegroundColor Green
}

Write-Host ""
Write-Host "Done! 🎉" -ForegroundColor Green
