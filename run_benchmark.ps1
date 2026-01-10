# ═══════════════════════════════════════════════════════════════════════════
# Benchmark Runner Script
# ═══════════════════════════════════════════════════════════════════════════
# Usage: .\run_benchmark.ps1 -Fresh              # Clear cache, run default config
#        .\run_benchmark.ps1 -Config train_te    # Run with TE training (no TE cache)
#        .\run_benchmark.ps1 -Config offload     # Run with TE offloading
#        .\run_benchmark.ps1 -Config workers     # Run with DataLoader workers
#        .\run_benchmark.ps1 -Config large       # Run with simulated large dataset
#        .\run_benchmark.ps1 -Profile            # Enable Python profiling
#
# Available configs:
#   default  - TE cached to disk, TEs frozen, no workers (baseline)
#   train_te - No TE caching, TEs trained with LR 1e-5
#   offload  - TEs offloaded to CPU, no caching
#   workers  - 4 workers, persistent, prefetch=4, more repeats
#   large    - 200 repeats (1000 effective images), tests throughput
#
# Results can be compared against results from the old kohya-ss repo separately.

param(
    [switch]$Fresh,       # Clear cache before running
    [switch]$Profile,     # Enable Python profiling
    [string]$Config = "default",  # Benchmark config variant
    [int]$Runs = 1        # Number of runs for averaging
)

# Map config names to actual config files
$configMap = @{
    "default"  = "benchmark_sdxl"
    "train_te" = "benchmark_sdxl_train_te"
    "offload"  = "benchmark_sdxl_offload"
    "workers"  = "benchmark_sdxl_workers"
    "large"    = "benchmark_sdxl_large"
}

$ErrorActionPreference = "Stop"
$venv = "d:\Projects\sd-scripts\venv\Scripts\python.exe"
$projectRoot = "d:\Projects\sd-scripts"

# Resolve config name
if (-not $configMap.ContainsKey($Config)) {
    Write-Host "[ERROR] Unknown config: $Config" -ForegroundColor Red
    Write-Host "Available: $($configMap.Keys -join ', ')" -ForegroundColor Yellow
    exit 1
}
$configName = $configMap[$Config]
$configFile = "$projectRoot\configs\$configName.yaml"

Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  SDXL PEFT Benchmark (New Data Pipeline)" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Clear cache if requested
if ($Fresh) {
    Write-Host "[INFO] Clearing benchmark cache..." -ForegroundColor Yellow
    $cacheDir = "$projectRoot\benchmark_cache"
    if (Test-Path $cacheDir) {
        Remove-Item -Recurse -Force $cacheDir
        Write-Host "[INFO] Deleted $cacheDir" -ForegroundColor Green
    }
}

# Create output directory
$outputDir = "$projectRoot\benchmark_output"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$script = "scripts/sdxl_peft.py"
Write-Host "[INFO] Config: $Config ($configName)" -ForegroundColor Green
Write-Host "[INFO] Running NEW pipeline (sdxl_peft.py with TrainingDataset)" -ForegroundColor Green

# Collect system info
Write-Host ""
Write-Host "─── System Info ───────────────────────────────────────────────" -ForegroundColor Gray
$gpuInfo = (& nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null).Trim()
Write-Host $gpuInfo
$pythonVersion = ((& $venv --version) 2>&1)
$pytorchVersion = ((& $venv -c 'import torch; print(torch.__version__)') 2>&1)
Write-Host "Python: $pythonVersion"
Write-Host "PyTorch: $pytorchVersion"
Write-Host ""

# Enable resource tracking
$env:BENCHMARK_RESOURCES = "1"

# Pre-run memory snapshot
$memBefore = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>$null).Trim()
Write-Host "[INFO] GPU Memory before: $memBefore" -ForegroundColor Gray

# Run benchmark and capture output
$totalTimes = @()
$allOutput = ""
for ($i = 1; $i -le $Runs; $i++) {
    Write-Host ""
    Write-Host "─── Run $i of $Runs ────────────────────────────────────────────" -ForegroundColor Cyan
    
    $startTime = Get-Date
    $logFile = "$outputDir\benchmark_run${i}.log"
    
    if ($Profile) {
        $profileFile = "$outputDir\profile_run${i}.prof"
        Write-Host "[INFO] Profiling enabled, output: $profileFile"
        & $venv -m cProfile -o $profileFile $script --config-name=$configName 2>&1 | Tee-Object -FilePath $logFile
    } else {
        & $venv $script --config-name=$configName 2>&1 | Tee-Object -FilePath $logFile
    }
    
    $endTime = Get-Date
    $elapsed = ($endTime - $startTime).TotalSeconds
    $totalTimes += $elapsed
    $allOutput = Get-Content $logFile -Raw
    
    Write-Host ""
    Write-Host "[RESULT] Run $i completed in $([math]::Round($elapsed, 2))s" -ForegroundColor Green
}

# Post-run memory snapshot
$memAfter = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>$null).Trim()

# ═══════════════════════════════════════════════════════════════════════════
# Parse benchmark output
# ═══════════════════════════════════════════════════════════════════════════

# Extract final 100% lines from progress bars (get the last complete one)
$latentCachingFinal = [regex]::Matches($allOutput, 'Caching \(GPU 0\): 100%[^\r\n]+') | Select-Object -Last 1
$teCachingFinal = $null
$cachingMatches = [regex]::Matches($allOutput, 'Caching \(GPU 0\): 100%[^\r\n]+')
if ($cachingMatches.Count -ge 2) {
    $teCachingFinal = $cachingMatches[$cachingMatches.Count - 1]
    $latentCachingFinal = $cachingMatches[$cachingMatches.Count - 2]
}
$trainingFinal = [regex]::Matches($allOutput, 'steps: 100%[^\r\n]+avr_loss[^\r\n]+') | Select-Object -Last 1

# Extract speed from final lines
$latentSpeed = if ($latentCachingFinal -and $latentCachingFinal.Value -match '(\d+\.\d+)it/s') { $matches[1] + " it/s" } else { "N/A" }
$teSpeed = if ($teCachingFinal -and $teCachingFinal.Value -match '(\d+\.\d+)it/s') { $matches[1] + " it/s" } else { "N/A" }
$trainSpeed = if ($trainingFinal -and $trainingFinal.Value -match '(\d+\.\d+)s/it') { $matches[1] + " s/it" } else { "N/A" }

# Extract resource tracker summaries - match the full output block
function Extract-ResourceSummary($text, $header) {
    if (-not $text) { return $null }
    # Match from header to next header or end, then extract just the stats
    $pattern = "$header[\s\S]*?Duration:[\s\S]*?CPU RAM:[\s\S]*?peak:[\s\S]*?\)"
    $match = [regex]::Match($text, $pattern)
    if ($match.Success) {
        # Extract just the lines we care about
        $block = $match.Value
        $lines = @()
        if ($block -match 'Duration:\s*([\d.]+s)') { $lines += "Duration: $($matches[1])" }
        
        # GPU Memory
        if ($block -match 'Allocated:\s*(\d+)\s*.*?\s*(\d+)\s*MB\s*\(peak:\s*(\d+)\s*MB\)') {
            $lines += "GPU Allocated: $($matches[1]) → $($matches[2]) MB (peak: $($matches[3]) MB)"
        }
        if ($block -match 'Reserved:\s*(\d+)\s*.*?\s*(\d+)\s*MB\s*\(peak:\s*(\d+)\s*MB\)') {
            $lines += "GPU Reserved: $($matches[1]) → $($matches[2]) MB (peak: $($matches[3]) MB)"
        }
        
        # CPU RAM
        if ($block -match 'Used:\s*(\d+)\s*.*?\s*(\d+)\s*MB\s*\(peak:\s*(\d+)\s*MB\)') {
            $lines += "CPU RAM: $($matches[1]) → $($matches[2]) MB (peak: $($matches[3]) MB)"
        }
        
        if ($lines.Count -gt 0) {
            return $lines -join "`n"
        }
    }
    return $null
}

$latentResource = Extract-ResourceSummary $allOutput "=== Latent Caching ==="
$teResource = Extract-ResourceSummary $allOutput "=== TE Caching ==="
$trainingResource = Extract-ResourceSummary $allOutput "=== Training ==="

# Parse config for key settings - use actual YAML keys
$configSettings = [ordered]@{}
if (Test-Path $configFile) {
    $configContent = Get-Content $configFile -Raw
    
    # Training settings
    if ($configContent -match 'train_batch_size:\s*(\d+)') { $configSettings["training.train_batch_size"] = $matches[1] }
    if ($configContent -match 'gradient_accumulation_steps:\s*(\d+)') { $configSettings["training.gradient_accumulation_steps"] = $matches[1] }
    if ($configContent -match 'max_train_steps:\s*(\d+)') { $configSettings["training.max_train_steps"] = $matches[1] }
    
    # Caching settings (all 4)
    if ($configContent -match 'cache_latents:\s*(true|false)') { $configSettings["data.caching.cache_latents"] = $matches[1] }
    if ($configContent -match 'cache_latents_to_disk:\s*(true|false)') { $configSettings["data.caching.cache_latents_to_disk"] = $matches[1] }
    if ($configContent -match 'cache_text_encoder_outputs:\s*(true|false)') { $configSettings["data.caching.cache_text_encoder_outputs"] = $matches[1] }
    if ($configContent -match 'cache_text_encoder_outputs_to_disk:\s*(true|false)') { $configSettings["data.caching.cache_text_encoder_outputs_to_disk"] = $matches[1] }
    if ($configContent -match 'vae_batch_size:\s*(\d+)') { $configSettings["data.caching.vae_batch_size"] = $matches[1] }
    if ($configContent -match 'num_workers:\s*(\d+)') { $configSettings["data.caching.num_workers"] = $matches[1] }
    
    # Loader settings
    if ($configContent -match 'prefetch_factor:\s*(\d+)') { $configSettings["data.loader.prefetch_factor"] = $matches[1] }
    if ($configContent -match 'persistent_workers:\s*(true|false)') { $configSettings["data.loader.persistent_workers"] = $matches[1] }
    if ($configContent -match 'pin_memory:\s*(true|false)') { $configSettings["data.loader.pin_memory"] = $matches[1] }
    
    # Performance settings
    if ($configContent -match 'gradient_checkpointing:\s*(true|false)') { $configSettings["performance.memory.gradient_checkpointing"] = $matches[1] }
    if ($configContent -match 'offload_text_encoders:\s*(true|false)') { $configSettings["performance.memory.offload_text_encoders"] = $matches[1] }
    if ($configContent -match 'no_half_vae:\s*(true|false)') { $configSettings["performance.precision.no_half_vae"] = $matches[1] }
}

# ═══════════════════════════════════════════════════════════════════════════
# Generate Markdown Report
# ═══════════════════════════════════════════════════════════════════════════

$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$reportFile = "$outputDir\benchmark_report_$timestamp.md"

$md = @"
# Benchmark Report

**Date:** $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
**Config:** $Config ($configName)
**Total Time:** $([math]::Round($totalTimes[0], 2))s

## System

| Key | Value |
|-----|-------|
| GPU | $gpuInfo |
| Python | $pythonVersion |
| PyTorch | $pytorchVersion |

## Configuration

| YAML Key | Value |
|----------|-------|
"@

# Add config settings (ordered, with newline)
foreach ($key in $configSettings.Keys) {
    $md += "`n| ``$key`` | $($configSettings[$key]) |"
}

$md += @"

## Speed Results

| Phase | Final Speed |
|-------|-------------|
| Latent Caching | $latentSpeed |
| TE Caching | $teSpeed |
| Training | $trainSpeed |

## Resource Usage
"@

# Add resource tracker summaries (clean)
if ($latentResource) {
    $md += @"

### Latent Caching

``````
$latentResource
``````
"@
}

if ($teResource) {
    $md += @"

### TE Caching

``````
$teResource
``````
"@
}

if ($trainingResource) {
    $md += @"

### Training

``````
$trainingResource
``````
"@
}

$md += @"

## GPU Memory

- **Before:** $memBefore
- **After:** $memAfter

---
*Generated by run_benchmark.ps1*
"@

# Write report
$md | Out-File -FilePath $reportFile -Encoding utf8

# Console Summary
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  BENCHMARK SUMMARY" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "Config:           $Config ($configName)"
Write-Host "Script:           $script"
Write-Host "Runs:             $Runs"
Write-Host "GPU Mem Before:   $memBefore"
Write-Host "GPU Mem After:    $memAfter"
Write-Host ""

if ($Runs -gt 1) {
    $avg = ($totalTimes | Measure-Object -Average).Average
    $min = ($totalTimes | Measure-Object -Minimum).Minimum
    $max = ($totalTimes | Measure-Object -Maximum).Maximum
    Write-Host "Time (avg):       $([math]::Round($avg, 2))s" -ForegroundColor Green
    Write-Host "Time (min):       $([math]::Round($min, 2))s"
    Write-Host "Time (max):       $([math]::Round($max, 2))s"
} else {
    Write-Host "Total Time:       $([math]::Round($totalTimes[0], 2))s" -ForegroundColor Green
}

if ($Profile) {
    Write-Host ""
    Write-Host "[INFO] To view profile:" -ForegroundColor Yellow
    Write-Host "  & $venv -c `"import pstats; p = pstats.Stats('$outputDir\profile_run1.prof'); p.sort_stats('cumtime').print_stats(30)`""
}

Write-Host ""
Write-Host "[INFO] Report saved: $reportFile" -ForegroundColor Green
Write-Host "[TIP] For accurate peak VRAM tracking, use: nvitop --monitor" -ForegroundColor Gray
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan

