# ═══════════════════════════════════════════════════════════════════════════
# Benchmark Runner Script
# ═══════════════════════════════════════════════════════════════════════════
# Usage: .\run_benchmark.ps1 -Fresh              # Clear cache, run default config
#        .\run_benchmark.ps1 -Config train_te    # Run with TE training (no TE cache)
#        .\run_benchmark.ps1 -Config offload     # Run with TE offloading
#        .\run_benchmark.ps1 -Config workers     # Run with DataLoader workers
#        .\run_benchmark.ps1 -Config large       # Run with simulated large dataset
#        .\run_benchmark.ps1 -Profile            # Enable Python profiling
#        .\run_benchmark.ps1 -Config peft_resource_basic -ResourceMonitorMode off  # baseline without monitor
#
# Available configs:
#   default  - TE cached to disk, TEs frozen, no workers (baseline)
#   train_te - No TE caching, TEs trained with LR 1e-5
#   offload  - TEs offloaded to CPU, no caching
#   workers  - 4 workers, persistent, prefetch=4, more repeats
#   large    - 200 repeats (1000 effective images), tests throughput
#   peft_resource_basic / peft_resource_sampled
#   finetune_resource_basic / finetune_resource_sampled
#
# Results can be compared against results from the old kohya-ss repo separately.

param(
    [switch]$Fresh,       # Clear cache before running
    [switch]$Profile,     # Enable Python profiling
    [string]$Config = "default",  # Benchmark config variant
    [int]$Runs = 1,       # Number of runs for averaging
    [string]$ResourceMonitorMode = ""  # "", off, basic, sampled, deep
)

# Map config names to actual config files
$configMap = @{
    "default"  = "benchmarks/benchmark_sdxl"
    "train_te" = "benchmarks/benchmark_sdxl_train_te"
    "offload"  = "benchmarks/benchmark_sdxl_offload"
    "workers"  = "benchmarks/benchmark_sdxl_workers"
    "large"    = "benchmarks/benchmark_sdxl_large"
    "test_core"         = "tests/test_core"
    "test_finetune"     = "tests/test_finetune"
    "test_checkpoint"   = "tests/test_checkpoint"
    "test_resume"       = "tests/test_resume"
    "test_sampling"     = "tests/test_sampling"
    "test_validation"   = "tests/test_validation"
    "test_text_encoder" = "tests/test_text_encoder"
    "test_memory_optim" = "tests/test_memory_optim"
    "test_advanced"     = "tests/test_advanced"
    "test_logging"      = "tests/test_logging"
    "peft_validation_run" = "tests/test_peft_validation_run"
    "peft_resource_basic" = "tests/test_peft_resource_basic"
    "peft_resource_sampled" = "tests/test_peft_resource_sampled"
    "finetune_resource_basic" = "benchmarks/benchmark_finetune_resource_basic"
    "finetune_resource_sampled" = "benchmarks/benchmark_finetune_resource_sampled"
}

$ErrorActionPreference = "Stop"
$venv = "d:\Projects\sd-scripts\.venv\Scripts\python.exe"
$projectRoot = "d:\Projects\sd-scripts"

# Ensure library/ is importable (package = false in pyproject.toml)
$env:PYTHONPATH = $projectRoot

# Resolve config name
if (-not $configMap.ContainsKey($Config)) {
    Write-Host "[ERROR] Unknown config: $Config" -ForegroundColor Red
    Write-Host "Available: $($configMap.Keys -join ', ')" -ForegroundColor Yellow
    exit 1
}
$configName = $configMap[$Config]
$configFile = Join-Path $projectRoot ("configs\" + ($configName -replace '/', '\') + ".yaml")

Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  SDXL Benchmark (Active Train Launcher)" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Cache directory
$cacheDir = "$projectRoot\benchmark_cache"
$cacheDirsToClear = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
$cacheDirsToClear.Add($cacheDir) | Out-Null

# Also clear config-specific cache_dir when present (e.g., test_* configs)
if (Test-Path $configFile) {
    $configContent = Get-Content $configFile -Raw
    $cacheDirMatch = [regex]::Match($configContent, 'cache_dir:\s*["'']?([^"'']+)["'']?')
    if ($cacheDirMatch.Success) {
        $configuredCacheDir = $cacheDirMatch.Groups[1].Value.Trim()
        if (-not [System.IO.Path]::IsPathRooted($configuredCacheDir)) {
            $configuredCacheDir = Join-Path $projectRoot $configuredCacheDir
        }
        $cacheDirsToClear.Add($configuredCacheDir) | Out-Null
    }
}

# Create output directory
$outputDir = "$projectRoot\benchmark_output"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

# Benchmark runs now always go through the canonical root launcher.
# Keep mode-specific detection only for operator-facing log messages.
$launcher = "train.py"
$finetuneConfigs = @(
    "tests/test_finetune",
    "benchmarks/benchmark_sdxl_finetune",
    "benchmarks/benchmark_finetune_resource_basic",
    "benchmarks/benchmark_finetune_resource_sampled"
)
if ($finetuneConfigs -contains $configName) {
    Write-Host "[INFO] Config: $Config ($configName)" -ForegroundColor Green
    Write-Host "[INFO] Running FINE-TUNE mode through $launcher" -ForegroundColor Green
} else {
    Write-Host "[INFO] Config: $Config ($configName)" -ForegroundColor Green
    Write-Host "[INFO] Running PEFT mode through $launcher" -ForegroundColor Green
}

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

# Resource monitor mode selection:
# - default behavior: sampled for *_resource_sampled configs, basic otherwise
# - explicit override via -ResourceMonitorMode (supports off for baseline runs)
$resourceMonitorMode = if (-not [string]::IsNullOrWhiteSpace($ResourceMonitorMode)) {
    $ResourceMonitorMode.ToLowerInvariant()
} elseif ($configName -like "*resource_sampled") {
    "sampled"
} else {
    "basic"
}

$validResourceModes = @("off", "basic", "sampled", "deep")
if ($resourceMonitorMode -notin $validResourceModes) {
    Write-Host "[ERROR] Invalid ResourceMonitorMode: $resourceMonitorMode" -ForegroundColor Red
    Write-Host "Valid values: $($validResourceModes -join ', ')" -ForegroundColor Yellow
    exit 1
}

$resourceMonitorEnabled = $resourceMonitorMode -ne "off"
$resourceMonitorArgs = @(
    "output.logging.resource_monitor.enabled=$($resourceMonitorEnabled.ToString().ToLowerInvariant())",
    "output.logging.resource_monitor.mode=$resourceMonitorMode",
    "output.logging.resource_monitor.log_every_n_steps=0",
    "output.logging.resource_monitor.rank_scope=main"
)
Write-Host "[INFO] Resource monitor overrides: $($resourceMonitorArgs -join ' ')" -ForegroundColor Gray

# Pre-run memory snapshots
$memBefore = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>$null).Trim()
$cpuRamBefore = [math]::Round((Get-Process -Id $PID).WorkingSet64 / 1MB, 0)

# Run benchmark and capture output
$totalTimes = @()
$firstRunOutput = ""  # First run has fresh caching, use for resource data
$allOutput = ""
for ($i = 1; $i -le $Runs; $i++) {
    # Clear cache before each run if -Fresh is specified
    if ($Fresh) {
        foreach ($dir in $cacheDirsToClear) {
            if (Test-Path $dir) {
                Remove-Item -Recurse -Force $dir
                Write-Host "[INFO] Cleared cache for run ${i}: $dir" -ForegroundColor Yellow
            }
        }
    }
    
    Write-Host ""
    Write-Host "─── Run $i of $Runs ────────────────────────────────────────────" -ForegroundColor Cyan
    
    $startTime = Get-Date
    $logFile = "$outputDir\benchmark_run${i}.log"
    
    if ($Profile) {
        $profileFile = "$outputDir\profile_run${i}.prof"
        Write-Host "[INFO] Profiling enabled, output: $profileFile"
        & $venv -m cProfile -o $profileFile $launcher "--config-name=$configName" @resourceMonitorArgs 2>&1 | Tee-Object -FilePath $logFile
    } else {
        & $venv $launcher "--config-name=$configName" @resourceMonitorArgs 2>&1 | Tee-Object -FilePath $logFile
    }
    $runExitCode = $LASTEXITCODE
    if ($runExitCode -ne 0) {
        Write-Host ""
        Write-Host "[ERROR] Benchmark run $i failed with exit code $runExitCode. See log: $logFile" -ForegroundColor Red
        exit $runExitCode
    }
    
    $endTime = Get-Date
    $elapsed = ($endTime - $startTime).TotalSeconds
    $totalTimes += $elapsed
    $allOutput = Get-Content $logFile -Raw
    
    # Preserve first run output for resource stats (has fresh caching data)
    if ($i -eq 1) {
        $firstRunOutput = $allOutput
    }
    
    Write-Host ""
    Write-Host "[RESULT] Run $i completed in $([math]::Round($elapsed, 2))s" -ForegroundColor Green
}

# Post-run memory snapshots
$memAfter = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>$null).Trim()
$cpuRamAfter = [math]::Round((Get-Process -Id $PID).WorkingSet64 / 1MB, 0)

# Use first run for resource/speed extraction (has fresh caching data)
$allOutput = $firstRunOutput

# ═══════════════════════════════════════════════════════════════════════════
# Parse benchmark output
# ═══════════════════════════════════════════════════════════════════════════

# Extract final 100% lines from progress bars (now with distinct names)
$latentCachingFinal = [regex]::Matches($allOutput, 'Latent Caching \(GPU 0\): 100%[^\r\n]+') | Select-Object -Last 1
$teCachingFinal = [regex]::Matches($allOutput, 'TE Caching \(GPU 0\): 100%[^\r\n]+') | Select-Object -Last 1
$trainingFinal = [regex]::Matches($allOutput, 'steps: 100%[^\r\n]+avr_loss[^\r\n]+') | Select-Object -Last 1

# Extract speed from final lines
$latentSpeed = if ($latentCachingFinal -and $latentCachingFinal.Value -match '(\d+\.\d+)it/s') { $matches[1] + " it/s" } else { "N/A" }
$teSpeed = if ($teCachingFinal -and $teCachingFinal.Value -match '(\d+\.\d+)it/s') { $matches[1] + " it/s" } else { "N/A" }
$trainSpeed = if ($trainingFinal -and $trainingFinal.Value -match '(\d+\.\d+)s/it') { $matches[1] + " s/it" } else { "N/A" }

function Get-CombinedConfigContent {
    param(
        [Parameter(Mandatory = $true)][string]$RootConfigFile,
        [Parameter(Mandatory = $true)][string]$ConfigDir
    )

    if (-not (Test-Path $RootConfigFile)) {
        return ""
    }

    $visited = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    $pending = [System.Collections.Generic.Queue[string]]::new()
    $pending.Enqueue((Resolve-Path $RootConfigFile).Path)

    $combined = ""
    while ($pending.Count -gt 0) {
        $current = $pending.Dequeue()
        if (-not (Test-Path $current)) {
            continue
        }

        $resolved = (Resolve-Path $current).Path
        if ($visited.Contains($resolved)) {
            continue
        }
        $visited.Add($resolved) | Out-Null

        $content = Get-Content $resolved -Raw
        $combined += $content + "`n"

        $defaultsMatches = [regex]::Matches($content, '(?m)^\s*-\s+(/?[A-Za-z0-9_./-]+)\s*$')
        foreach ($dm in $defaultsMatches) {
            $parentName = $dm.Groups[1].Value.TrimStart('/')
            if ($parentName -eq "_self_") { continue }
            $parentFile = Join-Path $ConfigDir (($parentName -replace '/', [IO.Path]::DirectorySeparatorChar) + ".yaml")
            if (Test-Path $parentFile) {
                $pending.Enqueue((Resolve-Path $parentFile).Path)
            }
        }
    }

    return $combined
}

function Is-NewLogRecordLine($line) {
    if ($line -match '^\d{4}-\d{2}-\d{2}\s') { return $true }
    if ($line -match '^\[\d{4}-\d{2}-\d{2}') { return $true }
    if ($line -match '^\s*(DEBUG|INFO|WARNING|ERROR|CRITICAL)\s') { return $true }
    if ($line -match '^\s*steps:\s') { return $true }
    if ($line -match '^\s*Epoch\s+\d+/\d+') { return $true }
    return $false
}

function Is-RichRecordLine($line) {
    if ($line -match '^\d{4}-\d{2}-\d{2}\s') { return $true }
    if ($line -match '^\[\d{4}-\d{2}-\d{2}') { return $true }
    if ($line -match '^\s*(DEBUG|INFO|WARNING|ERROR|CRITICAL)\s+.*\.py:\d+') { return $true }
    return $false
}

function Extract-LogBlockAtIndex($lines, $startIndex, $allowSimpleBullets = $false, $allowSimpleIndented = $false) {
    $firstLine = $lines[$startIndex]
    $isRichRecord = Is-RichRecordLine $firstLine
    $block = @($lines[$startIndex].TrimEnd())

    # Plain/simple formatter: treat non-startup log entries as one-line records
    # unless explicitly allowed for known wrapped blocks.
    if (-not $isRichRecord -and -not $allowSimpleBullets -and -not $allowSimpleIndented) {
        return ($block -join "`n").Trim()
    }

    for ($j = $startIndex + 1; $j -lt $lines.Count; $j++) {
        $line = $lines[$j]
        if ([string]::IsNullOrWhiteSpace($line)) {
            break
        }
        if (-not $isRichRecord -and $allowSimpleBullets) {
            if ($line -notmatch '^\s+-\s') {
                break
            }
            $block += $line.TrimEnd()
            continue
        }
        if (-not $isRichRecord -and $allowSimpleIndented) {
            if ($line -notmatch '^\s+') {
                break
            }
            if (Is-NewLogRecordLine $line) {
                break
            }
            $block += $line.TrimEnd()
            continue
        }
        if (Is-NewLogRecordLine $line) {
            break
        }
        $block += $line.TrimEnd()
    }
    return ($block -join "`n").Trim()
}

function Extract-LatestWrappedLogBlock($text, $anchorPattern, $allowSimpleBullets = $false, $allowSimpleIndented = $false) {
    if (-not $text) { return $null }
    $lines = $text -split "`r?`n"
    for ($i = $lines.Count - 1; $i -ge 0; $i--) {
        if ($lines[$i] -match $anchorPattern) {
            return Extract-LogBlockAtIndex $lines $i $allowSimpleBullets $allowSimpleIndented
        }
    }
    return $null
}

function Extract-AllWrappedLogBlocks($text, $anchorPattern, $allowSimpleBullets = $false, $allowSimpleIndented = $false) {
    if (-not $text) { return $null }
    $lines = $text -split "`r?`n"
    $blocks = @()
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match $anchorPattern) {
            $blocks += (Extract-LogBlockAtIndex $lines $i $allowSimpleBullets $allowSimpleIndented)
        }
    }
    if ($blocks.Count -eq 0) { return $null }
    return $blocks -join "`n`n"
}

$resourceMonitorStart = Extract-LatestWrappedLogBlock $allOutput 'Resource monitor started:'
$resourceStartup = Extract-LatestWrappedLogBlock $allOutput 'Resource startup estimates' $true
$resourceSessionSummary = Extract-LatestWrappedLogBlock $allOutput 'Resource session summary:'
$resourceSessionSummaryLine = if (-not $resourceSessionSummary) {
    [regex]::Matches($allOutput, 'Resource session summary:[^\r\n]+') | Select-Object -Last 1
} else { $null }
if (-not $resourceSessionSummary -and $resourceSessionSummaryLine) {
    $resourceSessionSummary = $resourceSessionSummaryLine.Value.Trim()
}
$latentResource = Extract-LatestWrappedLogBlock $allOutput 'phase\[latent_caching\]:' $false $true
$teResource = Extract-LatestWrappedLogBlock $allOutput 'phase\[te_caching\]:' $false $true
$trainingResource = Extract-AllWrappedLogBlocks $allOutput 'phase\[training_epoch_\d+\]:' $false $true

# Parse config for key settings - use actual YAML keys
# Follow Hydra defaults: chain so inherited settings are picked up
$configSettings = [ordered]@{}
if (Test-Path $configFile) {
    # Build combined content from this config plus all nested defaults.
    # Child content is visited first; first-match regex behavior preserves overrides.
    $combinedContent = Get-CombinedConfigContent -RootConfigFile $configFile -ConfigDir "$projectRoot\configs"
    
    # Training settings
    if ($combinedContent -match 'train_batch_size:\s*(\d+)') { $configSettings["training.train_batch_size"] = $matches[1] }
    if ($combinedContent -match 'gradient_accumulation_steps:\s*(\d+)') { $configSettings["training.gradient_accumulation_steps"] = $matches[1] }
    if ($combinedContent -match 'max_train_steps:\s*(\d+)') { $configSettings["training.max_train_steps"] = $matches[1] }
    
    # Caching settings (all 4)
    if ($combinedContent -match 'cache_latents:\s*(true|false)') { $configSettings["data.caching.cache_latents"] = $matches[1] }
    if ($combinedContent -match 'cache_latents_to_disk:\s*(true|false)') { $configSettings["data.caching.cache_latents_to_disk"] = $matches[1] }
    if ($combinedContent -match 'cache_text_encoder_outputs:\s*(true|false)') { $configSettings["data.caching.cache_text_encoder_outputs"] = $matches[1] }
    if ($combinedContent -match 'cache_text_encoder_outputs_to_disk:\s*(true|false)') { $configSettings["data.caching.cache_text_encoder_outputs_to_disk"] = $matches[1] }
    if ($combinedContent -match 'vae_batch_size:\s*(\d+)') { $configSettings["data.caching.vae_batch_size"] = $matches[1] }
    if ($combinedContent -match 'te_batch_size:\s*(\d+)') { $configSettings["data.caching.te_batch_size"] = $matches[1] }
    if ($combinedContent -match 'num_workers:\s*(\d+)') { $configSettings["data.caching.num_workers"] = $matches[1] }
    
    # Loader settings
    if ($combinedContent -match 'prefetch_factor:\s*(\d+)') { $configSettings["data.loader.prefetch_factor"] = $matches[1] }
    if ($combinedContent -match 'persistent_workers:\s*(true|false)') { $configSettings["data.loader.persistent_workers"] = $matches[1] }
    if ($combinedContent -match 'pin_memory:\s*(true|false)') { $configSettings["data.loader.pin_memory"] = $matches[1] }
    
    # Performance settings
    if ($combinedContent -match 'gradient_checkpointing:\s*(true|false)') { $configSettings["performance.memory.gradient_checkpointing"] = $matches[1] }
    if ($combinedContent -match 'offload_text_encoders:\s*(true|false)') { $configSettings["performance.memory.offload_text_encoders"] = $matches[1] }
    if ($combinedContent -match 'no_half_vae:\s*(true|false)') { $configSettings["performance.precision.no_half_vae"] = $matches[1] }
}

# Applied at runtime by this script (Hydra CLI overrides)
$configSettings["override.output.logging.resource_monitor.enabled"] = $resourceMonitorEnabled.ToString().ToLowerInvariant()
$configSettings["override.output.logging.resource_monitor.mode"] = $resourceMonitorMode
$configSettings["override.output.logging.resource_monitor.log_every_n_steps"] = "0"
$configSettings["override.output.logging.resource_monitor.rank_scope"] = "main"

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

# Add resource monitor summaries (new config-driven system)
if ($resourceMonitorStart) {
    $md += @"

### Monitor Start

``````
$resourceMonitorStart
``````
"@
}

if ($resourceStartup) {
    $md += @"

### Startup Estimates

``````
$resourceStartup
``````
"@
}

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

if ($resourceSessionSummary) {
    $md += @"

### Session Summary

``````
$resourceSessionSummary
``````
"@
}

if (-not $resourceMonitorStart -and -not $resourceStartup -and -not $latentResource -and -not $teResource -and -not $trainingResource -and -not $resourceSessionSummary) {
    $md += @"

_No resource monitor logs were detected in benchmark output._
"@
}

# Add timing stats for multi-run
$timingSection = ""
if ($Runs -gt 1) {
    $avg = [math]::Round(($totalTimes | Measure-Object -Average).Average, 2)
    $min = [math]::Round(($totalTimes | Measure-Object -Minimum).Minimum, 2)
    $max = [math]::Round(($totalTimes | Measure-Object -Maximum).Maximum, 2)
    $timingSection = @"

## Timing ($Runs runs)

| Metric | Value |
|--------|-------|
| Average | ${avg}s |
| Min | ${min}s |
| Max | ${max}s |

> Note: Resource usage data is from Run 1 (fresh caching). Runs 2+ use cached data.
"@
}

$md += @"
$timingSection
## Memory (before → after)

| Type | Before | After |
|------|--------|-------|
| GPU (nvidia-smi) | $memBefore | $memAfter |
| CPU RAM | ${cpuRamBefore} MB | ${cpuRamAfter} MB |

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
