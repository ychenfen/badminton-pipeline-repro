param(
    [Parameter(Mandatory = $true)]
    [string]$InputVideo,
    [string]$WorkRoot = "D:\yumaoqiu_repro",
    [string]$CourtPoints = "352,232,613,232,719,525,244,525",
    [switch]$ManualCourtSelection
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-FileExists {
    param([string]$PathToCheck)
    if (-not (Test-Path -LiteralPath $PathToCheck)) {
        throw "Missing required file: $PathToCheck"
    }
}

$bundleRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$tracknetScript = Join-Path $bundleRoot "scripts\tracknet_runtime\predict.py"
$overlayScript = Join-Path $bundleRoot "scripts\overlay\overlay_player_analytics.py"
$fxScript = Join-Path $bundleRoot "scripts\fx\video_fx_bullet_time.py"
$tracknetWeight = Join-Path $bundleRoot "weights\TrackNet_best.pt"
$yoloPoseWeight = Join-Path $bundleRoot "weights\yolov8s-pose.pt"

Assert-FileExists $InputVideo
Assert-FileExists $tracknetScript
Assert-FileExists $overlayScript
Assert-FileExists $fxScript
Assert-FileExists $tracknetWeight
Assert-FileExists $yoloPoseWeight

New-Item -ItemType Directory -Path $WorkRoot -Force | Out-Null
$tracknetOutDir = Join-Path $WorkRoot "tracknet_v3_result_regen"
New-Item -ItemType Directory -Path $tracknetOutDir -Force | Out-Null

$videoBase = [System.IO.Path]::GetFileNameWithoutExtension($InputVideo)
$tracknetVideoRaw = Join-Path $tracknetOutDir "$videoBase.mp4"
$tracknetVideoCanonical = Join-Path $tracknetOutDir "${videoBase}_tracknetv3.mp4"
$tracknetCsv = Join-Path $tracknetOutDir "${videoBase}_ball.csv"
$overlayOut = Join-Path $WorkRoot "end1_fix_swap2_precision_full_regen.mp4"
$fxOut = Join-Path $WorkRoot "end1_fix_swap2_precision_full_fx_regen.mp4"

Write-Host "[STEP 1/3] TrackNet inference..."
$tracknetArgs = @(
    $tracknetScript,
    "--video_file", $InputVideo,
    "--tracknet_file", $tracknetWeight,
    "--save_dir", $tracknetOutDir,
    "--output_video",
    "--device", "auto",
    "--large_video"
)
& python @tracknetArgs
if ($LASTEXITCODE -ne 0) {
    throw "TrackNet inference failed."
}

Assert-FileExists $tracknetVideoRaw
Assert-FileExists $tracknetCsv
Copy-Item -LiteralPath $tracknetVideoRaw -Destination $tracknetVideoCanonical -Force
Write-Host "[OK] TrackNet outputs:"
Write-Host "  $tracknetVideoCanonical"
Write-Host "  $tracknetCsv"

Write-Host "[STEP 2/3] Player analytics overlay..."
$overlayArgs = @(
    $overlayScript,
    "--video_path", $tracknetVideoCanonical,
    "--output_path", $overlayOut,
    "--ball_csv", $tracknetCsv,
    "--yolo_model", $yoloPoseWeight,
    "--tracker_cfg", "bytetrack.yaml",
    "--detect_interval", "1"
)

if ($ManualCourtSelection) {
    $overlayArgs += "--select_court_points"
} else {
    $overlayArgs += @("--no_select_court_points", "--court_points", $CourtPoints)
}

& python @overlayArgs
if ($LASTEXITCODE -ne 0) {
    throw "Overlay generation failed."
}
Assert-FileExists $overlayOut
Write-Host "[OK] Overlay output: $overlayOut"

Write-Host "[STEP 3/3] Bullet-time FX..."
$fxArgs = @(
    $fxScript,
    "--input", $overlayOut,
    "--output", $fxOut
)
& python @fxArgs
if ($LASTEXITCODE -ne 0) {
    throw "FX generation failed."
}
Assert-FileExists $fxOut

$summary = @(
    $tracknetVideoCanonical,
    $tracknetCsv,
    $overlayOut,
    $fxOut
) | ForEach-Object {
    $it = Get-Item -LiteralPath $_
    [PSCustomObject]@{
        Path = $it.FullName
        SizeMB = [Math]::Round($it.Length / 1MB, 2)
        LastWriteTime = $it.LastWriteTime
    }
}

Write-Host "[DONE] Repro pipeline finished."
$summary | Format-Table -AutoSize

