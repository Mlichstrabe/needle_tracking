# 一键启动 hybrid 三窗：JetArm TCP (后台线程) + IMU 串口 (后台线程) + Qt UI
param(
    [string]$HostArm = "192.168.55.1",
    [int]$Port = 8765,
    [string]$ImuPort = "",
    [string]$PoseMode = "hybrid"
)

$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $Root

if (-not $ImuPort) {
    $ports = @()
    try {
        $ports = python -c "import serial.tools.list_ports; print(' '.join(p.device for p in serial.tools.list_ports.comports()))" 2>$null
        $ports = $ports.Trim() -split '\s+' | Where-Object { $_ }
    } catch {}
    if ($ports.Count -ge 1) { $ImuPort = $ports[0] }
    Write-Host "[multitask] 自动 IMU 口: $ImuPort (全部: $($ports -join ', '))"
}

Write-Host @"

[multitask] 并行任务:
  1) TCP $HostArm`:$Port  -> IR+depth 帧队列
  2) $ImuPort -> IMU 四元数
  3) 主线程 -> 三窗 UI (位移 depth / 姿态 IMU)

JetArm 推流未开时请 SSH:
  ssh ubuntu@$HostArm `"bash /home/ubuntu/jetarm_marker_tools/start_live_ir_on_jetarm.sh`"

"@

$args = @(
    "tools/jetarm_marker/live_triple_view.py",
    "--host", $HostArm,
    "--port", "$Port",
    "--pose-mode", $PoseMode
)
if ($ImuPort) { $args += @("--imu-port", $ImuPort) }

python @args