# 从 JetArm (192.168.55.1) 拷贝 ROS2 bag 到本地 data/jetarm_marker/bags/
# 需要本机已配置 SSH 密钥或会提示输入密码。
# 用法（PowerShell）:
#   .\tools\jetarm_marker\fetch_bags_from_jetarm.ps1

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$Dest = Join-Path $RepoRoot "data\jetarm_marker\bags"
$Remote = "ubuntu@192.168.55.1"
# JetArm 上常见录制路径，若不同请改 -RemotePath
$RemotePath = "/home/ubuntu/ros2_ws/bags"

New-Item -ItemType Directory -Force -Path $Dest | Out-Null

foreach ($name in @("marker_static_clean_01", "marker_move_clean_01")) {
    $local = Join-Path $Dest $name
    if (Test-Path (Join-Path $local "metadata.yaml")) {
        Write-Host "[skip] 已存在: $local"
        continue
    }
    Write-Host "[fetch] $name ..."
    scp -r "${Remote}:${RemotePath}/${name}" $Dest
}

Write-Host "完成。可用 legacy/bag_probe.py 探测 topic，或直接用 detect_ir_markers.py 跑 IR 主路径。"
