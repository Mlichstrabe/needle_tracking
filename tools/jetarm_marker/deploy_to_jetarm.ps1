# 将实时 IR 工具同步到 JetArm（SSH scp）
$ErrorActionPreference = "Stop"
$Remote = "ubuntu@192.168.55.1"
$RemoteDir = "/home/ubuntu/jetarm_marker_tools"
$RepoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$LocalDir = Join-Path $RepoRoot "tools\jetarm_marker"

$files = @(
  "ir_stream_server.py",
  "start_live_ir_on_jetarm.sh"
)

ssh $Remote "mkdir -p $RemoteDir"
foreach ($name in $files) {
  scp (Join-Path $LocalDir $name) "${Remote}:${RemoteDir}/"
}
ssh $Remote "sed -i 's/\r$//' ${RemoteDir}/start_live_ir_on_jetarm.sh ${RemoteDir}/ir_stream_server.py; chmod +x ${RemoteDir}/start_live_ir_on_jetarm.sh"
Write-Host "已同步到 ${Remote}:${RemoteDir}"
Write-Host "SSH 启动: bash ${RemoteDir}/start_live_ir_on_jetarm.sh"
