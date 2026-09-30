param(
    [Parameter(Mandatory=$true)][string]$Url,
    [Parameter(Mandatory=$true)][string]$OutFile,
    [int]$TimeoutSec = 120
)
$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
try {
    Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $OutFile -TimeoutSec $TimeoutSec
} catch {
    [Console]::Error.WriteLine('HTTP transfer failed: ' + $_.Exception.Response.StatusCode + ': ' + $_.Exception.Message)
    exit 1
}
