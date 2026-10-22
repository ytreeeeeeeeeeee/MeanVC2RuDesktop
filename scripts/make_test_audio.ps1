param([string]$OutputDirectory = 'reports/test-input')
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$destination = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $destination | Out-Null
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $synth.SetOutputToWaveFile((Join-Path $destination 'reference.wav'))
    $synth.Speak('This is a reference recording for the voice conversion model. Clear speech is important for the target voice.')
    $synth.SetOutputToWaveFile((Join-Path $destination 'source.wav'))
    $synth.Speak('Welcome to Mean V C two desktop. This application converts speech in real time using the original streaming model. We are testing continuous audio on the processor and the graphics card.')
} finally {
    $synth.Dispose()
}
Write-Output $destination
