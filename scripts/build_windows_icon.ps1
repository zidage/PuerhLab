param(
    [string]$InputPng = "$PSScriptRoot/../pu-erh_lab/src/config/ICON/unnamed.png",
    [string]$OutputIco = "$PSScriptRoot/../pu-erh_lab/src/config/ICON/unnamed.ico",
    [int[]]$Sizes = @(16, 20, 24, 32, 40, 48, 64, 128, 256),
    [double]$PaddingRatio = 0.04,
    [int]$AlphaThreshold = 8,
    [int]$MaskAlphaThreshold = 0
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing

function Get-CropRectangle {
    param(
        [System.Drawing.Bitmap]$Bitmap,
        [int]$Threshold
    )

    $width = $Bitmap.Width
    $height = $Bitmap.Height

    $minX = $width
    $minY = $height
    $maxX = -1
    $maxY = -1

    for ($y = 0; $y -lt $height; $y++) {
        for ($x = 0; $x -lt $width; $x++) {
            if ($Bitmap.GetPixel($x, $y).A -gt $Threshold) {
                if ($x -lt $minX) { $minX = $x }
                if ($y -lt $minY) { $minY = $y }
                if ($x -gt $maxX) { $maxX = $x }
                if ($y -gt $maxY) { $maxY = $y }
            }
        }
    }

    if ($maxX -lt 0 -or $maxY -lt 0) {
        return New-Object System.Drawing.Rectangle(0, 0, $width, $height)
    }

    return New-Object System.Drawing.Rectangle($minX, $minY, ($maxX - $minX + 1), ($maxY - $minY + 1))
}

function New-ResizedBitmap {
    param(
        [System.Drawing.Bitmap]$Source,
        [System.Drawing.Rectangle]$CropRect,
        [int]$Size,
        [double]$Padding
    )

    $bmp = New-Object System.Drawing.Bitmap($Size, $Size, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $graphics = [System.Drawing.Graphics]::FromImage($bmp)
    try {
        $graphics.CompositingQuality = [System.Drawing.Drawing2D.CompositingQuality]::HighQuality
        $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
        $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
        $graphics.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
        $graphics.Clear([System.Drawing.Color]::Transparent)

        $inner = [int][Math]::Round($Size * (1.0 - 2.0 * $Padding))
        if ($inner -lt 1) { $inner = $Size }

        $scale = [Math]::Min($inner / [double]$CropRect.Width, $inner / [double]$CropRect.Height)
        $drawWidth = [int][Math]::Round($CropRect.Width * $scale)
        $drawHeight = [int][Math]::Round($CropRect.Height * $scale)
        if ($drawWidth -lt 1) { $drawWidth = 1 }
        if ($drawHeight -lt 1) { $drawHeight = 1 }

        $offsetX = [int][Math]::Floor(($Size - $drawWidth) / 2.0)
        $offsetY = [int][Math]::Floor(($Size - $drawHeight) / 2.0)
        $destination = New-Object System.Drawing.Rectangle($offsetX, $offsetY, $drawWidth, $drawHeight)
        $graphics.DrawImage($Source, $destination, $CropRect, [System.Drawing.GraphicsUnit]::Pixel)
    }
    finally {
        $graphics.Dispose()
    }

    return $bmp
}

function Convert-BitmapToIcoImageData {
    param(
        [System.Drawing.Bitmap]$Bitmap,
        [int]$MaskThreshold
    )

    $width = $Bitmap.Width
    $height = $Bitmap.Height
    $rect = New-Object System.Drawing.Rectangle(0, 0, $width, $height)

    $lockData = $Bitmap.LockBits($rect, [System.Drawing.Imaging.ImageLockMode]::ReadOnly, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    try {
        $stride = [Math]::Abs($lockData.Stride)
        $raw = New-Object byte[] ($stride * $height)
        [System.Runtime.InteropServices.Marshal]::Copy($lockData.Scan0, $raw, 0, $raw.Length)

        $xorStride = $width * 4
        $xorData = New-Object byte[] ($xorStride * $height)
        $andStride = [int]([Math]::Ceiling($width / 32.0) * 4)
        $andData = New-Object byte[] ($andStride * $height)

        for ($y = 0; $y -lt $height; $y++) {
            $srcRow = ($height - 1 - $y) * $stride
            $xorRow = $y * $xorStride
            $andRow = $y * $andStride

            for ($x = 0; $x -lt $width; $x++) {
                $srcIndex = $srcRow + ($x * 4)
                $dstIndex = $xorRow + ($x * 4)

                $xorData[$dstIndex] = $raw[$srcIndex]
                $xorData[$dstIndex + 1] = $raw[$srcIndex + 1]
                $xorData[$dstIndex + 2] = $raw[$srcIndex + 2]
                $xorData[$dstIndex + 3] = $raw[$srcIndex + 3]

                $alpha = $raw[$srcIndex + 3]
                if ($alpha -le $MaskThreshold) {
                    $maskByteIndex = $andRow + [int][Math]::Floor($x / 8.0)
                    $bit = 7 - ($x % 8)
                    $andData[$maskByteIndex] = $andData[$maskByteIndex] -bor ([byte](1 -shl $bit))
                }
            }
        }

        $memory = New-Object System.IO.MemoryStream
        try {
            $writer = New-Object System.IO.BinaryWriter($memory)
            try {
                $biSize = 40
                $biWidth = $width
                $biHeight = $height * 2
                $biPlanes = 1
                $biBitCount = 32
                $biCompression = 0
                $biSizeImage = $xorData.Length + $andData.Length

                $writer.Write([Int32]$biSize)
                $writer.Write([Int32]$biWidth)
                $writer.Write([Int32]$biHeight)
                $writer.Write([UInt16]$biPlanes)
                $writer.Write([UInt16]$biBitCount)
                $writer.Write([UInt32]$biCompression)
                $writer.Write([UInt32]$biSizeImage)
                $writer.Write([Int32]0)
                $writer.Write([Int32]0)
                $writer.Write([UInt32]0)
                $writer.Write([UInt32]0)

                $writer.Write($xorData)
                $writer.Write($andData)
            }
            finally {
                $writer.Dispose()
            }

            return $memory.ToArray()
        }
        finally {
            $memory.Dispose()
        }
    }
    finally {
        $Bitmap.UnlockBits($lockData)
    }
}

$inputPath = (Resolve-Path -LiteralPath $InputPng).Path
if (-not [System.IO.Path]::IsPathRooted($OutputIco)) {
    $OutputIco = [System.IO.Path]::GetFullPath((Join-Path (Get-Location).Path $OutputIco))
}

$outputDir = [System.IO.Path]::GetDirectoryName($OutputIco)
if (-not [string]::IsNullOrWhiteSpace($outputDir)) {
    [System.IO.Directory]::CreateDirectory($outputDir) | Out-Null
}

$source = [System.Drawing.Bitmap]::FromFile($inputPath)
try {
    $cropRect = Get-CropRectangle -Bitmap $source -Threshold $AlphaThreshold
    $entries = New-Object System.Collections.Generic.List[object]

    foreach ($size in $Sizes) {
        $bmp = New-ResizedBitmap -Source $source -CropRect $cropRect -Size $size -Padding $PaddingRatio
        try {
            $bytes = Convert-BitmapToIcoImageData -Bitmap $bmp -MaskThreshold $MaskAlphaThreshold
            $entries.Add([PSCustomObject]@{
                Size = $size
                Data = $bytes
                Length = $bytes.Length
            })
        }
        finally {
            $bmp.Dispose()
        }
    }

    $stream = [System.IO.File]::Open($OutputIco, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write)
    try {
        $writer = New-Object System.IO.BinaryWriter($stream)
        try {
            $writer.Write([UInt16]0)
            $writer.Write([UInt16]1)
            $writer.Write([UInt16]$entries.Count)

            $offset = 6 + (16 * $entries.Count)
            foreach ($entry in $entries) {
                $widthByte = if ($entry.Size -ge 256) { [byte]0 } else { [byte]$entry.Size }
                $heightByte = if ($entry.Size -ge 256) { [byte]0 } else { [byte]$entry.Size }

                $writer.Write($widthByte)
                $writer.Write($heightByte)
                $writer.Write([byte]0)
                $writer.Write([byte]0)
                $writer.Write([UInt16]1)
                $writer.Write([UInt16]32)
                $writer.Write([UInt32]$entry.Length)
                $writer.Write([UInt32]$offset)

                $offset += $entry.Length
            }

            foreach ($entry in $entries) {
                $writer.Write($entry.Data)
            }
        }
        finally {
            $writer.Dispose()
        }
    }
    finally {
        $stream.Dispose()
    }

    Write-Output "Input : $inputPath"
    Write-Output "Output: $OutputIco"
    Write-Output "Crop  : x=$($cropRect.X), y=$($cropRect.Y), w=$($cropRect.Width), h=$($cropRect.Height)"
    Write-Output ("Sizes : " + ($Sizes -join ', '))
    Write-Output "Format: BMP/DIB ICO (Explorer-compatible)"
    Write-Output "Done"
}
finally {
    $source.Dispose()
}
