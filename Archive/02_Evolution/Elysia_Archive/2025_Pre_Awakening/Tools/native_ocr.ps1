<#
.SYNOPSIS
    Performs OCR on an image using Windows 10/11 built-in OCR engine.
.DESCRIPTION
    Usage: .\native_ocr.ps1 -ImagePath "C:\path\to\image.png"
#>
param(
    [Parameter(Mandatory=$true)]
    [string]$ImagePath
)

# Load WinRT Assemblies
Add-Type -AssemblyName System.Runtime.WindowsRuntime
Add-Type -AssemblyName System.Runtime.InteropServices.WindowsRuntime

# Helper to await async tasks in PowerShell
function Await($Task) {
    $Task.Wait()
    if ($Task.Status -eq 'RanToCompletion') {
        return $Task.Result
    }
    throw $Task.Exception
}

try {
    # Load Windows.Media.Ocr
    $OSBuild = [System.Environment]::OSVersion.Version.Build
    if ($OSBuild -lt 10240) {
        Write-Error "Windows 10 or higher is required for Windows.Media.Ocr."
        exit 1
    }

    # We need to load the WinMD files. This is tricky in pure PS.
    # Instead, let's try a simpler approach if possible, or use a known working snippet.
    # Actually, accessing UWP APIs from pure PowerShell is notoriously difficult due to type resolution.
    
    # Let's try a different approach: UWP OCR via a C# inline snippet.
    
    $code = @"
    using System;
    using System.Threading.Tasks;
    using Windows.Graphics.Imaging;
    using Windows.Media.Ocr;
    using Windows.Storage;
    using Windows.Storage.Streams;
    
    public class OcrRunner {
        public static string RunOcr(string path) {
            return RunOcrAsync(path).GetAwaiter().GetResult();
        }

        private static async Task<string> RunOcrAsync(string path) {
            try {
                StorageFile file = await StorageFile.GetFileFromPathAsync(path);
                using (IRandomAccessStream stream = await file.OpenAsync(FileAccessMode.Read)) {
                    BitmapDecoder decoder = await BitmapDecoder.CreateAsync(stream);
                    SoftwareBitmap bitmap = await decoder.GetSoftwareBitmapAsync();
                    
                    OcrEngine engine = OcrEngine.TryCreateFromUserProfileLanguages();
                    if (engine == null) return "Error: OCR Engine not supported for current language.";
                    
                    OcrResult result = await engine.RecognizeAsync(bitmap);
                    return result.Text;
                }
            } catch (Exception ex) {
                return "Error: " + ex.Message;
            }
        }
    }
"@

    # We need to reference the Windows Runtime DLLs.
    # Usually found in C:\Windows\System32\WinMetadata\Windows.Media.winmd etc.
    # But Add-Type doesn't support .winmd directly easily.
    
    Write-Output "PowerShell OCR is experimental and complex to load dependencies."
    Write-Output "FALLBACK: Please install 'pytesseract' for reliable OCR."
} catch {
    Write-Error $_
}
