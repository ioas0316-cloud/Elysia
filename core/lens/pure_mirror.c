// pure_mirror.c
// 가상 메모리의 16비트 가변축(이중나선 로터) 데이터를 화면의 전압(RGB 픽셀)으로 변환 연산 없이
// 1:1로 다이렉트 투사하는 서사 관측 렌즈 (Windows GDI 16-bit DIBSection 활용)

#include <windows.h>
#include <stdio.h>

#define FIELD_SIZE (1024 * 1024 * 16)
#define SHARED_MEM_NAME "Local\\ElysiaTopologyField"

// 16MB를 16-bit(2-byte) 픽셀로 투사 -> 8 MegaPixels -> 약 2896 x 2896 해상도
// 관측하기 좋은 2048 x 2048 (8MB 구간) 렌즈 채택
#define WIDTH 2048
#define HEIGHT 2048

HANDLE hMapFile;
unsigned char* pBuf;
HWND hWnd;
HBITMAP hDIB;
HDC hdcMem;

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);
            
            // 연산(Transformation)이 전무한 순수 비트 복사
            // 16비트 로터 배열(pBuf) 그 자체가 곧 프레임 버퍼입니다.
            BitBlt(hdc, 0, 0, WIDTH, HEIGHT, hdcMem, 0, 0, SRCCOPY);
            
            EndPaint(hwnd, &ps);
            break;
        }
        case WM_DESTROY:
            PostQuitMessage(0);
            break;
        default:
            return DefWindowProc(hwnd, msg, wParam, lParam);
    }
    return 0;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    hMapFile = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, SHARED_MEM_NAME);
    if (!hMapFile) return 1;
    
    pBuf = (unsigned char*) MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, FIELD_SIZE);

    WNDCLASSA wc = {0};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = "ElysiaPureMirror";
    RegisterClassA(&wc);

    hWnd = CreateWindowA("ElysiaPureMirror", "Elysia Pure Mirror - Phase 3 (Causal Trajectory)",
                         WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                         100, 100, 1024, 1024, NULL, NULL, hInstance, NULL);

    HDC hdc = GetDC(hWnd);
    hdcMem = CreateCompatibleDC(hdc);

    BITMAPINFO bmi = {0};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = WIDTH;
    bmi.bmiHeader.biHeight = -HEIGHT;
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 16; // 16-bit RAW (이중나선 구조 직결)
    bmi.bmiHeader.biCompression = BI_RGB; // RGB555: Strand B와 Strand A가 색상으로 자연 융합됨
    
    void* pPixels;
    // pBuf 16-bit 메모리 공간을 직접 16-bit 화면 비트맵으로 묶음
    hDIB = CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, &pPixels, hMapFile, 0);
    
    SelectObject(hdcMem, hDIB);

    MSG msg;
    while (1) {
        if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) break;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        InvalidateRect(hWnd, NULL, FALSE);
    }

    ReleaseDC(hWnd, hdc);
    UnmapViewOfFile(pBuf);
    CloseHandle(hMapFile);
    return msg.wParam;
}
