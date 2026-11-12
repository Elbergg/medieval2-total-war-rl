#include <windows.h>
#include <stdio.h>

/**
 * Headless M2TWEOP Launcher for Medieval II: Total War (Steam 1.52)
 *
 * Emulates M2TWEOP GUI.exe startup handshake:
 * 1. Creates Win32 named shared memory "M2TWEOPStartMem1" (4096 bytes).
 * 2. Formats exact 5-token payload (no trailing newline):
 *    "m2tweopStartCommand\neopModFolder:\n<folder>\nGameVer:\n2"
 * 3. Launches "medieval2.exe @medieval2.preference.cfg".
 * 4. Waits for process termination and cleans up handles.
 */

int main(int argc, char* argv[]) {
    const char* modFolder = ".";
    if (argc > 1) {
        modFolder = argv[1];
    }
    
    HANDLE hMap = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, 4096, "M2TWEOPStartMem1");
    if (!hMap) {
        printf("Failed to create file mapping: %lu\n", GetLastError());
        return 1;
    }
    char* pBuf = (char*)MapViewOfFile(hMap, FILE_MAP_ALL_ACCESS, 0, 0, 4096);
    if (!pBuf) {
        printf("Failed to map view of file: %lu\n", GetLastError());
        CloseHandle(hMap);
        return 1;
    }
    
    // EXACT FORMAT: 5 lines, NO trailing newline!
    sprintf(pBuf, "m2tweopStartCommand\neopModFolder:\n%s\nGameVer:\n2", modFolder);
    printf("Shared memory M2TWEOPStartMem1 initialized with:\n---\n%s\n---\n", pBuf);
    
    STARTUPINFOA si = { sizeof(si) };
    PROCESS_INFORMATION pi = { 0 };
    char cmdLine[] = "medieval2.exe @medieval2.preference.cfg";
    
    printf("Launching: %s\n", cmdLine);
    if (!CreateProcessA(NULL, cmdLine, NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi)) {
        printf("Failed to launch medieval2.exe: %lu\n", GetLastError());
        UnmapViewOfFile(pBuf);
        CloseHandle(hMap);
        return 1;
    }
    
    printf("medieval2.exe launched! PID = %lu. Waiting for process...\n", pi.dwProcessId);
    WaitForSingleObject(pi.hProcess, INFINITE);
    
    UnmapViewOfFile(pBuf);
    CloseHandle(hMap);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return 0;
}
