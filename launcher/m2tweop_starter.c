#include <windows.h>
#include <stdio.h>

/**
 * Headless M2TWEOP Launcher for Medieval II: Total War (Steam 1.52)
 *
 * Emulates M2TWEOP GUI.exe startup handshake:
 * 1. Creates Win32 named shared memory "M2TWEOPStartMem1" (4096 bytes).
 * 2. Emulates exact M2TWEOP payload format:
 *    "m2tweopStartCommand\neopModFolder:\n<folder>\nGameVer:\n2"
 * 3. Launches "medieval2.exe @medieval2.preference.cfg" suspended (CREATE_SUSPENDED).
 * 4. Applies in-memory hotpatches:
 *    - Patch 1 (0x00413320, 13 bytes):
 *      Initializes stack error byte 0x1f(%esp) = 0 and jumps directly to 0x4133b6.
 *      Fixes the uninitialized stack byte bug when audio is disabled and bypasses all audio dialogs.
 *    - Patch 2 (0x004153ff, 5 bytes):
 *      Bypasses WinMain modal crash dialog (cmp $1, %esi -> jmp 0x415466).
 * 5. Resumes medieval2.exe main thread.
 */

static void apply_patch(HANDLE hProcess, DWORD addr, const unsigned char* bytes, size_t size) {
    DWORD oldProtect;
    VirtualProtectEx(hProcess, (LPVOID)addr, size, PAGE_EXECUTE_READWRITE, &oldProtect);
    WriteProcessMemory(hProcess, (LPVOID)addr, bytes, size, NULL);
    VirtualProtectEx(hProcess, (LPVOID)addr, size, oldProtect, &oldProtect);
}

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
    
    sprintf(pBuf, "m2tweopStartCommand\neopModFolder:\n%s\nGameVer:\n2", modFolder);
    printf("Shared memory M2TWEOPStartMem1 initialized with:\n---\n%s\n---\n", pBuf);
    
    STARTUPINFOA si = { sizeof(si) };
    PROCESS_INFORMATION pi = { 0 };
    char cmdLine[] = "medieval2.exe @medieval2.preference.cfg";
    
    printf("Launching suspended: %s\n", cmdLine);
    if (!CreateProcessA(NULL, cmdLine, NULL, NULL, FALSE, CREATE_SUSPENDED, NULL, NULL, &si, &pi)) {
        printf("Failed to launch medieval2.exe: %lu\n", GetLastError());
        UnmapViewOfFile(pBuf);
        CloseHandle(hMap);
        return 1;
    }
    
    // Patch 1: Clear error flag on stack and jump directly to 0x4133b6 when audio disabled (0x00413320, 13 bytes)
    unsigned char patch1[13] = {
        0xc6, 0x44, 0x24, 0x1f, 0x00,  // movb $0, 0x1f(%esp)
        0xe9, 0x8c, 0x00, 0x00, 0x00,  // jmp 0x004133b6
        0x90, 0x90, 0x90               // nop nop nop
    };
    apply_patch(pi.hProcess, 0x00413320, patch1, sizeof(patch1));

    // Patch 2: Bypass audio failure modal dialog at 0x004133a2 (20 bytes)
    unsigned char patch2[20] = {
        0xc6, 0x05, 0x76, 0x28, 0x6a, 0x01, 0x01, // movb $1, 0x16a2876 (audio.disabled = 1)
        0xeb, 0x0b,                               // jmp 0x004133b6 (+11)
        0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90 // 11 nops
    };
    apply_patch(pi.hProcess, 0x004133a2, patch2, sizeof(patch2));

    // Patch 3: NOP out abort jump at 0x004133d6 (6 bytes: je 0x414e34 -> 6 nops)
    unsigned char patch3[6] = { 0x90, 0x90, 0x90, 0x90, 0x90, 0x90 };
    apply_patch(pi.hProcess, 0x004133d6, patch3, sizeof(patch3));

    // Patch 4: Force clean exit on shutdown at 0x00414e8e (11 bytes: jmp 0x00412f56)
    unsigned char patch4[11] = {
        0xe9, 0xc3, 0xe0, 0xff, 0xff,  // jmp 0x00412f56
        0x90, 0x90, 0x90, 0x90, 0x90, 0x90 // 6 nops
    };
    apply_patch(pi.hProcess, 0x00414e8e, patch4, sizeof(patch4));

    printf("Headless audio, startup-abort bypass & shutdown hotpatches applied cleanly. Resuming PID=%lu...\n", pi.dwProcessId);
    ResumeThread(pi.hThread);
    
    WaitForSingleObject(pi.hProcess, INFINITE);
    
    DWORD exitCode = 0;
    GetExitCodeProcess(pi.hProcess, &exitCode);
    printf("medieval2.exe exited with code: %lu\n", exitCode);

    UnmapViewOfFile(pBuf);
    CloseHandle(hMap);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return (int)exitCode;
}
