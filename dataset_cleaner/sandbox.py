"""Fail-closed Windows AppContainer runner for syntax-only compiler checks.

Dataset examples are written only to the AppContainer's private local-data
folder. The child receives no network capabilities, no inherited handles other
than standard streams, and runs in a memory/time/process-limited Job Object.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

MAX_SOURCE_CHARS = 1_000_000
MAX_OUTPUT_BYTES = 64 * 1024
PROCESS_MEMORY_LIMIT = 256 * 1024 * 1024
PROCESS_TIME_LIMIT_100NS = 5 * 10_000_000
SANDBOX_PROFILE = "DataSetCleanerSyntaxValidator"
PYTHON_AST_CHECKER = """import ast, json, sys, warnings
source_path = sys.argv[1]
result = {"syntax_error": None, "parser_error": None, "warnings": []}
try:
    with open(source_path, "r", encoding="utf-8") as source_file:
        source_text = source_file.read()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SyntaxWarning)
        try:
            ast.parse(source_text, filename="<dataset>")
        except SyntaxError as error:
            result["syntax_error"] = error.msg
        except Exception as error:
            result["parser_error"] = type(error).__name__ + ": " + str(error)[:400]
        result["warnings"] = [str(item.message)[:400] for item in caught]
except Exception as error:
    result["parser_error"] = type(error).__name__ + ": " + str(error)[:400]
sys.stdout.write(json.dumps(result, ensure_ascii=True))
"""


class SandboxUnavailable(RuntimeError):
    """Raised when the OS sandbox cannot be established; callers must fail closed."""


class SandboxTimeout(RuntimeError):
    """Raised when a syntax validator exceeds its wall-clock limit."""


class SandboxOutputLimit(RuntimeError):
    """Raised when a validator exceeds the captured diagnostic output limit."""


class SandboxInputLimit(RuntimeError):
    """Raised when source text exceeds the bounded syntax-check input size."""


@dataclass(frozen=True)
class SandboxedResult:
    returncode: int
    stdout: str
    stderr: str


@lru_cache(maxsize=1)
def _win_api() -> tuple[ctypes.WinDLL, ctypes.WinDLL, ctypes.WinDLL]:
    if os.name != "nt":
        raise SandboxUnavailable("External syntax validators are disabled outside the Windows AppContainer backend.")
    return (
        ctypes.WinDLL("kernel32", use_last_error=True),
        ctypes.WinDLL("userenv", use_last_error=True),
        ctypes.WinDLL("advapi32", use_last_error=True),
    )


def _raise_last_error(action: str) -> None:
    code = ctypes.get_last_error()
    raise SandboxUnavailable(f"Windows AppContainer sandbox setup failed during {action} (WinError {code}).")


@lru_cache(maxsize=1)
def _container_sid(kernel32: ctypes.WinDLL, userenv: ctypes.WinDLL, advapi32: ctypes.WinDLL) -> tuple[ctypes.c_void_p, str]:
    sid = ctypes.c_void_p()
    create = userenv.CreateAppContainerProfile
    create.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
    create.restype = ctypes.c_long
    hr = create(SANDBOX_PROFILE, "DataSetCleaner syntax validators", "Isolated parser-only compiler process", None, 0, ctypes.byref(sid))
    if ctypes.c_uint32(hr).value == 0x800700B7:  # HRESULT_FROM_WIN32(ERROR_ALREADY_EXISTS)
        derive = userenv.DeriveAppContainerSidFromAppContainerName
        derive.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_void_p)]
        derive.restype = ctypes.c_long
        hr = derive(SANDBOX_PROFILE, ctypes.byref(sid))
    if hr < 0 or not sid.value:
        raise SandboxUnavailable(f"Windows could not create the syntax-validator AppContainer (HRESULT 0x{ctypes.c_uint32(hr).value:08x}).")

    sid_text_ptr = ctypes.c_wchar_p()
    convert = advapi32.ConvertSidToStringSidW
    convert.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_wchar_p)]
    convert.restype = ctypes.c_int
    if not convert(sid, ctypes.byref(sid_text_ptr)):
        _raise_last_error("converting its AppContainer identity")
    try:
        sid_text = sid_text_ptr.value
    finally:
        kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        kernel32.LocalFree.restype = ctypes.c_void_p
        kernel32.LocalFree(ctypes.cast(sid_text_ptr, ctypes.c_void_p))
    return sid, sid_text


def _appcontainer_data_folder(userenv: ctypes.WinDLL, sid_text: str) -> Path:
    path_ptr = ctypes.c_wchar_p()
    get_path = userenv.GetAppContainerFolderPath
    get_path.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_wchar_p)]
    get_path.restype = ctypes.c_long
    hr = get_path(sid_text, ctypes.byref(path_ptr))
    if hr < 0 or not path_ptr.value:
        raise SandboxUnavailable(f"Windows could not resolve the AppContainer's private data folder (HRESULT 0x{ctypes.c_uint32(hr).value:08x}).")
    try:
        folder = Path(path_ptr.value)
    finally:
        ctypes.WinDLL("ole32").CoTaskMemFree.argtypes = [ctypes.c_void_p]
        ctypes.WinDLL("ole32").CoTaskMemFree(ctypes.cast(path_ptr, ctypes.c_void_p))
    folder.mkdir(parents=True, exist_ok=True)
    return folder


class _SecurityAttributes(ctypes.Structure):
    _fields_ = [("nLength", ctypes.c_uint32), ("lpSecurityDescriptor", ctypes.c_void_p), ("bInheritHandle", ctypes.c_int)]


class _StartupInfo(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32), ("lpReserved", ctypes.c_wchar_p), ("lpDesktop", ctypes.c_wchar_p), ("lpTitle", ctypes.c_wchar_p),
        ("dwX", ctypes.c_uint32), ("dwY", ctypes.c_uint32), ("dwXSize", ctypes.c_uint32), ("dwYSize", ctypes.c_uint32),
        ("dwXCountChars", ctypes.c_uint32), ("dwYCountChars", ctypes.c_uint32), ("dwFillAttribute", ctypes.c_uint32),
        ("dwFlags", ctypes.c_uint32), ("wShowWindow", ctypes.c_uint16), ("cbReserved2", ctypes.c_uint16),
        ("lpReserved2", ctypes.c_void_p), ("hStdInput", ctypes.c_void_p), ("hStdOutput", ctypes.c_void_p), ("hStdError", ctypes.c_void_p),
    ]


class _StartupInfoEx(ctypes.Structure):
    _fields_ = [("StartupInfo", _StartupInfo), ("lpAttributeList", ctypes.c_void_p)]


class _SecurityCapabilities(ctypes.Structure):
    _fields_ = [("AppContainerSid", ctypes.c_void_p), ("Capabilities", ctypes.c_void_p), ("CapabilityCount", ctypes.c_uint32), ("Reserved", ctypes.c_uint32)]


class _ProcessInformation(ctypes.Structure):
    _fields_ = [("hProcess", ctypes.c_void_p), ("hThread", ctypes.c_void_p), ("dwProcessId", ctypes.c_uint32), ("dwThreadId", ctypes.c_uint32)]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64), ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t), ("PriorityClass", ctypes.c_uint32), ("SchedulingClass", ctypes.c_uint32),
    ]


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInformation), ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _drain_pipe(kernel32: ctypes.WinDLL, handle: int, output: bytearray, output_lock: threading.Lock, exceeded: threading.Event) -> None:
    read_file = kernel32.ReadFile
    read_file.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
    read_file.restype = ctypes.c_int
    while True:
        buffer = ctypes.create_string_buffer(4096)
        count = ctypes.c_uint32()
        if not read_file(handle, buffer, len(buffer), ctypes.byref(count), None) or count.value == 0:
            return
        with output_lock:
            room = max(0, MAX_OUTPUT_BYTES - len(output))
            if room:
                output.extend(buffer.raw[: min(room, count.value)])
            if count.value > room:
                exceeded.set()


def _run_appcontainer(executable: str, arguments: Sequence[str], source: str, suffix: str, timeout: float) -> SandboxedResult:
    if len(source) > MAX_SOURCE_CHARS:
        raise SandboxInputLimit(f"Syntax input exceeds the {MAX_SOURCE_CHARS:,}-character sandbox limit.")
    kernel32, userenv, advapi32 = _win_api()
    sid, sid_text = _container_sid(kernel32, userenv, advapi32)
    root = _appcontainer_data_folder(userenv, sid_text)

    CREATE_SUSPENDED = 0x00000004
    CREATE_NO_WINDOW = 0x08000000
    CREATE_UNICODE_ENVIRONMENT = 0x00000400
    EXTENDED_STARTUPINFO_PRESENT = 0x00080000
    STARTF_USESTDHANDLES = 0x00000100
    PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x00020002
    PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES = 0x00020009
    HANDLE_FLAG_INHERIT = 0x00000001
    GENERIC_READ = 0x80000000
    FILE_SHARE_READ = 0x00000001
    FILE_SHARE_WRITE = 0x00000002
    OPEN_EXISTING = 3
    WAIT_OBJECT_0 = 0
    WAIT_TIMEOUT = 0x00000102
    JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    JOB_OBJECT_LIMIT_PROCESS_TIME = 0x00000002
    JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
    JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

    def checked(handle: int | None, action: str) -> int:
        if handle is None or handle == INVALID_HANDLE_VALUE:
            _raise_last_error(action)
        return int(handle)

    create_pipe = kernel32.CreatePipe
    create_pipe.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(_SecurityAttributes), ctypes.c_uint32]
    create_pipe.restype = ctypes.c_int
    set_handle_info = kernel32.SetHandleInformation
    set_handle_info.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
    set_handle_info.restype = ctypes.c_int
    create_file = kernel32.CreateFileW
    create_file.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(_SecurityAttributes), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    create_file.restype = ctypes.c_void_p
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [ctypes.c_void_p]
    close_handle.restype = ctypes.c_int

    with tempfile.TemporaryDirectory(prefix="syntax-", dir=root) as work:
        workdir = Path(work)
        source_path = workdir / ("input" + suffix)
        source_path.write_text(source, encoding="utf-8", newline="\n")
        replaced_args = [str(source_path) if arg == "{input}" else str(arg) for arg in arguments]
        command_line = ctypes.create_unicode_buffer(subprocess.list2cmdline([executable, *replaced_args]))

        sa = _SecurityAttributes(ctypes.sizeof(_SecurityAttributes), None, 1)
        stdout_read, stdout_write = ctypes.c_void_p(), ctypes.c_void_p()
        stderr_read, stderr_write = ctypes.c_void_p(), ctypes.c_void_p()
        if not create_pipe(ctypes.byref(stdout_read), ctypes.byref(stdout_write), ctypes.byref(sa), 0):
            _raise_last_error("creating bounded stdout pipe")
        if not set_handle_info(stdout_read, HANDLE_FLAG_INHERIT, 0):
            _raise_last_error("restricting stdout handle inheritance")
        if not create_pipe(ctypes.byref(stderr_read), ctypes.byref(stderr_write), ctypes.byref(sa), 0):
            _raise_last_error("creating bounded stderr pipe")
        if not set_handle_info(stderr_read, HANDLE_FLAG_INHERIT, 0):
            _raise_last_error("restricting stderr handle inheritance")
        nul = checked(create_file("NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, ctypes.byref(sa), OPEN_EXISTING, 0, None), "opening the null input handle")

        inherited_handles = (ctypes.c_void_p * 3)(nul, stdout_write.value, stderr_write.value)
        attribute_size = ctypes.c_size_t()
        init_attributes = kernel32.InitializeProcThreadAttributeList
        init_attributes.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_size_t)]
        init_attributes.restype = ctypes.c_int
        init_attributes(None, 2, 0, ctypes.byref(attribute_size))
        attribute_buffer = ctypes.create_string_buffer(attribute_size.value)
        attribute_list = ctypes.cast(attribute_buffer, ctypes.c_void_p)
        if not init_attributes(attribute_list, 2, 0, ctypes.byref(attribute_size)):
            _raise_last_error("initializing restricted process attributes")
        delete_attributes = kernel32.DeleteProcThreadAttributeList
        delete_attributes.argtypes = [ctypes.c_void_p]
        update_attribute = kernel32.UpdateProcThreadAttribute
        update_attribute.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p]
        update_attribute.restype = ctypes.c_int
        capabilities = _SecurityCapabilities(sid, None, 0, 0)  # No network/device capabilities.
        if not update_attribute(attribute_list, 0, PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES, ctypes.byref(capabilities), ctypes.sizeof(capabilities), None, None):
            _raise_last_error("applying AppContainer isolation")
        if not update_attribute(attribute_list, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, ctypes.cast(inherited_handles, ctypes.c_void_p), ctypes.sizeof(inherited_handles), None, None):
            _raise_last_error("restricting inherited handles")

        startup = _StartupInfoEx()
        startup.StartupInfo.cb = ctypes.sizeof(_StartupInfoEx)
        startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES
        startup.StartupInfo.hStdInput = nul
        startup.StartupInfo.hStdOutput = stdout_write.value
        startup.StartupInfo.hStdError = stderr_write.value
        startup.lpAttributeList = attribute_list
        process_info = _ProcessInformation()
        env_values = {
            "SystemRoot": os.environ.get("SystemRoot", r"C:\Windows"),
            "WINDIR": os.environ.get("WINDIR", r"C:\Windows"),
            "PATH": str(Path(executable).resolve().parent),
            "TEMP": work,
            "TMP": work,
        }
        env_text = "\0".join(f"{k}={v}" for k, v in sorted(env_values.items(), key=lambda kv: kv[0].casefold())) + "\0\0"
        env_buffer = ctypes.create_unicode_buffer(env_text)
        create_process = kernel32.CreateProcessW
        create_process.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.POINTER(_SecurityAttributes), ctypes.POINTER(_SecurityAttributes), ctypes.c_int, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_void_p, ctypes.POINTER(_ProcessInformation)]
        create_process.restype = ctypes.c_int

        job = kernel32.CreateJobObjectW(None, None)
        checked(job, "creating a validator Job Object")
        limits = _ExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_TIME | JOB_OBJECT_LIMIT_ACTIVE_PROCESS | JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        limits.BasicLimitInformation.PerProcessUserTimeLimit = PROCESS_TIME_LIMIT_100NS
        # GCC's driver starts confined helper processes such as cc1. Job
        # membership is inherited, so a small bounded tree remains sandboxed.
        limits.BasicLimitInformation.ActiveProcessLimit = 4
        limits.ProcessMemoryLimit = PROCESS_MEMORY_LIMIT
        set_job_info = kernel32.SetInformationJobObject
        set_job_info.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        set_job_info.restype = ctypes.c_int
        if not set_job_info(job, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(limits), ctypes.sizeof(limits)):
            _raise_last_error("setting CPU, memory, and process-count limits")

        pi_created = False
        stdout_data, stderr_data = bytearray(), bytearray()
        output_lock = threading.Lock()
        output_exceeded = threading.Event()
        readers: list[threading.Thread] = []
        try:
            if not create_process(executable, command_line, None, None, 1,
                                  CREATE_SUSPENDED | CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT,
                                  ctypes.cast(env_buffer, ctypes.c_void_p), work, ctypes.cast(ctypes.pointer(startup), ctypes.c_void_p), ctypes.byref(process_info)):
                _raise_last_error("launching the validator in AppContainer")
            pi_created = True
            close_handle(stdout_write.value)
            close_handle(stderr_write.value)
            stdout_write = ctypes.c_void_p()
            stderr_write = ctypes.c_void_p()
            assign = kernel32.AssignProcessToJobObject
            assign.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            assign.restype = ctypes.c_int
            if not assign(job, process_info.hProcess):
                _raise_last_error("assigning the AppContainer process to its resource-limited job")

            readers = [
                threading.Thread(target=_drain_pipe, args=(kernel32, stdout_read.value, stdout_data, output_lock, output_exceeded), daemon=True),
                threading.Thread(target=_drain_pipe, args=(kernel32, stderr_read.value, stderr_data, output_lock, output_exceeded), daemon=True),
            ]
            for thread in readers:
                thread.start()
            resume = kernel32.ResumeThread
            resume.argtypes = [ctypes.c_void_p]
            resume.restype = ctypes.c_uint32
            if resume(process_info.hThread) == 0xFFFFFFFF:
                _raise_last_error("resuming the confined validator")

            deadline = time.monotonic() + timeout
            wait = kernel32.WaitForSingleObject
            wait.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            wait.restype = ctypes.c_uint32
            timed_out = False
            while True:
                result = wait(process_info.hProcess, 50)
                if result == WAIT_OBJECT_0:
                    break
                if result != WAIT_TIMEOUT:
                    _raise_last_error("waiting for the confined validator")
                if output_exceeded.is_set():
                    kernel32.TerminateJobObject(job, 1)
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    kernel32.TerminateJobObject(job, 1)
                    break
            wait(process_info.hProcess, 2000)
            for thread in readers:
                thread.join(timeout=2)
            if timed_out:
                raise SandboxTimeout(f"Sandboxed validator exceeded its {timeout:g}-second wall-clock limit.")
            if output_exceeded.is_set():
                raise SandboxOutputLimit(f"Sandboxed validator exceeded its {MAX_OUTPUT_BYTES}-byte diagnostic limit.")
            exit_code = ctypes.c_uint32()
            get_exit_code = kernel32.GetExitCodeProcess
            get_exit_code.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
            get_exit_code.restype = ctypes.c_int
            if not get_exit_code(process_info.hProcess, ctypes.byref(exit_code)):
                _raise_last_error("reading the confined validator exit code")
            return SandboxedResult(exit_code.value, stdout_data.decode("utf-8", "replace"), stderr_data.decode("utf-8", "replace"))
        finally:
            if pi_created:
                close_handle(process_info.hThread)
                close_handle(process_info.hProcess)
            delete_attributes(attribute_list)
            for handle in (stdout_read.value, stderr_read.value, stdout_write.value, stderr_write.value, nul, job):
                if handle:
                    close_handle(handle)


def run_sandboxed_validator(executable: str, arguments: Sequence[str], source: str, suffix: str, timeout: float = 5) -> SandboxedResult:
    """Run a syntax-only validator in a no-network, resource-limited AppContainer.

    The command must use ``{input}`` for the staged parser input. No shell is
    involved, no project file is mounted, and failure to sandbox never falls
    back to an unrestricted subprocess.
    """
    exe = str(Path(executable).expanduser().resolve())
    if not Path(exe).is_file():
        raise SandboxUnavailable("Validator executable does not exist.")
    return _run_appcontainer(exe, arguments, source, suffix, timeout)
