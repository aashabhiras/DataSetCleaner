"""COBOL estate analysis, division parsing, compliance scanning, and JCL generation.

Supports enterprise mainframe modernization requirements:
- Division and section structure analysis (IDENTIFICATION, ENVIRONMENT, DATA, PROCEDURE).
- Missing copybook detection (COPY <name>).
- Enterprise compliance audit (private IPv4, mainframe production DSNs, PII, credentials, proprietary notices).
- Companion JCL (Job Control Language) job execution script generation.
- Classification into 'success' (fully complete, compliant) vs 'rejected' (needs Qwen-14B rectification).
"""

from __future__ import annotations

import re
from typing import Any

# COBOL division patterns
_RE_IDENT_DIV = re.compile(r"\bIDENTIFICATION\s+DIVISION\b", re.IGNORECASE)
_RE_ENV_DIV = re.compile(r"\bENVIRONMENT\s+DIVISION\b", re.IGNORECASE)
_RE_DATA_DIV = re.compile(r"\bDATA\s+DIVISION\b", re.IGNORECASE)
_RE_PROC_DIV = re.compile(r"\bPROCEDURE\s+DIVISION\b", re.IGNORECASE)
_RE_PROG_ID = re.compile(r"\bPROGRAM-ID\.\s*([A-Za-z0-9_-]+)", re.IGNORECASE)

# Copybook references: COPY <name> [IN/OF <lib>] [REPLACING ...]
_RE_COPYBOOK = re.compile(r"\bCOPY\s+([A-Za-z0-9_-]+)(?:\s+(?:IN|OF)\s+[A-Za-z0-9_-]+)?(?:\s+REPLACING\s+.*?)?\s*\.", re.IGNORECASE)

# Source format indicator
_RE_FREE_FORMAT = re.compile(r">>SOURCE\s+FORMAT\s+(?:IS\s+)?FREE", re.IGNORECASE)

# Compliance patterns: enterprise hurdles
_COMPLIANCE_PATTERNS = [
    ("private_ipv4", re.compile(r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b")),
    ("production_dsn", re.compile(r"\b(?:SYS1|SYS2|PROD|PRODUCTION)\.[A-Z0-9#$@_]+(?:\.[A-Z0-9#$@_]+)*\b", re.IGNORECASE)),
    ("hardcoded_credential", re.compile(r"\b(?:PASSWORD|PASSWD|PWD|SECRET_KEY)\s*=\s*['\"][^'\"]+['\"]", re.IGNORECASE)),
    ("racf_user", re.compile(r"\bUSERID\s*=\s*['\"][A-Za-z0-9]+['\"]", re.IGNORECASE)),
    ("pii_ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("pii_credit_card", re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b")),
    ("proprietary_notice", re.compile(r"\b(?:CONFIDENTIAL|PROPRIETARY\s+TO|RESTRICTED\s+RIGHTS|DO\s+NOT\s+DISTRIBUTE)\b", re.IGNORECASE)),
]


def parse_cobol_structure(code: str) -> dict[str, Any]:
    """Parse COBOL structural elements: divisions, program ID, format, copybooks."""
    text = str(code or "")
    has_ident = bool(_RE_IDENT_DIV.search(text))
    has_env = bool(_RE_ENV_DIV.search(text))
    has_data = bool(_RE_DATA_DIV.search(text))
    has_proc = bool(_RE_PROC_DIV.search(text))

    prog_id_match = _RE_PROG_ID.search(text)
    program_id = prog_id_match.group(1).strip() if prog_id_match else None

    is_free_format = bool(_RE_FREE_FORMAT.search(text))
    copybooks = [m.group(1).upper() for m in _RE_COPYBOOK.finditer(text)]

    # Detect sections
    sections = []
    for s in ("WORKING-STORAGE", "LOCAL-STORAGE", "LINKAGE", "FILE", "CONFIGURATION", "INPUT-OUTPUT"):
        if re.search(rf"\b{s}\s+SECTION\b", text, re.IGNORECASE):
            sections.append(s)

    return {
        "program_id": program_id,
        "is_free_format": is_free_format,
        "divisions": {
            "identification": has_ident,
            "environment": has_env,
            "data": has_data,
            "procedure": has_proc,
        },
        "sections": sections,
        "copybooks": copybooks,
        "is_complete_program": has_ident and has_proc,
    }


def scan_cobol_compliance(code: str) -> list[tuple[str, str]]:
    """Scan COBOL source for compliance hurdles: private IPs, DSNs, PII, proprietary banners.

    Returns list of (hurdle_type, detected_snippet).
    """
    text = str(code or "")
    findings: list[tuple[str, str]] = []
    for hurdle_type, regex in _COMPLIANCE_PATTERNS:
        match = regex.search(text)
        if match:
            snippet = match.group(0)[:60]
            findings.append((hurdle_type, snippet))
    return findings


def classify_cobol_record(code: str, available_copybooks: set[str] | None = None) -> tuple[str, str, dict[str, Any]]:
    """Classify a COBOL record into 'success' (clean, compilable asset) or 'rejected' (needs rectification).

    Returns:
        (status, reason, metadata)
    """
    struct = parse_cobol_structure(code)
    compliance = scan_cobol_compliance(code)

    # 1. Compliance blockers
    if compliance:
        hurdle, snippet = compliance[0]
        return "rejected", f"compliance_{hurdle}:{snippet}", {"structure": struct, "compliance_violations": compliance}

    # 2. Structural completeness
    if not struct["divisions"]["identification"]:
        return "rejected", "missing_division:IDENTIFICATION", {"structure": struct}
    if not struct["divisions"]["procedure"]:
        return "rejected", "missing_division:PROCEDURE", {"structure": struct}

    # 3. Missing copybook dependencies
    if struct["copybooks"]:
        available = available_copybooks or set()
        missing = [cb for cb in struct["copybooks"] if cb.upper() not in available]
        if missing:
            return "rejected", f"missing_copybook:{missing[0]}", {"structure": struct, "missing_copybooks": missing}

    return "success", "", {"structure": struct, "compliance_clean": True}


def generate_mainframe_jcl(
    program_name: str,
    job_name: str | None = None,
    step_name: str = "STEP01",
    sysin_data: str | None = None,
) -> str:
    """Generate companion z/OS standard Job Control Language (JCL) script for execution."""
    prog = re.sub(r"[^A-Za-z0-9]", "", program_name.upper())[:8] or "MAINPROG"
    job = (re.sub(r"[^A-Za-z0-9]", "", (job_name or f"{prog}J").upper())[:8]) or "RUNJOB"

    lines = [
        f"//{job:<8} JOB (ACCT#),'DATASETCLEANER RUN',",
        "//         CLASS=A,MSGCLASS=X,MSGLEVEL=(1,1),NOTIFY=&SYSUID",
        "//*",
        "//* COMPANION MAINFRAME EXECUTION JCL",
        f"//* RUNS COMPILED COBOL PROGRAM: {prog}",
        "//*",
        f"//{step_name:<8} EXEC PGM={prog}",
        "//STEPLIB  DD DSN=SYS1.LINKLIB,DISP=SHR",
        "//SYSOUT   DD SYSOUT=*",
        "//SYSPRINT DD SYSOUT=*",
    ]
    if sysin_data:
        lines.append("//SYSIN    DD *")
        for line in sysin_data.strip().splitlines():
            lines.append(line[:80])
        lines.append("/*")
    else:
        lines.append("//SYSIN    DD DUMMY")
    lines.append("//")
    return "\n".join(lines) + "\n"
