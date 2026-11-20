from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dataset_cleaner.sources import (
    SOURCE_EXTENSIONS,
    SourceSpec,
    iter_source,
    normalize_language_name,
    scan_file_header_license,
)


class SourcesTests(unittest.TestCase):
    def test_added_source_extensions_present(self) -> None:
        cobol_exts = {".cbl", ".cob", ".cpy", ".cobol"}
        cuda_exts = {".cu", ".cuh"}
        hcl_exts = {".tf", ".tfvars", ".hcl"}
        for ext in cobol_exts | cuda_exts | hcl_exts | {".sql"}:
            self.assertIn(ext, SOURCE_EXTENSIONS)

    def test_normalize_language_name(self) -> None:
        self.assertEqual(normalize_language_name(".cbl"), "cobol")
        self.assertEqual(normalize_language_name(".cob"), "cobol")
        self.assertEqual(normalize_language_name(".cpy"), "cobol")
        self.assertEqual(normalize_language_name(".cobol"), "cobol")
        self.assertEqual(normalize_language_name(".cu"), "cuda")
        self.assertEqual(normalize_language_name(".cuh"), "cuda")
        self.assertEqual(normalize_language_name(".tf"), "hcl")
        self.assertEqual(normalize_language_name(".tfvars"), "hcl")
        self.assertEqual(normalize_language_name(".hcl"), "hcl")
        self.assertEqual(normalize_language_name(".sql"), "sql")
        self.assertEqual(normalize_language_name(".py"), "py")
        self.assertEqual(normalize_language_name(".pyi"), "python")

    def test_scan_file_header_spdx_license(self) -> None:
        text_with_spdx = (
            "// SPDX-License-Identifier: Apache-2.0\n"
            "// Copyright 2026 Test Author\n"
            "int main() { return 0; }\n"
        )
        lic, conf = scan_file_header_license(text_with_spdx, fallback_license="MIT")
        self.assertEqual(lic, "Apache-2.0")
        self.assertEqual(conf, "file_header_spdx")

    def test_scan_file_header_fallback_license(self) -> None:
        text_without_spdx = (
            "// Simple comment without spdx\n"
            "int main() { return 0; }\n"
        )
        lic, conf = scan_file_header_license(text_without_spdx, fallback_license="BSD-3-Clause")
        self.assertEqual(lic, "BSD-3-Clause")
        self.assertEqual(conf, "repository_level")

    def test_iter_source_with_folder_and_spdx(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            f1 = root / "sample.cob"
            f1.write_text("      * SPDX-License-Identifier: MIT\n       IDENTIFICATION DIVISION.\n", encoding="utf-8")
            f2 = root / "sample.cu"
            f2.write_text("// CUDA kernel\n__global__ void k() {}\n", encoding="utf-8")

            spec = SourceSpec(path=root, license_id="Apache-2.0")
            rows = list(iter_source(spec))
            self.assertEqual(len(rows), 2)
            row_map = {r.row["filename"]: r.row for r in rows}

            cob_row = row_map["sample.cob"]
            self.assertEqual(cob_row["language"], "cobol")
            self.assertEqual(cob_row["license"], "MIT")
            self.assertEqual(cob_row["license_confidence"], "file_header_spdx")

            cu_row = row_map["sample.cu"]
            self.assertEqual(cu_row["language"], "cuda")
            self.assertEqual(cu_row["license"], "Apache-2.0")
            self.assertEqual(cu_row["license_confidence"], "repository_level")


if __name__ == "__main__":
    unittest.main()
