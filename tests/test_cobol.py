import unittest
from dataset_cleaner.cobol import (
    classify_cobol_record,
    generate_mainframe_jcl,
    parse_cobol_structure,
    scan_cobol_compliance,
)


class TestCobolEstate(unittest.TestCase):
    def test_complete_cobol_program(self):
        code = """
        IDENTIFICATION DIVISION.
        PROGRAM-ID. ACCTPAY.
        ENVIRONMENT DIVISION.
        CONFIGURATION SECTION.
        DATA DIVISION.
        WORKING-STORAGE SECTION.
        01 WS-AMOUNT PIC 9(7)V99 VALUE 150.75.
        PROCEDURE DIVISION.
            DISPLAY 'AMOUNT: ' WS-AMOUNT.
            GOBACK.
        """
        struct = parse_cobol_structure(code)
        self.assertEqual(struct["program_id"], "ACCTPAY")
        self.assertTrue(struct["divisions"]["identification"])
        self.assertTrue(struct["divisions"]["environment"])
        self.assertTrue(struct["divisions"]["data"])
        self.assertTrue(struct["divisions"]["procedure"])
        self.assertIn("WORKING-STORAGE", struct["sections"])

        status, reason, meta = classify_cobol_record(code)
        self.assertEqual(status, "success")
        self.assertEqual(reason, "")

    def test_missing_divisions_rejection(self):
        # Snippet with only PROCEDURE DIVISION (missing IDENTIFICATION)
        snippet = """
        PROCEDURE DIVISION.
            DISPLAY 'HELLO'.
            STOP RUN.
        """
        status, reason, meta = classify_cobol_record(snippet)
        self.assertEqual(status, "rejected")
        self.assertIn("missing_division:IDENTIFICATION", reason)

    def test_copybook_detection_and_missing_rejection(self):
        code = """
        IDENTIFICATION DIVISION.
        PROGRAM-ID. CUSTPRG.
        DATA DIVISION.
        WORKING-STORAGE SECTION.
            COPY CUSTREC.
        PROCEDURE DIVISION.
            DISPLAY 'PROCESSED'.
            GOBACK.
        """
        struct = parse_cobol_structure(code)
        self.assertEqual(struct["copybooks"], ["CUSTREC"])

        # If copybook is not available -> rejected
        status, reason, meta = classify_cobol_record(code, available_copybooks=set())
        self.assertEqual(status, "rejected")
        self.assertIn("missing_copybook:CUSTREC", reason)

        # If copybook is available -> success
        status, reason, meta = classify_cobol_record(code, available_copybooks={"CUSTREC"})
        self.assertEqual(status, "success")

    def test_compliance_scanning_private_ip(self):
        code = """
        IDENTIFICATION DIVISION.
        PROGRAM-ID. SOCKCLI.
        DATA DIVISION.
        WORKING-STORAGE SECTION.
        01 SERVER-IP PIC X(15) VALUE '10.240.12.8'.
        PROCEDURE DIVISION.
            DISPLAY 'CONNECTING'.
            GOBACK.
        """
        findings = scan_cobol_compliance(code)
        self.assertTrue(any(f[0] == "private_ipv4" for f in findings))

        status, reason, meta = classify_cobol_record(code)
        self.assertEqual(status, "rejected")
        self.assertIn("compliance_private_ipv4", reason)

    def test_compliance_scanning_pii_and_proprietary(self):
        code_pii = "01 SSN-FIELD PIC X(11) VALUE '123-45-6789'."
        findings = scan_cobol_compliance(code_pii)
        self.assertTrue(any(f[0] == "pii_ssn" for f in findings))

        code_prop = "* CONFIDENTIAL AND PROPRIETARY TO GLOBAL CORP"
        findings = scan_cobol_compliance(code_prop)
        self.assertTrue(any(f[0] == "proprietary_notice" for f in findings))

    def test_jcl_generation(self):
        jcl = generate_mainframe_jcl("PAYROLL1", job_name="PAYJOB01", sysin_data="20261115\nDEPT01")
        self.assertIn("//PAYJOB01 JOB", jcl)
        self.assertIn("EXEC PGM=PAYROLL1", jcl)
        self.assertIn("//SYSIN    DD *", jcl)
        self.assertIn("20261115", jcl)
        self.assertIn("/*", jcl)

    def test_rectification_is_strictly_scoped_to_cobol(self):
        import json
        from pathlib import Path
        nb_path = Path(__file__).resolve().parents[1] / "kaggle_batch_inference.ipynb"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        cell1 = "".join(nb["cells"][1]["source"])
        cell6 = "".join(nb["cells"][6]["source"])

        # Check Cell 1 guard
        self.assertIn('if OUTPUT_FORMAT in ("cobol_estate", "cobol_rectify") and LANGUAGE != "cobol":', cell1)
        self.assertIn('raise ValueError', cell1)

        # Check Cell 6 validator guard
        self.assertIn('if ctx.get("lang") != "cobol":', cell6)
        self.assertIn('cobol_rectify_is_strictly_restricted_to_cobol', cell6)
        self.assertIn('cobol_estate_is_strictly_restricted_to_cobol', cell6)


if __name__ == "__main__":
    unittest.main()
