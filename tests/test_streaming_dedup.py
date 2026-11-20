import concurrent.futures
import unittest
from dataset_cleaner.dedup import StreamingDeduplicator


class TestStreamingDeduplicator(unittest.TestCase):
    def setUp(self):
        self.dedup = StreamingDeduplicator(threshold_near=0.80, threshold_semantic=0.88)

    def test_raw_source_check(self):
        code1 = "       IDENTIFICATION DIVISION.\n       PROGRAM-ID. HELLO.\n       PROCEDURE DIVISION.\n           DISPLAY 'HELLO'."
        is_dup, reason = self.dedup.check_raw_source(code1)
        self.assertFalse(is_dup)
        self.assertEqual(reason, "")

        # Exact copy with minor whitespace differences
        code2 = "IDENTIFICATION DIVISION.\nPROGRAM-ID. HELLO.\nPROCEDURE DIVISION.\nDISPLAY 'HELLO'."
        is_dup, reason = self.dedup.check_raw_source(code2)
        self.assertTrue(is_dup)
        self.assertIn("exact_raw_duplicate", reason)

    def test_exact_pair_and_completion_dedup(self):
        inst = "Write a COBOL program that prints hello"
        code = "       IDENTIFICATION DIVISION.\n       PROGRAM-ID. HELLO."

        is_dup, reason, _ = self.dedup.check_and_add(inst, code)
        self.assertFalse(is_dup)

        # Exact repeat
        is_dup, reason, _ = self.dedup.check_and_add(inst, code)
        self.assertTrue(is_dup)
        self.assertEqual(reason, "exact_pair_duplicate")

        # Different prompt, identical code
        inst2 = "Create a hello world program in COBOL"
        is_dup, reason, _ = self.dedup.check_and_add(inst2, code)
        self.assertTrue(is_dup)
        self.assertEqual(reason, "exact_completion_duplicate")

    def test_algorithmic_preservation_rule(self):
        """Preserves distinct algorithms for identical instructions."""
        inst = "Compute the factorial of a positive integer N"
        # Solution A: Iterative
        code_iterative = """
        IDENTIFICATION DIVISION.
        PROGRAM-ID. FACT-ITER.
        DATA DIVISION.
        WORKING-STORAGE SECTION.
        01 N PIC 9(4) VALUE 5.
        01 RES PIC 9(9) VALUE 1.
        01 I PIC 9(4) VALUE 1.
        PROCEDURE DIVISION.
            PERFORM VARYING I FROM 1 BY 1 UNTIL I > N
                MULTIPLY I BY RES
            END-PERFORM
            DISPLAY 'RESULT=' RES
            GOBACK.
        """
        # Solution B: Recursive
        code_recursive = """
        IDENTIFICATION DIVISION.
        PROGRAM-ID. FACT-REC RECURSIVE.
        DATA DIVISION.
        LOCAL-STORAGE SECTION.
        01 N PIC 9(4).
        01 RES PIC 9(9).
        LINKAGE SECTION.
        01 L-N PIC 9(4).
        01 L-RES PIC 9(9).
        PROCEDURE DIVISION USING L-N, L-RES.
            IF L-N <= 1
                MOVE 1 TO L-RES
            ELSE
                SUBTRACT 1 FROM L-N GIVING N
                CALL 'FACT-REC' USING N, RES
                MULTIPLY L-N BY RES GIVING L-RES
            END-IF
            GOBACK.
        """
        is_dup1, _, _ = self.dedup.check_and_add(inst, code_iterative)
        self.assertFalse(is_dup1)

        # Both have same prompt, but completely distinct implementations:
        # Algorithmic Preservation Rule MUST preserve solution B!
        is_dup2, reason2, _ = self.dedup.check_and_add(inst, code_recursive)
        self.assertFalse(is_dup2)
        stats = self.dedup.stats()
        self.assertGreaterEqual(stats["algorithmic_preservations"], 1)

    def test_minhash_near_duplicate(self):
        """Detects near duplicate boilerplates with minor variable renames."""
        inst1 = "Define customer record layout in COBOL"
        code1 = "01 WS-CUST-RECORD. 05 WS-CUST-ID PIC X(10). 05 WS-CUST-NAME PIC X(30). 05 WS-CUST-ADDR PIC X(50)."
        is_dup1, _, _ = self.dedup.check_and_add(inst1, code1)
        self.assertFalse(is_dup1)

        # Almost identical, trivial prefix changes
        inst2 = "Define customer record layout in COBOL"
        code2 = "01 CUST-RECORD. 05 CUST-ID PIC X(10). 05 CUST-NAME PIC X(30). 05 CUST-ADDR PIC X(50)."
        is_dup2, reason, details = self.dedup.check_and_add(inst2, code2)
        self.assertTrue(is_dup2)
        self.assertIn(reason, ("minhash_near_duplicate", "semantic_duplicate", "semantic_code_duplicate"))

    def test_concurrent_streaming_access(self):
        """Verifies thread-safety under heavy concurrent multi-threaded execution."""
        dedup = StreamingDeduplicator()

        def _worker(worker_id):
            added = 0
            dups = 0
            for i in range(25):
                inst = f"Task description for item {i % 5} from worker {worker_id}"
                code = f"DISPLAY 'Worker {worker_id} Item {i % 5}'."
                is_dup, _, _ = dedup.check_and_add(inst, code)
                if is_dup:
                    dups += 1
                else:
                    added += 1
            return added, dups

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_worker, w) for w in range(4)]
            results = [f.result() for f in futures]

        stats = dedup.stats()
        self.assertEqual(stats["total_checked"], 100)
        self.assertGreater(stats["total_indexed"], 0)
        self.assertGreater(stats["exact_duplicate_completions"] + stats["exact_duplicate_pairs"] + stats["near_duplicate_lsh"], 0)

    def test_state_serialization_checkpoint(self):
        """Verifies dump_state and load_state for checkpoint persistence."""
        dedup1 = StreamingDeduplicator()
        dedup1.check_and_add("Task 1", "Code for task 1")
        dedup1.check_and_add("Task 2", "Code for task 2")

        state = dedup1.dump_state()
        dedup2 = StreamingDeduplicator()
        dedup2.load_state(state)

        # dedup2 should detect Task 1 as an exact duplicate
        is_dup, reason, _ = dedup2.check_and_add("Task 1", "Code for task 1")
        self.assertTrue(is_dup)
        self.assertEqual(reason, "exact_pair_duplicate")

        # Novel item should be added
        is_dup3, _, _ = dedup2.check_and_add("Task 3", "Code for task 3")
        self.assertFalse(is_dup3)


if __name__ == "__main__":
    unittest.main()
