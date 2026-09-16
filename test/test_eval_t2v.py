from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("eval_t2v", ROOT / "script/eval_t2v.py")
assert SPEC and SPEC.loader
EVAL_T2V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL_T2V)


class ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_manifest(self, lines: list[object]) -> Path:
        directory = Path(self.temporary_directory.name)
        path = directory / "input.jsonl"
        path.write_text("".join(json.dumps(line) + "\n" for line in lines))
        return path

    def test_load_requests_normalizes_optional_seed(self) -> None:
        path = self.write_manifest(
            [
                {"id": "case-1", "prompt": "  first prompt  ", "seed": 7},
                {"id": "case_2", "prompt": "second prompt"},
            ]
        )
        self.assertEqual(
            EVAL_T2V.load_requests(path),
            [
                {"id": "case-1", "prompt": "first prompt", "seed": 7},
                {"id": "case_2", "prompt": "second prompt", "seed": 0},
            ],
        )

    def test_rejects_duplicate_id(self) -> None:
        path = self.write_manifest(
            [{"id": "same", "prompt": "one"}, {"id": "same", "prompt": "two"}]
        )
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            EVAL_T2V.load_requests(path)

    def test_rejects_unsafe_id_and_boolean_seed(self) -> None:
        for record, message in (
            ({"id": "../escape", "prompt": "prompt"}, "safe filename"),
            ({"id": "case", "prompt": "prompt", "seed": True}, "nonnegative integer"),
        ):
            with self.subTest(record=record):
                with self.assertRaisesRegex(ValueError, message):
                    EVAL_T2V.load_requests(self.write_manifest([record]))

    def test_dry_run_needs_no_gpu_dependencies(self) -> None:
        manifest = self.write_manifest([{"id": "case", "prompt": "prompt", "seed": 9}])
        output = manifest.parent / "output"
        process = subprocess.run(
            [
                sys.executable,
                str(ROOT / "script/eval_t2v.py"),
                "--manifest",
                str(manifest),
                "--variant",
                "flash",
                "--output",
                str(output),
                "--dry-run",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        plan = json.loads(process.stdout)
        self.assertEqual(plan["schema"], "lynnreal-t2v-eval-v1")
        self.assertEqual(plan["geometry"], [1344, 768, 240])
        self.assertEqual(plan["samples"], 1)
        self.assertFalse(output.exists())


class SubmitterTests(unittest.TestCase):
    def test_submit_and_dry_run_are_mutually_exclusive(self) -> None:
        process = subprocess.run(
            [
                "bash",
                str(ROOT / "script/aliyun_thailand_b300_submit.sh"),
                "--submit",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(process.returncode, 2)
        self.assertIn("do not combine or repeat", process.stderr)


if __name__ == "__main__":
    unittest.main()
