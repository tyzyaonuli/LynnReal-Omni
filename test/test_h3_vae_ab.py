import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "script"))
from eval_h3_vae_ab import resolve, completed_variant, sha256
from report_h3_vae_ab import build


class H3VaeABTests(unittest.TestCase):
    def test_resume_rejects_missing_or_corrupt_outputs(self):
        with tempfile.TemporaryDirectory() as out:
            folder = Path(out)
            files = {"latent_sha256": "latents.pt", "audio_sha256": "audio.pt",
                     "output_sha256": "baseline.mp4", "preview_sha256": "baseline-web.mp4"}
            record = {"status": "complete"}
            for key, name in files.items():
                (folder / name).write_bytes(name.encode())
                record[key] = sha256(folder / name)
            (folder / "baseline.json").write_text(json.dumps(record))
            self.assertTrue(completed_variant(folder, "baseline"))
            (folder / "baseline.mp4").write_bytes(b"truncated")
            self.assertFalse(completed_variant(folder, "baseline"))
            (folder / "baseline.mp4").unlink()
            self.assertFalse(completed_variant(folder, "baseline"))

    def test_original_inputs_and_unknown_case(self):
        manifest, cases = resolve(ROOT / "eval/h3_vae_ab/manifest.json", None)
        self.assertEqual([c["request"]["seed"] for c in cases], list(range(41001, 41011)))
        self.assertEqual(manifest["sampling"]["actual_nfe"], 5)
        with self.assertRaises(ValueError):
            resolve(ROOT / "eval/h3_vae_ab/manifest.json", "unknown")

    def test_dry_run_without_model_imports(self):
        with tempfile.TemporaryDirectory() as out:
            process = subprocess.run([sys.executable, "-S", str(ROOT / "script/eval_h3_vae_ab.py"),
                                      "--output", out, "--dry-run"], capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(json.loads((Path(out) / "plan.json").read_text())["gpu_count"], 1)

    def test_report_rejects_mismatched_latent_and_audio(self):
        with tempfile.TemporaryDirectory() as out:
            folder = Path(out) / "case1"
            folder.mkdir()
            record = {"case": {"title": "Test", "request": {"prompt": "</script>", "seed": 1}},
                      "latent_sha256": "same", "audio_sha256": "same", "decoded_audio_sha256": "same", "decoder": {}}
            (folder / "baseline.json").write_text(json.dumps(record))
            (folder / "light.json").write_text(json.dumps(record))
            self.assertEqual(len(build(Path(out))), 1)
            manifest_path = Path(out) / "manifest.json"
            manifest_path.write_text(json.dumps({"cases": [dict(record["case"], id="different")]}))
            with self.assertRaisesRegex(ValueError, "Manifest/result mismatch"):
                build(Path(out), manifest_path)
            self.assertIn('\\u003c/script>', (Path(out) / "player.html").read_text(encoding="utf-8"))
            for key in ("latent_sha256", "audio_sha256", "decoded_audio_sha256"):
                (folder / "light.json").write_text(json.dumps(dict(record, **{key: "different"})))
                with self.assertRaises(ValueError):
                    build(Path(out))


if __name__ == "__main__":
    unittest.main()
