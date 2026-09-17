import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "script"))
import prepare_h3_startup_cache as startup
from check_h3_retry import check as retry_check
from eval_h3_vae_ab import resolve, completed_variant, sha256
from report_h3_vae_ab import build
from publish_h3_vae_ab import publication_files


class H3VaeABTests(unittest.TestCase):
    def test_retry_check_preserves_original_and_requires_regeneration(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory) / 'dlcnew'
            root = job / 'eval'
            case = root / 'case01'
            case.mkdir(parents=True)
            (job / 'resumed-from-job.txt').write_text('dlcold')
            (root / 'run.json').write_text(json.dumps({'case_ids': ['case01']}))
            target = case / 'light-web.mp4'
            target.write_bytes(b'video')
            (case / 'baseline.json').write_text('{}')
            result = case / 'light.json'
            result.write_text(json.dumps({'status': 'complete', 'job_id': 'dlcold'}))
            retry_check(root, 'case01', 'prepare')
            self.assertFalse(target.exists())
            self.assertEqual((job / 'injected-light-web.mp4').read_bytes(), b'video')
            with self.assertRaises(FileNotFoundError):
                retry_check(root, 'case01', 'verify')
            target.write_bytes(b'video')
            with self.assertRaises(AssertionError):
                retry_check(root, 'case01', 'verify')
            result.write_text(json.dumps({'status': 'complete', 'job_id': 'dlcnew'}))
            retry_check(root, 'case01', 'verify')
            self.assertEqual(json.loads((job / 'retry-acceptance.json').read_text())['status'], 'passed')

    def test_cold_cache_isolated_and_warm_reuses_verified_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'env/cache.tar.gz'
            archive.parent.mkdir()
            archive.write_bytes(b'environment')
            (archive.parent / 'h3-vae-ab-env-ready.json').write_text(json.dumps({'archive_sha256': startup.digest(archive)}))
            for name in ('h3-bfc8ed0353f5a9733be73e6b2c98ec0948195b86', 'light-vae-e453444c1a52b73a0c5eb0023d208c473c2bb26a'):
                folder = root / 'models' / name
                folder.mkdir(parents=True)
                (folder / 'weights').write_bytes(b'abc')
                (folder / 'H3_VAE_AB_READY.json').write_text(json.dumps({'complete': True, 'files': [{'path': 'weights', 'size': 3, 'sha256': hashlib.sha256(b'abc').hexdigest(), 'verified': True}]}))
            def extract(command, **unused):
                overlay = Path(command[-1]) / 'overlay'
                overlay.mkdir()
                (overlay / 'package.py').write_text('pass')
            with patch.object(startup, 'PERSIST', root), patch.object(startup, 'ARCHIVE', archive), patch.object(startup.subprocess, 'run', side_effect=extract):
                for mode in ('cold', 'warm'):
                    with patch.object(sys, 'argv', ['prepare', '--experiment', 'test', '--mode', mode, '--output', str(root / mode)]):
                        startup.main()
                cold = json.loads((root / 'cold/startup-cache.json').read_text())
                warm = json.loads((root / 'warm/startup-cache.json').read_text())
                self.assertEqual(cold['copied_bytes'], 6)
                self.assertEqual(warm['copied_bytes'], 0)
                self.assertEqual(cold['archive_sha256'], warm['archive_sha256'])
                self.assertEqual((root / 'cache/startup-ab/test/h3/weights').read_bytes(), b'abc')
                with patch.object(sys, 'argv', ['prepare', '--experiment', 'test', '--mode', 'cold', '--output', str(root / 'cold')]):
                    with self.assertRaises(FileExistsError):
                        startup.main()

    def test_publication_only_selects_player_and_previews(self):
        report = [{"id": f"case-{i}", "a": f"case-{i}/baseline-web.mp4",
                   "b": f"case-{i}/light-web.mp4"} for i in range(10)]
        self.assertEqual(len(publication_files(report)), 21)
        report[0]["a"] = "../latents.pt"
        with self.assertRaises(ValueError):
            publication_files(report)

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
        self.assertEqual([c["request"]["seed"] for c in cases[:10]], list(range(41001, 41011)))
        _, details = resolve(ROOT / 'eval/h3_vae_ab/manifest.json', 'human-details')
        self.assertEqual(len(details), 12)
        self.assertEqual([c['request']['seed'] for c in details], list(range(42001,42013)))
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
            build(Path(out), media_base="https://example.com/previews/")
            self.assertIn("https://example.com/previews/case1/baseline-web.mp4", (Path(out) / "player.html").read_text(encoding="utf-8"))
            self.assertEqual(json.loads((Path(out) / "report.json").read_text(encoding="utf-8"))[0]["a"], "case1/baseline-web.mp4")
            manifest_path = Path(out) / "manifest.json"
            manifest_path.write_text(json.dumps({"cases": [dict(record["case"], id="different")]}))
            with self.assertRaisesRegex(ValueError, "Manifest/result mismatch"):
                build(Path(out), manifest_path)
            self.assertIn('\\u003c/script>', (Path(out) / "player.html").read_text(encoding="utf-8"))
            tae = dict(record, status='complete', width=1344, height=768, frames=124, fps=24)
            (folder / 'tae.json').write_text(json.dumps(tae))
            build(Path(out), media_base='https://example.com/', tae_results=Path(out))
            self.assertIn('https://example.com/case1/tae-web.mp4', (Path(out)/'player.html').read_text(encoding='utf-8'))
            self.assertNotIn('__SOURCE_NOTE__', (Path(out)/'player.html').read_text(encoding='utf-8'))
            (folder / 'tae.json').write_text(json.dumps(dict(tae, latent_sha256='wrong')))
            with self.assertRaisesRegex(ValueError, 'TAE input/audio mismatch'):
                build(Path(out), tae_results=Path(out))
            for key in ("latent_sha256", "audio_sha256", "decoded_audio_sha256"):
                (folder / "light.json").write_text(json.dumps(dict(record, **{key: "different"})))
                with self.assertRaises(ValueError):
                    build(Path(out))


if __name__ == "__main__":
    unittest.main()
