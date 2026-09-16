"""Publish only the static player and its browser videos to an isolated OSS prefix."""
import argparse
import json
from pathlib import Path
import re
import subprocess
from report_h3_vae_ab import build


def publication_files(report):
    files = ["player.html"]
    for case in report:
        case_id = case["id"]
        if not re.fullmatch(r"[a-z0-9-]+", case_id):
            raise ValueError("Invalid case ID")
        for key, variant in (("a", "baseline"), ("b", "light")):
            expected = f"{case_id}/{variant}-web.mp4"
            if case[key] != expected:
                raise ValueError("Unexpected preview path")
            files.append(expected)
    if len(set(files)) != len(files) or len(files) != 21:
        raise ValueError("Publication requires exactly ten distinct A/B pairs")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--profile", default="shengdong")
    parser.add_argument("--publish", action="store_true", help="Default only prints the exact publication plan")
    parser.add_argument("--html-only", action="store_true", help="Refresh HTML using already published previews")
    parser.add_argument('--tae-reference', type=Path, help='Add historical TAE videos in randomized blind comparison')
    args = parser.parse_args()
    if not re.fullmatch(r"dlc[a-z0-9]+", args.job_id):
        raise ValueError("Invalid job ID")
    files = publication_files(json.loads(args.report.read_text(encoding="utf-8")))
    bucket = "leap-worldmodel-thailand"
    source = f"oss://{bucket}/world-model/results/lynnreal-omni/{args.job_id}/eval/"
    prefix = f"world-model/public/lynnreal-h3-vae-ab/{args.job_id}/"
    target = f"oss://{bucket}/{prefix}"
    print(json.dumps({"source": source, "target": target, "files": files, "publish": args.publish}, indent=2), flush=True)
    if not args.publish:
        return
    def oss(*command):
        subprocess.run(["aliyun", "--profile", args.profile, "oss", *command,
                        "--region", "ap-southeast-7", "--endpoint", "oss-ap-southeast-7.aliyuncs.com"], check=True)
    # Copy and publish the videos first, then the player. Never change bucket ACL.
    if not args.html_only:
        for name in files[1:]:
            oss("cp", source + name, target + name, "--force")
            oss("set-acl", target + name, "public-read", "--force")
    media_base = f"https://{bucket}.oss-ap-southeast-7.aliyuncs.com/{prefix}"
    build(args.report.parent, Path(__file__).resolve().parents[1] / "eval/h3_vae_ab/manifest.json", media_base, args.tae_reference)
    oss("cp", str(args.report.parent / "player.html"), target + "player.html", "--force")
    oss("set-acl", target + "player.html", "public-read", "--force")
    print(f"https://{bucket}.oss-ap-southeast-7.aliyuncs.com/{prefix}player.html")


if __name__ == "__main__":
    main()
