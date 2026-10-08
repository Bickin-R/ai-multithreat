"""Import local weapon-image candidates into a review-only staging area."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from PIL import Image, UnidentifiedImageError

DATA_ROOT = Path("data/weapon")
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_index(path):
    if not path.exists(): return []
    rows=[]
    for number,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip(): continue
        try: row=json.loads(line)
        except json.JSONDecodeError as exc: raise ValueError(f"{path}:{number}: malformed candidate index") from exc
        if not isinstance(row,dict): raise ValueError(f"{path}:{number}: candidate entry must be an object")
        rows.append(row)
    return rows


def _write_index(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row,separators=(",",":"))+"\n" for row in rows),encoding="utf-8")
    temporary.replace(path)


def _unique_name(name, digest, destination):
    stem=re.sub(r"[^A-Za-z0-9_-]+","_",Path(name).stem).strip("_")[:60] or "candidate"
    candidate=f"{stem}_{digest[:12]}{Path(name).suffix.lower()}"
    number=1
    while (destination/candidate).exists():
        if sha256(destination/candidate)==digest: return candidate
        candidate=f"{stem}_{digest[:12]}_{number}{Path(name).suffix.lower()}"; number+=1
    return candidate


def import_candidates(source, data_root=DATA_ROOT):
    source=Path(source).expanduser().resolve(); root=Path(data_root)
    if not source.is_dir(): raise ValueError(f"Input directory does not exist: {source}")
    staging=root/"candidates"; image_dir=staging/"images"; index_path=staging/"index.jsonl"
    image_dir.mkdir(parents=True,exist_ok=True)
    rows=_read_index(index_path); known={row.get("sha256") for row in rows}; added=duplicates=invalid=0
    # Final images already reviewed by a person also count as known content.
    final_images=root/"images"
    if final_images.is_dir():
        for existing in final_images.rglob("*"):
            if existing.is_file() and existing.suffix.lower() in SUPPORTED:
                try: known.add(sha256(existing))
                except OSError: pass
    paths=sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
    for path in paths:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image: width,height=image.size
            digest=sha256(path)
            if digest in known: duplicates+=1; continue
            filename=_unique_name(path.name,digest,image_dir)
            shutil.copy2(path,image_dir/filename)
            rel=(Path("candidates")/"images"/filename).as_posix()
            relative=path.relative_to(source).as_posix()
            group=Path(relative).parts[0] if len(Path(relative).parts)>1 else Path(path).stem
            # Capture sessions are groups too; keep all frames together when splitting.
            match=re.match(r"(.+_\d{2,})_\d{6}$",Path(path).stem)
            if match: group=match.group(1)
            rows.append({"candidate":rel,"source_name":path.name,"source_group":group,
                         "sha256":digest,"width":width,"height":height,"status":"pending"})
            known.add(digest); added+=1
        except (OSError,UnidentifiedImageError,ValueError):
            invalid+=1
    _write_index(index_path,rows)
    return {"scanned":len(paths),"added":added,"duplicates":duplicates,"invalid_images":invalid,
            "pending_total":sum(row.get("status")=="pending" for row in rows),
            "staging_directory":str(staging)}


def main(argv=None):
    parser=argparse.ArgumentParser(description="Copy local candidate images into the weapon review queue; never writes train/val/test.")
    parser.add_argument("--input",required=True,help="Local directory of candidate images")
    args=parser.parse_args(argv)
    try: report=import_candidates(args.input)
    except ValueError as exc: parser.error(str(exc))
    print(json.dumps(report,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
