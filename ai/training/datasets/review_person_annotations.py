"""Human review of Person annotations without editing source JSONL manifests."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from PIL import Image, ImageDraw

SPLITS = ("train", "val", "test")
FLAG_NAMES = ("missing_people", "incorrect_boxes", "uncertain_labels")


def load_manifest(root, split):
    """Read one split as-is; this tool never writes to the manifest."""
    path = Path(root) / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Person manifest not found: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("image"), str):
            raise ValueError(f"{path}:{line_number}: expected an image record")
        if not isinstance(row.get("objects"), list):
            raise ValueError(f"{path}:{line_number}: objects must be a list")
        rows.append(row)
    return rows


def _review_key(row):
    return row["split"], Path(row["image"]).as_posix()


def load_reviews(path):
    path = Path(path)
    if not path.is_file():
        return {}
    latest = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or row.get("split") not in SPLITS or not isinstance(row.get("image"), str):
            raise ValueError(f"{path}:{line_number}: malformed review result")
        latest[_review_key(row)] = row
    return latest


def save_review(path, split, image, reviewed, flags, notes=""):
    """Atomically update the separate review JSONL, preserving other results."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {', '.join(SPLITS)}")
    if set(flags) != set(FLAG_NAMES):
        raise ValueError(f"flags must contain exactly {FLAG_NAMES}")
    path = Path(path)
    reviews = load_reviews(path)
    row = {
        "split": split,
        "image": Path(image).as_posix(),
        "reviewed": bool(reviewed),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "flags": {name: bool(flags[name]) for name in FLAG_NAMES},
        "notes": str(notes).strip(),
    }
    reviews[_review_key(row)] = row
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for item in reviews.values():
            stream.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)
    return row


def build_summary(root="data/person", split="all", review_file="data/person/reviews/review_results.jsonl"):
    """Summarize pending, reviewed and flagged manifest records."""
    root = Path(root)
    selected = SPLITS if split == "all" else (split,)
    reviews = load_reviews(review_file)
    records = []
    for split_name in selected:
        for manifest_row in load_manifest(root, split_name):
            key = (split_name, Path(manifest_row["image"]).as_posix())
            review = reviews.get(key)
            flags = review.get("flags", {}) if review else {}
            flagged = any(bool(flags.get(name, False)) for name in FLAG_NAMES) or bool(review and review.get("notes", "").strip())
            records.append({
                "split": split_name,
                "image": manifest_row["image"],
                "reviewed": bool(review and review.get("reviewed") is True),
                "flagged": flagged,
                "flags": {name: bool(flags.get(name, False)) for name in FLAG_NAMES},
                "notes": review.get("notes", "") if review else "",
            })
    return {
        "root": str(root),
        "split": split,
        "total": len(records),
        "reviewed_count": sum(record["reviewed"] for record in records),
        "pending_count": sum(not record["reviewed"] for record in records),
        "flagged_count": sum(record["flagged"] for record in records),
        "reviewed": [record for record in records if record["reviewed"]],
        "pending": [record for record in records if not record["reviewed"]],
        "flagged": [record for record in records if record["flagged"]],
    }


def render_summary(summary):
    lines = [
        f"Person annotation review: {summary['root']} ({summary['split']})",
        f"Total: {summary['total']}  Reviewed: {summary['reviewed_count']}  "
        f"Pending: {summary['pending_count']}  Flagged: {summary['flagged_count']}",
        "",
        "PENDING HUMAN REVIEW:",
    ]
    lines.extend(f"  [{row['split']}] {row['image']}" for row in summary["pending"])
    lines.extend(("", "FLAGGED FOR FOLLOW-UP:"))
    for row in summary["flagged"]:
        flags = ", ".join(name for name, value in row["flags"].items() if value) or "notes"
        lines.append(f"  [{row['split']}] {row['image']} — {flags}")
        if row["notes"]:
            lines.append(f"    {row['notes']}")
    return "\n".join(lines) + "\n"


def draw_annotation_overlay(image, objects):
    """Draw numbered existing boxes on a copy of an image."""
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    for box_number, obj in enumerate(objects, 1):
        x1, y1, x2, y2 = map(float, obj["bbox"])
        draw.rectangle((x1, y1, x2, y2), outline=(40, 255, 80), width=3)
        draw.text((x1 + 3, y1 + 3), f"Person {box_number}", fill=(255, 255, 255),
                  stroke_width=1, stroke_fill=(0, 0, 0))
    return result


def _launch_review(root, split, review_file, max_width, max_height):
    try:
        import tkinter as tk
        from tkinter import messagebox
        from PIL import ImageTk
    except ImportError as exc:
        raise RuntimeError("Tk is unavailable in this Python. Install Tk support or run --summary-only; no extra Python package is required.") from exc

    root = Path(root).resolve()
    selected = SPLITS if split == "all" else (split,)
    entries = [(split_name, row) for split_name in selected for row in load_manifest(root, split_name)]
    if not entries:
        raise ValueError("No annotated images were found for review")
    reviews = load_reviews(review_file)
    window = tk.Tk()
    window.title("Inferno Person annotation review")
    window.geometry("1400x950")
    index = 0
    photo = None
    canvas = tk.Canvas(window, bg="#202020", highlightthickness=0)
    canvas.pack(fill="both", expand=True, padx=8, pady=8)
    status = tk.StringVar()
    tk.Label(window, textvariable=status, anchor="w").pack(fill="x", padx=8)
    controls = tk.Frame(window)
    controls.pack(fill="x", padx=8, pady=8)
    reviewed_var = tk.BooleanVar()
    tk.Checkbutton(controls, text="Reviewed by a human", variable=reviewed_var).pack(side="left")
    flag_vars = {name: tk.BooleanVar() for name in FLAG_NAMES}
    for name, label in (("missing_people", "Suspected missing people"),
                        ("incorrect_boxes", "Suspected incorrect boxes"),
                        ("uncertain_labels", "Uncertain labels")):
        tk.Checkbutton(controls, text=label, variable=flag_vars[name]).pack(side="left", padx=4)
    notes = tk.StringVar()
    tk.Label(window, text="Review notes (does not edit annotations):", anchor="w").pack(fill="x", padx=8)
    tk.Entry(window, textvariable=notes).pack(fill="x", padx=8)
    button_row = tk.Frame(window)
    button_row.pack(fill="x", padx=8, pady=8)

    def current_review():
        split_name, record = entries[index]
        image_rel = Path(record["image"]).as_posix()
        review = reviews.get((split_name, image_rel), {})
        return split_name, record, image_rel, review

    def show_image():
        nonlocal photo
        split_name, record, image_rel, review = current_review()
        image_path = (root / image_rel).resolve()
        if not image_path.is_relative_to(root) or not image_path.is_file():
            raise FileNotFoundError(f"Image is missing or escapes dataset root: {image_rel}")
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        width, height = image.size
        scale = min(max_width / width, max_height / height, 1.0)
        shown = image.resize((max(1, round(width * scale)), max(1, round(height * scale))))
        # Scale the image and boxes together while preserving the original source.
        if scale != 1.0:
            scaled_objects = [{"bbox": [round(float(value) * scale) for value in obj["bbox"]]}
                              for obj in record["objects"]]
        else:
            scaled_objects = record["objects"]
        shown = draw_annotation_overlay(shown, scaled_objects)
        photo = ImageTk.PhotoImage(shown)
        canvas.configure(width=shown.width, height=shown.height)
        canvas.delete("all")
        canvas.create_image(0, 0, image=photo, anchor="nw")
        status.set(f"{index + 1}/{len(entries)}  [{split_name}] {image_rel}  "
                   f"{width}x{height}  boxes={len(record['objects'])}  "
                   f"manifest reviewed={record.get('reviewed', '<missing>')}")
        reviewed_var.set(review.get("reviewed") is True)
        for name in FLAG_NAMES:
            flag_vars[name].set(bool(review.get("flags", {}).get(name, False)))
        notes.set(review.get("notes", ""))

    def save_current():
        split_name, _, image_rel, _ = current_review()
        review = save_review(review_file, split_name, image_rel, reviewed_var.get(),
                             {name: var.get() for name, var in flag_vars.items()}, notes.get())
        reviews[(split_name, image_rel)] = review
        return review

    def navigate(step):
        nonlocal index
        save_current()
        index = min(max(index + step, 0), len(entries) - 1)
        show_image()

    tk.Button(button_row, text="Previous", command=lambda: navigate(-1)).pack(side="left", padx=4)
    tk.Button(button_row, text="Save review", command=save_current).pack(side="left", padx=4)
    tk.Button(button_row, text="Next", command=lambda: navigate(1)).pack(side="left", padx=4)

    def close_window():
        if messagebox.askyesno("Save review", "Save the current review before closing?", parent=window):
            save_current()
        window.destroy()

    window.protocol("WM_DELETE_WINDOW", close_window)
    window.bind("<Left>", lambda _event: navigate(-1))
    window.bind("<Right>", lambda _event: navigate(1))
    show_image()
    window.mainloop()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Review Person annotations without modifying source manifests.")
    parser.add_argument("--data", default="data/person")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--review-file", default="data/person/reviews/review_results.jsonl",
                        help="Separate human review output JSONL")
    parser.add_argument("--summary-only", action="store_true", help="Print review status without opening the GUI")
    parser.add_argument("--summary-out", help="Optionally write the concise status report to this text file")
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--max-height", type=int, default=820)
    args = parser.parse_args(argv)
    if args.max_width < 1 or args.max_height < 1:
        parser.error("--max-width and --max-height must be positive")
    summary = build_summary(args.data, args.split, args.review_file)
    if args.summary_only:
        report = render_summary(summary)
        print(report, end="")
        if args.summary_out:
            output = Path(args.summary_out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(report, encoding="utf-8")
    else:
        _launch_review(args.data, args.split, args.review_file, args.max_width, args.max_height)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
