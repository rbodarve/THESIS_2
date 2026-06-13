#!/usr/bin/env python3
"""
classify_erotica_images.py
==========================

Independent content-classification utility (separate from the Computer-Vision
and NLP parts of the thesis).

It walks every image inside the Roboflow COCO export
``erotica detection.v7i.coco.zip`` (or an already-extracted folder) and writes a
CSV with, for each image:

    * presence    -> one of {none, male, female, both, undetermined}
    * intercourse -> one of {no, possible, likely}

These attributes are NOT in the dataset annotations (which only contain
parts / nsfw / safe boxes), so they are *inferred* by running the NudeNet
body-part detector on each image and reasoning over the detected parts and
their bounding-box geometry.

How the columns are derived
---------------------------
NudeNet returns body-part detections such as FEMALE_BREAST_EXPOSED,
MALE_GENITALIA_EXPOSED, ANUS_EXPOSED, etc. We keep detections whose score is
>= --score-threshold and map them:

    female indicators : FACE_FEMALE, FEMALE_BREAST_EXPOSED/COVERED,
                        FEMALE_GENITALIA_EXPOSED/COVERED
    male indicators   : FACE_MALE, MALE_GENITALIA_EXPOSED, MALE_BREAST_EXPOSED
    ambiguous parts   : BUTTOCKS_*, ANUS_*, BELLY_*, FEET_*, ARMPITS_*

    presence:
        both         -> at least one male AND one female indicator
        male/female  -> only that gender's indicators
        undetermined -> only ambiguous parts detected (gender unclear)
        none         -> nothing detected above threshold

    intercourse (heuristic from box geometry):
        likely       -> a male-genitalia box overlaps a "receptive" box
                        (female genitalia / anus / buttocks)
        possible     -> both a male-genitalia box and a receptive box exist
                        but they do not overlap
        no           -> otherwise

NOTE: "intercourse" is a geometric heuristic, not a trained classifier. Treat
it as a weak signal for triage, not ground truth.

Usage
-----
    # default: read straight from the zip next to this script
    python classify_erotica_images.py

    # custom paths / quick smoke test on 50 images
    python classify_erotica_images.py \
        --zip "erotica detection.v7i.coco.zip" \
        --out erotica_image_classification.csv \
        --limit 50

    # from an already-extracted folder instead of the zip
    python classify_erotica_images.py --images-dir ./extracted

Dependencies (already present in the conda `claude` env):
    pip install nudenet onnxruntime opencv-python-headless numpy pandas tqdm
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import zipfile
from pathlib import Path

try:
    from nudenet import NudeDetector
except ImportError:
    sys.exit("nudenet is not installed. Run:  pip install nudenet")

try:
    from tqdm import tqdm
except ImportError:  # progress bar is optional
    def tqdm(it, **kw):
        return it


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

FEMALE_LABELS = {
    "FACE_FEMALE",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_BREAST_COVERED",
    "FEMALE_GENITALIA_EXPOSED",
    "FEMALE_GENITALIA_COVERED",
}
MALE_LABELS = {
    "FACE_MALE",
    "MALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
}
# Boxes used for the intercourse-contact heuristic.
MALE_GENITAL_LABELS = {"MALE_GENITALIA_EXPOSED"}
RECEPTIVE_LABELS = {
    "FEMALE_GENITALIA_EXPOSED",
    "FEMALE_GENITALIA_COVERED",
    "ANUS_EXPOSED",
    "ANUS_COVERED",
    "BUTTOCKS_EXPOSED",
}

CSV_FIELDS = [
    "split",
    "image",
    "image_id",
    "width",
    "height",
    "presence",
    "intercourse",
    "male_present",
    "female_present",
    "num_detections",
    "detected_labels",
]

# --- Optional CLIP intercourse pass -----------------------------------------
# Columns added when CLIP scoring is enabled (augment or --intercourse-model clip).
CLIP_FIELDS = ["intercourse_clip", "clip_score", "intercourse_disagree"]

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
# Zero-shot prompts. clip_score = summed softmax prob of the POSITIVE group.
CLIP_POS_PROMPTS = [
    "two people having sexual intercourse",
    "a man and a woman having sex",
    "explicit sexual penetration between two people",
    "a couple engaged in a sex act",
    "oral sex between two people",
]
# Deliberately many and varied so benign images draw softmax mass away from the
# positive group (zero-shot CLIP is uncalibrated; this is the main defense
# against false positives like portraits/selfies scoring as "intercourse").
CLIP_NEG_PROMPTS = [
    "a single nude person alone, not having sex",
    "a portrait of one person",
    "a selfie of a person",
    "a fully clothed person posing",
    "two people standing next to each other, not having sex",
    "people at a concert or performance",
    "a person at the beach or pool",
    "a non-sexual safe-for-work photograph",
    "a close-up of a face",
]

# Zero-shot presence/gender groups. Each image is matched to the group with the
# highest summed softmax prob; if that top prob is below the margin the image is
# labelled 'undetermined' (CLIP wasn't confident enough to call a gender).
CLIP_PRESENCE_GROUPS = {
    "female": ["a photo of a woman", "a nude woman", "a female person"],
    "male": ["a photo of a man", "a nude man", "a male person"],
    "both": ["a man and a woman together", "a heterosexual couple",
             "a male and a female person in the same photo"],
    "none": ["a photo with no people in it", "scenery or objects with no person",
             "an empty scene with no people"],
}
CLIP_PRESENCE_MARGIN = 0.40  # top-group prob below this -> undetermined


def _xywh_to_xyxy(box):
    x, y, w, h = box
    return x, y, x + w, y + h


def _boxes_overlap(a, b):
    ax1, ay1, ax2, ay2 = _xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = _xywh_to_xyxy(b)
    return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2


def classify(detections, score_threshold):
    """Return (presence, intercourse, male_present, female_present, kept_dets)."""
    kept = [d for d in detections if d.get("score", 0.0) >= score_threshold]

    classes = {d["class"] for d in kept}
    male_present = bool(classes & MALE_LABELS)
    female_present = bool(classes & FEMALE_LABELS)
    ambiguous = bool(classes - MALE_LABELS - FEMALE_LABELS)

    if male_present and female_present:
        presence = "both"
    elif male_present:
        presence = "male"
    elif female_present:
        presence = "female"
    elif ambiguous:
        presence = "undetermined"
    else:
        presence = "none"

    male_boxes = [d["box"] for d in kept if d["class"] in MALE_GENITAL_LABELS]
    recv_boxes = [d["box"] for d in kept if d["class"] in RECEPTIVE_LABELS]
    if male_boxes and recv_boxes:
        if any(_boxes_overlap(m, r) for m in male_boxes for r in recv_boxes):
            intercourse = "likely"
        else:
            intercourse = "possible"
    else:
        intercourse = "no"

    return presence, intercourse, male_present, female_present, kept


def load_coco_meta(name_to_meta, fp):
    """Populate name_to_meta from an open COCO json file object."""
    data = json.load(fp)
    for img in data.get("images", []):
        name_to_meta[img["file_name"]] = (
            img.get("id", ""),
            img.get("width", ""),
            img.get("height", ""),
        )


def iter_zip_images(zip_path):
    """Yield (split, filename, image_bytes) for every image in the zip."""
    zf = zipfile.ZipFile(zip_path)
    meta = {}
    for ann in [n for n in zf.namelist() if n.endswith("_annotations.coco.json")]:
        with zf.open(ann) as f:
            load_coco_meta(meta, f)
    for name in zf.namelist():
        if not name.lower().endswith(IMAGE_EXTS):
            continue
        parts = name.split("/")
        split = parts[0] if len(parts) > 1 else ""
        fname = parts[-1]
        yield split, fname, meta.get(fname, ("", "", "")), zf.read(name)


def iter_dir_images(root):
    """Yield (split, filename, meta, image_bytes) for an extracted folder."""
    root = Path(root)
    meta = {}
    for ann in root.rglob("_annotations.coco.json"):
        with open(ann) as f:
            load_coco_meta(meta, f)
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        split = p.parent.name if p.parent != root else ""
        yield split, p.name, meta.get(p.name, ("", "", "")), p.read_bytes()


def _encode_text(ctx, prompts):
    """Return L2-normalized text features for a list of prompts."""
    torch = ctx["torch"]
    inp = ctx["processor"](text=prompts, return_tensors="pt",
                           padding=True).to(ctx["device"])
    with torch.no_grad():
        feats = ctx["model"].get_text_features(**inp)
    return feats / feats.norm(dim=-1, keepdim=True)


def build_clip_context():
    """Load CLIP once and pre-encode all task prompts (text features cached)."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    ctx = {"torch": torch, "model": model, "processor": processor,
           "device": device, "logit_scale": model.logit_scale.exp()}

    # Intercourse task: positive prompts first, then negatives.
    inter_prompts = CLIP_POS_PROMPTS + CLIP_NEG_PROMPTS
    ctx["inter_text"] = _encode_text(ctx, inter_prompts)
    ctx["n_pos"] = len(CLIP_POS_PROMPTS)

    # Presence task: flat prompt list plus a label per prompt for grouping.
    flat, labels = [], []
    for cat, ps in CLIP_PRESENCE_GROUPS.items():
        flat.extend(ps)
        labels.extend([cat] * len(ps))
    ctx["pres_text"] = _encode_text(ctx, flat)
    ctx["pres_labels"] = labels
    ctx["pres_cats"] = list(CLIP_PRESENCE_GROUPS.keys())
    print(f"CLIP loaded ({CLIP_MODEL_NAME}) on {device}.")
    return ctx


def _image_features(ctx, pil_images):
    torch = ctx["torch"]
    inp = ctx["processor"](images=pil_images, return_tensors="pt").to(ctx["device"])
    with torch.no_grad():
        feats = ctx["model"].get_image_features(**inp)
    return feats / feats.norm(dim=-1, keepdim=True)


def clip_score_images(ctx, pil_images):
    """Intercourse-only: list of summed positive-prompt probabilities."""
    img = _image_features(ctx, pil_images)
    logits = ctx["logit_scale"] * img @ ctx["inter_text"].t()
    probs = logits.softmax(dim=1)
    return probs[:, : ctx["n_pos"]].sum(dim=1).tolist()


def clip_classify_images(ctx, pil_images):
    """One image encode -> (presence, presence_conf, inter_label, inter_score).

    Returns a list of tuples, one per image.
    """
    img = _image_features(ctx, pil_images)

    # Intercourse
    ip = (ctx["logit_scale"] * img @ ctx["inter_text"].t()).softmax(dim=1)
    inter_scores = ip[:, : ctx["n_pos"]].sum(dim=1).tolist()

    # Presence: softmax over all presence prompts, then sum per group.
    pp = (ctx["logit_scale"] * img @ ctx["pres_text"].t()).softmax(dim=1)
    labels = ctx["pres_labels"]
    cats = ctx["pres_cats"]
    out = []
    for row, inter_s in zip(pp.tolist(), inter_scores):
        group = {c: 0.0 for c in cats}
        for lab, p in zip(labels, row):
            group[lab] += p
        best, conf = max(group.items(), key=lambda kv: kv[1])
        presence = best if conf >= CLIP_PRESENCE_MARGIN else "undetermined"
        out.append((presence, conf, inter_s))
    return out


def resolve_image_bytes(split, fname, zip_path=None, images_dir=None):
    """Fetch raw bytes for a (split, image) pair from the zip or a folder."""
    if images_dir:
        root = Path(images_dir)
        for cand in (root / split / fname, root / fname):
            if cand.exists():
                return cand.read_bytes()
        hits = list(root.rglob(fname))
        return hits[0].read_bytes() if hits else None
    zf = zipfile.ZipFile(zip_path)
    member = f"{split}/{fname}" if split else fname
    try:
        return zf.read(member)
    except KeyError:
        for n in zf.namelist():
            if n.endswith("/" + fname) or n == fname:
                return zf.read(n)
        return None


def _clip_scope_match(row, scope):
    """Decide whether a CSV row should be CLIP-scored for the given scope."""
    presence = row.get("presence", "")
    if presence in ("none", "error", ""):
        return scope == "all"
    if scope == "all" or scope == "detected":
        return True
    # scope == "candidates": both genders, or any genitalia/anus detected
    labels = row.get("detected_labels", "") or ""
    return presence == "both" or "GENITALIA" in labels or "ANUS" in labels


def augment_with_clip(args):
    """Read an existing CSV, add CLIP intercourse columns for in-scope rows."""
    import io
    from PIL import Image

    in_csv = args.augment_clip
    out_csv = args.out
    if out_csv == str(Path(__file__).resolve().parent / "erotica_image_classification.csv"):
        out_csv = str(Path(in_csv).with_name(Path(in_csv).stem + "_clip.csv"))

    with open(in_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} rows from {in_csv}")

    todo = [r for r in rows if _clip_scope_match(r, args.clip_scope)]
    print(f"CLIP scope '{args.clip_scope}': scoring {len(todo)} rows "
          f"(batch={args.clip_batch}, threshold={args.clip_threshold})")

    ctx = build_clip_context()
    src_zip = None if args.images_dir else args.zip

    scored = 0
    for i in tqdm(range(0, len(todo), args.clip_batch), desc="clip", unit="batch"):
        batch = todo[i : i + args.clip_batch]
        imgs, valid = [], []
        for r in batch:
            data = resolve_image_bytes(r["split"], r["image"], src_zip, args.images_dir)
            if data is None:
                r["intercourse_clip"] = "missing"
                r["clip_score"] = ""
                r["intercourse_disagree"] = ""
                continue
            try:
                imgs.append(Image.open(io.BytesIO(data)).convert("RGB"))
                valid.append(r)
            except Exception:
                r["intercourse_clip"] = "error"
                r["clip_score"] = ""
                r["intercourse_disagree"] = ""
        if not imgs:
            continue
        scores = clip_score_images(ctx, imgs)
        for r, s in zip(valid, scores):
            label = "yes" if s >= args.clip_threshold else "no"
            box_pos = r.get("intercourse", "no") in ("likely", "possible")
            r["intercourse_clip"] = label
            r["clip_score"] = f"{s:.4f}"
            r["intercourse_disagree"] = int(box_pos != (label == "yes"))
            scored += 1

    fields = CSV_FIELDS + CLIP_FIELDS
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            for c in CLIP_FIELDS:
                r.setdefault(c, "")
            w.writerow({k: r.get(k, "") for k in fields})

    disagree = sum(1 for r in rows if str(r.get("intercourse_disagree", "")) == "1")
    clip_yes = sum(1 for r in rows if r.get("intercourse_clip") == "yes")
    print(f"\nDone. Wrote {out_csv}")
    print(f"CLIP-scored rows: {scored} | clip=yes: {clip_yes} | "
          f"disagreements with box heuristic: {disagree}")


def clip_full_pass(args):
    """Add CLIP presence + intercourse for EVERY row of an existing CSV.

    Encodes each image once and derives both columns, so NudeNet and CLIP can be
    compared on the same image set. Adds: clip_presence, clip_presence_conf,
    clip_intercourse, clip_score, intercourse_disagree.
    """
    import io
    from PIL import Image

    in_csv = args.clip_full
    out_csv = args.out
    if out_csv == str(Path(__file__).resolve().parent / "erotica_image_classification.csv"):
        out_csv = str(Path(in_csv).with_name(Path(in_csv).stem + "_clipfull.csv"))

    with open(in_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]
    print(f"Loaded {len(rows)} rows from {in_csv}; scoring ALL with CLIP "
          f"(batch={args.clip_batch}, intercourse threshold={args.clip_threshold})")

    ctx = build_clip_context()
    src_zip = None if args.images_dir else args.zip
    new_cols = ["clip_presence", "clip_presence_conf",
                "clip_intercourse", "clip_score", "intercourse_disagree"]

    scored = 0
    for i in tqdm(range(0, len(rows), args.clip_batch), desc="clip", unit="batch"):
        batch = rows[i : i + args.clip_batch]
        imgs, valid = [], []
        for r in batch:
            data = resolve_image_bytes(r["split"], r["image"], src_zip, args.images_dir)
            try:
                imgs.append(Image.open(io.BytesIO(data)).convert("RGB"))
                valid.append(r)
            except Exception:
                for c in new_cols:
                    r[c] = "error" if c in ("clip_presence", "clip_intercourse") else ""
        if not imgs:
            continue
        for r, (presence, conf, inter_s) in zip(valid, clip_classify_images(ctx, imgs)):
            label = "yes" if inter_s >= args.clip_threshold else "no"
            box_pos = r.get("intercourse", "no") in ("likely", "possible")
            r["clip_presence"] = presence
            r["clip_presence_conf"] = f"{conf:.4f}"
            r["clip_intercourse"] = label
            r["clip_score"] = f"{inter_s:.4f}"
            r["intercourse_disagree"] = int(box_pos != (label == "yes"))
            scored += 1

    fields = CSV_FIELDS + new_cols
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            for c in new_cols:
                r.setdefault(c, "")
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nDone. Wrote {out_csv}. CLIP-scored rows: {scored}")


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--zip", default=str(here / "erotica detection.v7i.coco.zip"),
                     help="Path to the COCO zip (default: alongside this script).")
    src.add_argument("--images-dir",
                     help="Read images from an extracted folder instead of the zip.")
    ap.add_argument("--out", default=str(here / "erotica_image_classification.csv"),
                    help="Output CSV path.")
    ap.add_argument("--score-threshold", type=float, default=0.20,
                    help="Min detection score to count a body part (default 0.20).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only the first N images (0 = all). For testing.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip images already present in an existing output CSV.")
    # --- CLIP intercourse augmentation ---
    ap.add_argument("--augment-clip", metavar="CSV",
                    help="Add CLIP intercourse columns to an existing NudeNet CSV "
                         "(does NOT re-run NudeNet). Output -> <CSV>_clip.csv unless "
                         "--out is given. Needs torch + transformers.")
    ap.add_argument("--clip-scope", choices=["candidates", "detected", "all"],
                    default="candidates",
                    help="Which rows CLIP scores: 'candidates' = both-gender or "
                         "genitalia/anus rows (default, fastest); 'detected' = any "
                         "row with a detection; 'all' = every row.")
    ap.add_argument("--clip-batch", type=int, default=16,
                    help="CLIP batch size (default 16).")
    ap.add_argument("--clip-threshold", type=float, default=0.5,
                    help="Min summed positive-prompt prob to label clip=yes (0.5).")
    ap.add_argument("--clip-full", metavar="CSV",
                    help="Add CLIP presence (gender) AND intercourse columns to "
                         "every row of an existing NudeNet CSV, for model-vs-model "
                         "comparison. Output -> <CSV>_clipfull.csv unless --out given.")
    args = ap.parse_args()

    if args.clip_full:
        clip_full_pass(args)
        return

    if args.augment_clip:
        augment_with_clip(args)
        return

    if args.images_dir:
        source = iter_dir_images(args.images_dir)
        print(f"Source: folder {args.images_dir}")
    else:
        if not os.path.exists(args.zip):
            sys.exit(f"Zip not found: {args.zip}")
        source = iter_zip_images(args.zip)
        print(f"Source: zip {args.zip}")

    done = set()
    write_header = True
    if args.resume and os.path.exists(args.out):
        with open(args.out, newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["split"], row["image"]))
        write_header = False
        print(f"Resuming: {len(done)} images already classified.")

    detector = NudeDetector()

    counts = {"none": 0, "male": 0, "female": 0, "both": 0, "undetermined": 0}
    sex_total = inter_total = 0

    mode = "a" if (args.resume and not write_header) else "w"
    with open(args.out, mode, newline="") as out:
        writer = csv.DictWriter(out, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        n = 0
        for split, fname, (img_id, w, h), data in tqdm(source, desc="classifying", unit="img"):
            if args.limit and n >= args.limit:
                break
            if (split, fname) in done:
                continue
            n += 1
            try:
                dets = detector.detect(data)
            except Exception as e:  # corrupt image etc. — record and move on
                writer.writerow({
                    "split": split, "image": fname, "image_id": img_id,
                    "width": w, "height": h, "presence": "error",
                    "intercourse": "error", "male_present": "",
                    "female_present": "", "num_detections": "",
                    "detected_labels": f"ERROR: {e}",
                })
                continue

            presence, intercourse, male_p, female_p, kept = classify(
                dets, args.score_threshold)
            counts[presence] = counts.get(presence, 0) + 1
            sex_total += 1
            if intercourse != "no":
                inter_total += 1

            labels = ";".join(f"{d['class']}:{d['score']:.2f}" for d in kept)
            writer.writerow({
                "split": split, "image": fname, "image_id": img_id,
                "width": w, "height": h, "presence": presence,
                "intercourse": intercourse, "male_present": int(male_p),
                "female_present": int(female_p), "num_detections": len(kept),
                "detected_labels": labels,
            })
            if n % 200 == 0:
                out.flush()

    print(f"\nDone. Wrote {args.out}")
    print(f"Classified this run: {sex_total} images")
    print("presence breakdown:", counts)
    print(f"intercourse (possible/likely): {inter_total}")


if __name__ == "__main__":
    main()
