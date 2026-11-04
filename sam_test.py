"""
Spottr - SAM 3.1 image test on Modal.

Detects every "person" in one image (or a folder of images) on a Modal GPU,
then saves a copy of each image with boxes, scores and a head count drawn on it.

Run from the VS Code terminal (with your .venv active):
    modal run sam_test.py --path pool.jpg
    modal run sam_test.py --path test_images/ --prompt "swimmer" --threshold 0.4

Needs:
    - Modal set up (`modal setup`)
    - A Modal Secret named "huggingface-secret" containing HF_TOKEN
    - Approved access to facebook/sam3.1 on Hugging Face
"""

from pathlib import Path

import modal

app = modal.App("spottr-sam31-test")

# ---------------------------------------------------------------------------
# The cloud machine: Python 3.12, PyTorch with CUDA, Meta's official SAM 3 code
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch",
        "torchvision",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("huggingface_hub", "pillow", "numpy", "einops")
    .run_commands(
        'pip install "sam3[notebooks] @ git+https://github.com/facebookresearch/sam3.git"'
    )
)

# Weights download once into this Volume and are reused on every later run
weights = modal.Volume.from_name("sam31-weights", create_if_missing=True)
WEIGHTS_DIR = "/weights"
CHECKPOINT = "sam3.1_multiplex.pt"


@app.cls(
    gpu="H100",  # fast GPU for the real speed test (~$4/hr, billed per second)
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={WEIGHTS_DIR: weights},
    timeout=900,  # hard stop after 15 min so a stuck run can't burn credits
    scaledown_window=60,  # shut the GPU down 60 s after the last image
)
class Sam31:
    @modal.enter()
    def load(self):
        """Runs once when the GPU container starts: download + load the model."""
        import inspect
        import os

        from huggingface_hub import hf_hub_download
        from sam3 import model_builder
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        # Show what this version of the SAM 3 repo offers (useful for debugging
        # and for writing the video-tracking script next)
        builders = [n for n in dir(model_builder) if n.startswith("build")]
        print("Model builders available:", builders)
        print("build_sam3_image_model", inspect.signature(build_sam3_image_model))

        ckpt_path = os.path.join(WEIGHTS_DIR, CHECKPOINT)
        if not os.path.exists(ckpt_path):
            print("Downloading SAM 3.1 weights (first run only, ~3.5 GB)...")
            hf_hub_download(
                repo_id="facebook/sam3.1",
                filename=CHECKPOINT,
                local_dir=WEIGHTS_DIR,
            )
            weights.commit()  # save to the Volume for next time

        try:
            model = build_sam3_image_model(
                checkpoint_path=ckpt_path, load_from_HF=False
            )
            self.model_name = "SAM 3.1"
        except Exception as e:  # noqa: BLE001
            # If this repo version can't load the 3.1 checkpoint into the image
            # model, fall back to the default SAM 3 weights so the test still runs.
            print(f"Could not load SAM 3.1 into the image model: {e!r}")
            print("Falling back to default SAM 3 image weights (facebook/sam3).")
            model = build_sam3_image_model()
            self.model_name = "SAM 3 (fallback)"

        self.Sam3Processor = Sam3Processor
        self.model = model
        print(f"Loaded {self.model_name}")

    @modal.method()
    def detect(
        self,
        image_bytes: bytes,
        prompt: str,
        threshold: float,
        mask_iou: float,
        mask_contain: float,
    ) -> dict:
        import io
        import time

        import torch
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        processor = self.Sam3Processor(self.model, confidence_threshold=threshold)

        start = time.perf_counter()
        # SAM 3 mixes bfloat16 and float32 internally; autocast lets PyTorch
        # handle the mix (this is how Meta's own examples run inference)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = processor.set_image(img)
            output = processor.set_text_prompt(state=state, prompt=prompt)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000

        boxes = output["boxes"].float()
        scores = output["scores"].float()
        masks = output["masks"]

        # --- Remove duplicates by comparing OUTLINES (masks), not boxes ---
        # Two people close together (a kid holding a parent) have overlapping
        # boxes but mostly separate outlines, so both are kept. One person
        # detected twice has near-identical outlines, so the weaker one goes.
        keep, removed = list(range(len(scores))), []
        if len(scores) > 1:
            m = masks.reshape(len(scores), -1).float()  # one row per detection
            m = (m > 0.5).float()
            area = m.sum(dim=1)
            inter = m @ m.T  # shared pixels for every pair
            union = area[:, None] + area[None, :] - inter
            iou = inter / union.clamp(min=1)  # overall outline overlap
            smaller = torch.minimum(area[:, None], area[None, :])
            contain = inter / smaller.clamp(min=1)  # how much of the smaller one sits inside

            order = torch.argsort(scores, descending=True).tolist()
            keep, dropped = [], set()
            for i in order:
                if i in dropped:
                    continue
                keep.append(i)
                for j in order:
                    if j == i or j in dropped or j in keep:
                        continue
                    if iou[i, j] > mask_iou:
                        dropped.add(j)
                        removed.append((j, f"outline overlap {iou[i, j]:.2f}"))
                    elif contain[i, j] > mask_contain:
                        dropped.add(j)
                        removed.append((j, f"inside another {contain[i, j]:.2f}"))

        b, s = boxes.cpu().tolist(), scores.cpu().tolist()
        return {
            "model": self.model_name,
            "boxes": [b[i] for i in keep],  # [x1, y1, x2, y2] in pixels
            "scores": [s[i] for i in keep],
            "removed": [
                {"box": b[j], "score": s[j], "reason": why} for j, why in removed
            ],
            "count": len(keep),
            "raw_count": len(s),
            "ms": round(elapsed_ms, 1),
        }


def draw(image_path: Path, result: dict, prompt: str, out_dir: Path) -> Path:
    """Draws boxes, scores and the head count on a copy of the image."""
    from PIL import Image, ImageDraw

    img = Image.open(image_path).convert("RGB")
    d = ImageDraw.Draw(img)
    # Kept detections: blue, numbered so you can match them to the terminal list
    for n, ((x1, y1, x2, y2), s) in enumerate(zip(result["boxes"], result["scores"]), 1):
        d.rectangle([x1, y1, x2, y2], outline=(0, 200, 255), width=3)
        d.text((x1 + 4, max(0, y1 - 14)), f"#{n} {s:.2f}", fill=(0, 200, 255))
    # Removed duplicates: red, so you can check nothing real was dropped
    for r in result["removed"]:
        x1, y1, x2, y2 = r["box"]
        d.rectangle([x1, y1, x2, y2], outline=(255, 60, 60), width=2)
        d.text((x1 + 4, y2 + 2), f"dup {r['score']:.2f}", fill=(255, 60, 60))
    d.rectangle([0, 0, 520, 26], fill=(0, 0, 0))
    d.text(
        (8, 6),
        f'{result["model"]} | "{prompt}": {result["count"]} '
        f'(raw {result["raw_count"]}, {len(result["removed"])} dup) | {result["ms"]} ms',
        fill=(255, 255, 255),
    )
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{image_path.stem}_detected.jpg"
    img.save(out_path)
    return out_path


@app.local_entrypoint()
def main(
    path: str,
    prompt: str = "person",
    threshold: float = 0.5,
    mask_iou: float = 0.7,
    mask_contain: float = 0.85,
):
    """Runs on your laptop: sends images to the GPU and saves the results.

    threshold     minimum confidence to count a detection (0.5)
    mask_iou      drop the weaker of two detections whose outlines overlap more than this (0.7)
    mask_contain  drop a detection if more than this share of its outline sits inside
                  a stronger one's outline (0.85)
    """
    p = Path(path)
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = sorted(f for f in p.iterdir() if f.suffix.lower() in exts) if p.is_dir() else [p]
    if not images:
        print(f"No images found at {p}")
        return

    sam = Sam31()
    out_dir = Path("results")
    for img_path in images:
        result = sam.detect.remote(
            img_path.read_bytes(), prompt, threshold, mask_iou, mask_contain
        )
        out = draw(img_path, result, prompt, out_dir)
        print(
            f"{img_path.name}: {result['count']} x '{prompt}' "
            f"(raw {result['raw_count']}, removed {len(result['removed'])} duplicates) "
            f"in {result['ms']} ms ({result['model']}) -> {out}"
        )
        for n, ((x1, y1, x2, y2), s) in enumerate(
            zip(result["boxes"], result["scores"]), 1
        ):
            print(f"  #{n:<3} score {s:.2f}  box ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})")
        for r in result["removed"]:
            x1, y1, x2, y2 = r["box"]
            print(
                f"  dup  score {r['score']:.2f}  box ({x1:.0f}, {y1:.0f}) to "
                f"({x2:.0f}, {y2:.0f})  removed: {r['reason']}"
            )
