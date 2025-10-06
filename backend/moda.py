# app.py
import io, base64, math
import modal
from pydantic import BaseModel

# --- Image / Environment ---
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "torchvision", "torchaudio",
        "numpy", "opencv-python", "Pillow",
        "fastapi[standard]", "uvicorn"
    )
    .pip_install("git+https://github.com/facebookresearch/segment-anything-2.git")
    # Use *add_* APIs (copy_* is deprecated in recent Modal)
    .add_local_dir("sam2", "/root/sam2", copy=True)
    .add_local_file("sam2.1_hiera_tiny.pt", "/root/sam2.1_hiera_tiny.pt", copy=True)
    .add_local_file("sam2.1_best_e6_miou0.6201.pt", "/root/sam2.1_best_e6_miou0.6201.pt", copy=True)
)

app = modal.App("sam2-segmentation-api", image=image)

# --- SAM2 config ---
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t"
CKPT_PATH = "/root/sam2.1_hiera_tiny.pt"
FT_PATH   = "/root/sam2.1_best_e6_miou0.6201.pt"
IMG_SIZE  = 1024

# --- Request schema ---
class SegmentRequest(BaseModel):
    image: str  # data URL (data:image/png;base64,...)
    prompt: str | None = None  # not used in this auto-grid demo

# We’ll import heavy deps in-function so the container can warm & snapshot
def _grid_points(h, w, n=9):
    r = int(math.sqrt(n))
    ys = (np.linspace(h * 0.15, h * 0.85, r)).astype(np.int32)
    xs = (np.linspace(w * 0.15, w * 0.85, r)).astype(np.int32)
    return np.array([(y, x) for y in ys for x in xs], dtype=np.int32)

def _ensure_batched(coords, labels):
    if coords is None or labels is None:
        return coords, labels
    if getattr(coords, "ndim", None) == 2:
        coords = coords[None, ...]
    if getattr(labels, "ndim", None) is not None:
        if labels.ndim == 1:
            labels = labels[None, ...]
        elif labels.ndim == 3 and labels.shape[-1] == 1:
            labels = labels.squeeze(-1)
    return coords, labels

# NOTE: Prefer the typed GPU config
@app.function(image=image, gpu=modal.gpu.A10G(), keep_warm=1, timeout=600)
@modal.fastapi_endpoint()  # modern name; replaces @modal.web_endpoint
def segment(req: SegmentRequest):
    import numpy as np
    import torch, cv2
    from PIL import Image
    from fastapi import HTTPException

    try:
        # Lazy-load once per container
        global predictor
        if "predictor" not in globals():
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            sam2 = build_sam2(MODEL_CFG, CKPT_PATH, device=device.type)
            predictor = SAM2ImagePredictor(sam2)
            sd = torch.load(FT_PATH, map_location="cpu")
            predictor.model.load_state_dict(sd, strict=False)
            predictor.model.eval()

        # Decode input image (expects data URL)
        header, b64 = req.image.split(",", 1) if "," in req.image else ("", req.image)
        image_bytes = base64.b64decode(b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)
        ih, iw = image_np.shape[:2]

        # Resize & set image
        img_r = cv2.resize(image_np, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        predictor.set_image(img_r)

        # Auto-point grid prompts (simple demo)
        pts = _grid_points(IMG_SIZE, IMG_SIZE, n=9)
        ipt = np.stack([pts[:, 1], pts[:, 0]], axis=1).astype(np.float32)
        ilb = np.ones((ipt.shape[0],), dtype=np.float32)

        # NOTE: Uses private internals; stable but brittle across SAM2 updates
        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
            ipt, ilb, box=None, mask_logits=None, normalize_coords=True
        )
        unnorm_coords, labels = _ensure_batched(unnorm_coords, labels)

        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
            points=(unnorm_coords, labels), boxes=None, masks=None
        )

        batched_mode = unnorm_coords.shape[0] > 1
        high_res_features = [lvl[-1].unsqueeze(0) for lvl in predictor._features["high_res_feats"]]
        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
            image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )

        up = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
        prob = torch.sigmoid(up[:, 0]).detach().cpu().numpy()
        sc = prd_scores[:, 0].detach().cpu().numpy()
        best = prob[sc.argmax()]
        best = cv2.resize((best > 0.95).astype("uint8") * 255, (iw, ih), interpolation=cv2.INTER_NEAREST)

        # Encode mask as data URL
        buf = io.BytesIO()
        Image.fromarray(best).save(buf, format="PNG")
        mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"mask": f"data:image/png;base64,{mask_b64}"}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
