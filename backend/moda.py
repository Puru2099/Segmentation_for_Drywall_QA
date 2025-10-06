from pydantic import BaseModel
import modal

# --- Modal Setup ---
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "torchvision", "torchaudio", "numpy", "opencv-python", "Pillow", "fastapi", "uvicorn", "sam2")
    .add_file("sam2.1_hiera_tiny.pt", "/root/sam2.1_hiera_tiny.pt")
    .add_file("sam2.1_best_e6_miou0.6201.pt", "/root/sam2.1_best_e6_miou0.6201.pt")
)

app = modal.App("sam2-segmentation-api", image=image)

# --- Constants ---
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t"
CKPT_PATH = "/root/sam2.1_hiera_tiny.pt"
FT_PATH = "/root/sam2.1_best_e6_miou0.6201.pt"
IMG_SIZE = 1024

# --- Import SAM2 ---
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def _grid_points(h, w, n=9):
    r = int(math.sqrt(n))
    ys = np.linspace(h * 0.15, h * 0.85, r).astype(np.int32)
    xs = np.linspace(w * 0.15, w * 0.85, r).astype(np.int32)
    return np.array([(y, x) for y in ys for x in xs], dtype=np.int32)


def _ensure_batched(coords, labels):
    if coords is None or labels is None:
        return coords, labels
    if hasattr(coords, "ndim") and coords.ndim == 2:
        coords = coords[None, ...]
    if hasattr(labels, "ndim"):
        if labels.ndim == 1:
            labels = labels[None, ...]
        elif labels.ndim == 3 and labels.shape[-1] == 1:
            labels = labels.squeeze(-1)
    return coords, labels


class SegmentRequest(BaseModel):
    image: str
    prompt: str


@app.function(gpu="A10G", keep_warm=1)
@modal.web_endpoint(method="POST")
def segment(req: SegmentRequest):
    """Segment an image using SAM2 + fine-tuned weights."""
    try:
        # --- Lazy load model (first call per container) ---
        global predictor
        if "predictor" not in globals():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            sam2 = build_sam2(MODEL_CFG, CKPT_PATH, device=device.type)
            predictor = SAM2ImagePredictor(sam2)

            # Load fine-tuned weights
            sd = torch.load(FT_PATH, map_location="cpu")
            predictor.model.load_state_dict(sd, strict=False)
            predictor.model.eval()

        # --- Decode base64 image ---
        image_bytes = base64.b64decode(req.image.split(",")[1])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)

        # --- Preprocess & predict mask ---
        ih, iw = image_np.shape[:2]
        img_r = cv2.resize(image_np, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        predictor.set_image(img_r)

        pts = _grid_points(IMG_SIZE, IMG_SIZE, n=9)
        ipt = np.stack([pts[:, 1], pts[:, 0]], axis=1).astype(np.float32)
        ilb = np.ones((ipt.shape[0],), dtype=np.float32)

        mask_input, unnorm_coords, labels, _ = predictor._prep_prompts(
            ipt, ilb, box=None, mask_logits=None, normalize_coords=True
        )
        unnorm_coords, labels = _ensure_batched(unnorm_coords, labels)

        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
            points=(unnorm_coords, labels), boxes=None, masks=None
        )

        batched_mode = unnorm_coords.shape[0] > 1
        high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
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

        # --- Convert mask to base64 ---
        mask_pil = Image.fromarray(best)
        buffered = io.BytesIO()
        mask_pil.save(buffered, format="PNG")
        mask_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"mask": mask_base64}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
