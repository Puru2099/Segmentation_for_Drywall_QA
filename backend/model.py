import os, math, numpy as np, torch, cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- Configuration ---
IMG_SIZE = 1024
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t"
# Make sure to place the base model checkpoint in the `backend` directory
CKPT_PATH = "sam2.1_hiera_tiny.pt"
# Make sure to place your fine-tuned model in the `backend` directory
FT_PATH = "sam2.1_best_e6_miou0.6201.pt"

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

def load_model():
    """Loads the SAM model and fine-tuned weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam2 = build_sam2(MODEL_CFG, CKPT_PATH, device=device.type)
    predictor = SAM2ImagePredictor(sam2)

    sd = torch.load(FT_PATH, map_location="cpu")
    predictor.model.load_state_dict(sd, strict=False)
    predictor.model.eval()
    return predictor

def predict_mask(predictor, img_np, prompt, thr=0.5, n_points=9):
    """Generates a segmentation mask for the given image and prompt."""
    ih, iw = img_np.shape[:2]
    img_r = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    predictor.set_image(img_r)
    pts = _grid_points(IMG_SIZE, IMG_SIZE, n=n_points)
    ipt = np.stack([pts[:, 1], pts[:, 0]], axis=1).astype(np.float32)
    ilb = np.ones((ipt.shape[0],), dtype=np.float32)

    mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
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
    best = cv2.resize((best > thr).astype("uint8") * 255, (iw, ih), interpolation=cv2.INTER_NEAREST)
    return best