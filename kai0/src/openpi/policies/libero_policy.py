import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class HintLookupTransform(transforms.DataTransformFn):
    """pi05 × LMWM (A1/A2): 按 (dataset_id, episode_index, frame_index) 查预算 hint 注入 data["lmwm_hint"].

    hint.npz (export_pi05_hint.py 产) 含平行数组 suite/episode_index/frame_index/hint[N,(K,)D].
    suite → dataset_id 由 datasets_yaml 的 domain_ids 顺序决定 (与本 transform 的 suite_order 一致).
    须在 repack **之前**运行 (读原始样本的 episode_index/frame_index/dataset_id); repack structure
    需含 "lmwm_hint":"lmwm_hint" 以保留. 缺失帧回退零向量 (训练不崩; 应极少).

    hint_path: hint.npz 路径; suite_order: list[str] 按 dataset_id 索引 (suite_order[did]=suite 名).
    """

    hint_path: str
    suite_order: tuple

    def __post_init__(self):
        z = np.load(self.hint_path, allow_pickle=True)
        suites = z["suite"].astype(str)
        eps = z["episode_index"].astype(np.int64)
        fis = z["frame_index"].astype(np.int64)
        hint = z["hint"]  # [N, D] 或 [N, K, D]
        # 建 (did, ep) -> {frame: row} 的紧凑索引: per-(did,ep) 存一个 frame->row 映射.
        name_to_did = {s: i for i, s in enumerate(self.suite_order)}
        index: dict = {}
        for row in range(len(hint)):
            did = name_to_did.get(str(suites[row]))
            if did is None:
                continue
            index.setdefault((did, int(eps[row])), {})[int(fis[row])] = row
        object.__setattr__(self, "_hint", np.asarray(hint))
        object.__setattr__(self, "_index", index)
        object.__setattr__(self, "_dim", hint.shape[1:])  # (D,) 或 (K,D)

    def __call__(self, data: dict) -> dict:
        did = int(np.asarray(data["dataset_id"]).reshape(-1)[0])
        ep = int(np.asarray(data["episode_index"]).reshape(-1)[0])
        fi = int(np.asarray(data["frame_index"]).reshape(-1)[0])
        row = self._index.get((did, ep), {}).get(fi)
        # 模型消费形状 = [hint_len, D]: 单发 D→[1,D]; best-of-K [K,D]→[K,D].
        if row is None:
            shape = (1, self._dim[-1]) if len(self._dim) == 1 else self._dim
            data["lmwm_hint"] = np.zeros(shape, dtype=np.float32)
        else:
            h = self._hint[row].astype(np.float32)          # [D] 或 [K, D]
            data["lmwm_hint"] = h[None] if h.ndim == 1 else h
        return data


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        # LMWM hint (pi05 × LMWM A1/A2): offline-precomputed subgoal vector, injected by
        # HintLookupTransform upstream (per episode/frame). Absent for A0 → pass-through no-op.
        if "lmwm_hint" in data:
            inputs["lmwm_hint"] = data["lmwm_hint"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][:, :7])}
