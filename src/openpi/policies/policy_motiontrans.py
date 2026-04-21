import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_motiontrans_example() -> dict:
    """Creates a random input example for the motiontrans policy."""
    return {
        "state": np.random.rand(48),
        "image_1": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "image_2": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "image_3": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    ######Zeqing######
    image = np.asarray(image)
    #print("Before parse:", image.min(), image.max())
    if np.issubdtype(image.dtype, np.floating):
        image = ((image + 1.0) / 2.0 * 255).clip(0,255).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    #print("After parse:", image.min(), image.max())
    ####################
    return image


@dataclasses.dataclass(frozen=True)
class MotionTransInputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    # Number of scene-camera history frames (image_1..image_N). Images beyond this are wrist cameras.
    # Scene cameras are keyed as base_{i}_rgb (receives RandomCrop + ColorJitter augmentation).
    # Wrist cameras are keyed as wrist_{i}_rgb (receives only ColorJitter, no geometric transforms),
    # matching the "wrist" key check in model.preprocess_observation.
    image_history_length: int = 1

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0  # We don't mask for pi0-FAST.

        # Get the state. We are padding from 8 to the model action dim.
        # For pi0-FAST, we don't pad the state (action_dim = 7, which is < 8, so pad is skipped).
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        # Count total images available in this sample.
        total_images = 1
        while f"image_{total_images + 1}" in data:
            total_images += 1

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        image_dict, image_mask_dict = {}, {}

        # Scene camera history: image_1..image_{H} → base_0_rgb..base_{H-1}_rgb
        # Keys without "wrist" receive RandomCrop + ColorJitter in preprocess_observation.
        for i in range(self.image_history_length):
            if f"image_{i + 1}" in data:
                image = _parse_image(data[f"image_{i + 1}"])
                image_dict[f"base_{i}_rgb"] = image
                image_mask_dict[f"base_{i}_rgb"] = np.True_

        # Wrist cameras: image_{H+1}.. → right_wrist_0_rgb, right_wrist_1_rgb, left_wrist_0_rgb, left_wrist_1_rgb
        # Matches zarr DEFAULT_WRIST_CAMERAS order: right_minus, right_plus, left_minus, left_plus.
        # Keys containing "wrist" receive only ColorJitter (no geometric transforms) in preprocess_observation.
        _WRIST_NAMES = ["right_wrist_0_rgb", "right_wrist_1_rgb", "left_wrist_0_rgb", "left_wrist_1_rgb"]
        wrist_idx = 0
        for raw_idx in range(self.image_history_length + 1, total_images + 1):
            if f"image_{raw_idx}" in data:
                image = _parse_image(data[f"image_{raw_idx}"])
                key = _WRIST_NAMES[wrist_idx] if wrist_idx < len(_WRIST_NAMES) else f"wrist_{wrist_idx}_rgb"
                image_dict[key] = image
                image_mask_dict[key] = np.True_
                wrist_idx += 1

        inputs = {
            "state": state,
            "image": image_dict,
            "image_mask": image_mask_dict
        }

        # Actions are only available during training.
        if "actions" in data:
            # We are padding from 7 to the model action dim.
            # For pi0-FAST, this is a no-op (since action_dim = 7).
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "alpha" in data:
            inputs["alpha"] = data["alpha"]

        return inputs


@dataclasses.dataclass(frozen=True)
class MotionTransOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        data.update({"actions": np.asarray(data["actions"])})
        return data
