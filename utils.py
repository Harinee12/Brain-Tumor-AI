import torch
import timm
import numpy as np
import cv2

import torch.nn.functional as F

import torchvision.transforms as transforms

from PIL import Image

from pytorch_grad_cam import GradCAM

from pytorch_grad_cam.utils.image import (
    show_cam_on_image
)

# ======================================
# CLASS NAMES
# ======================================

class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

# ======================================
# IMAGE TRANSFORM
# ======================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])

# ======================================
# LOAD EFFICIENTNET
# ======================================

def load_efficientnet():

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=4,
        drop_rate=0.3
    )

    model.load_state_dict(
        torch.load(
            "models/efficientnet_model.pth",
            map_location=torch.device("cpu")
        )
    )

    model.eval()

    return model

# ======================================
# LOAD RESNET
# ======================================

def load_resnet():

    model = timm.create_model(
        "resnet18",
        pretrained=False,
        num_classes=4
    )

    model.load_state_dict(
        torch.load(
            "models/resnet_model.pth",
            map_location=torch.device("cpu")
        )
    )

    model.eval()

    return model

# ======================================
# ENABLE DROPOUT
# ======================================

def enable_dropout(model):

    for module in model.modules():

        if module.__class__.__name__.startswith(
            "Dropout"
        ):

            module.train()

# ======================================
# ENSEMBLE PREDICTION
# ======================================

def predict_image(
    efficientnet_model,
    resnet_model,
    image
):

    input_tensor = transform(
        image
    ).unsqueeze(0)

    # ==================================
    # MC DROPOUT
    # ==================================

    enable_dropout(
        efficientnet_model
    )

    efficientnet_predictions = []

    with torch.no_grad():

        for _ in range(10):

            outputs = efficientnet_model(
                input_tensor
            )

            probs = F.softmax(
                outputs,
                dim=1
            )

            efficientnet_predictions.append(
                probs.numpy()
            )

    efficientnet_predictions = np.array(
        efficientnet_predictions
    )

    efficientnet_mean = efficientnet_predictions.mean(
        axis=0
    )

    uncertainty = efficientnet_predictions.std(
        axis=0
    ).mean()

    # ==================================
    # RESNET PREDICTION
    # ==================================

    with torch.no_grad():

        resnet_outputs = resnet_model(
            input_tensor
        )

        resnet_probs = F.softmax(
            resnet_outputs,
            dim=1
        ).numpy()

    # ==================================
    # ENSEMBLE AVERAGE
    # ==================================

    ensemble_probs = (
        efficientnet_mean + resnet_probs
    ) / 2

    predicted_class = np.argmax(
        ensemble_probs
    )

    confidence = np.max(
        ensemble_probs
    )

    prediction = class_names[
        predicted_class
    ]

    confidence_score = round(
        confidence * 100,
        2
    )

    uncertainty_score = round(
        float(uncertainty),
        4
    )

    return (
        prediction,
        confidence_score,
        uncertainty_score
    )

# ======================================
# GRADCAM
# ======================================

def generate_gradcam(
    model,
    image
):

    rgb_image = image.resize((224, 224))

    rgb_image = np.array(
        rgb_image
    ).astype(np.float32) / 255.0

    input_tensor = transform(
        image
    ).unsqueeze(0)

    target_layers = [
        model.conv_head
    ]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    grayscale_cam = cam(
        input_tensor=input_tensor
    )[0]

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    return visualization