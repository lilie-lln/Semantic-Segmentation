import io
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from PIL import Image

from inference import predict_mask
from utils import load_model

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATHS = {
    "unet": "../saved_models/unet/best_model.pth",
    "resnet34_unet": "../saved_models/resnet34_unet/best_model.pth",
}

models = {
    name: load_model(name, path, device)
    for name, path in MODEL_PATHS.items()
}


@app.get("/")
def root():
    return {
        "status": "ok",
        "device": str(device),
        "models": list(models.keys()),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form("unet"),
):
    image = Image.open(io.BytesIO(await file.read()))
    mask = predict_mask(models[model_name], image, device)

    buffer = io.BytesIO()
    mask.save(buffer, format="PNG")

    return Response(buffer.getvalue(), media_type="image/png")