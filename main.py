from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io

# Initialize FastAPI
app = FastAPI()

# CORS Middleware (to allow frontend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Class names from your model
class_names = [
    "adenocarcinoma_left.lower.lobe",
    "large.cell.carcinoma_left.hilum",
    "normal",
    "squamous.cell.carcinoma_left.hilum"
]

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your ResNet18 model
model = models.resnet18()
model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.4),
    torch.nn.Linear(256, 4)
)
model.load_state_dict(torch.load("/model/chest_cancer_model.pth", map_location=device))  # <- your .pth file
model.to(device)
model.eval()

# Transformations same as training
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Because in your notebook, input was (1, 3, 64, 64)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/")
def home():
    return {"message": "Chest Cancer Prediction API is Live!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            prediction = class_names[predicted.item()]

        return JSONResponse(content={"prediction": prediction})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
