import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import io

# Load model
class_names = [
    "adenocarcinoma_left.lower.lobe",
    "large.cell.carcinoma_left.hilum",
    "normal",
    "squamous.cell.carcinoma_left.hilum"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18()
model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.4),
    torch.nn.Linear(256, 4)
)
model.load_state_dict(torch.load("/model/chest_cancer_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Add custom CSS for UI improvements
st.markdown("""
    <style>
        body {
            background-color: #2A0039;  /* Dark purple background */
            color: #F2C1D1;  /* Light pink text */
            font-family: 'Arial', sans-serif;
        }
        
        .stButton>button {
            background-color: #9B4D96; /* Purple button */
            color: #FFFFFF;
            border: none;
            border-radius: 8px;
            padding: 15px 30px;
            font-size: 18px;
            transition: background-color 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #6F2C91;  /* Darker purple for hover effect */
        }

        .stFileUploader>label {
            background-color: #9B4D96;
            color: #fff;
            padding: 15px;
            border-radius: 8px;
            font-size: 20px;
            transition: background-color 0.3s ease;
        }

        .stFileUploader>label:hover {
            background-color: #6F2C91;
        }

        h1 {
            font-size: 48px;
            text-align: center;
            color: #F2C1D1;
            margin-top: 50px;
        }

        h2 {
            color: #F2C1D1;
            font-size: 24px;
            text-align: center;
        }

        .stImage {
            border-radius: 10px;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
            transition: transform 0.3s ease;
        }

        .stImage:hover {
            transform: scale(1.05);
        }

        .result-text {
            font-size: 22px;
            font-weight: bold;
            text-align: center;
            color: #F2C1D1;
            margin-top: 30px;
        }

        .result-text span {
            font-size: 24px;
            font-weight: bold;
        }

        .result-text .cancer {
            color: #E84A5F;
        }

        .result-text .no-cancer {
            color: #4CAF50;
        }

    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("Chest Cancer Detection with PyTorch")

# Centering the upload section
st.write("Upload a chest x-ray image for cancer detection. The model will classify it as one of the following categories:")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # Read and display the uploaded image
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Apply transformations
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            prediction = class_names[predicted.item()]

        # Display prediction result
        result_color = "no-cancer" if prediction == "normal" else "cancer"
        st.markdown(f'<p class="result-text">The model predicts: <span class="{result_color}">{prediction}</span></p>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")
