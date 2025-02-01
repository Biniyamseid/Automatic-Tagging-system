from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI(title="Automatic Tagging System")

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8080",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TaggingService:
    def __init__(self, model_path, vectorizer_path, tag_classes):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.tag_classes = tag_classes
        print(f"Loaded tag classes: {self.tag_classes}")  # Debug print
    
    def predict_tags(self, text):
        # Preprocess and vectorize text
        vectorized_text = self.vectorizer.transform([text])
        print(f"Vectorized text shape: {vectorized_text.shape}")  # Debug print
        
        # Get raw predictions
        raw_predictions = self.model.predict(vectorized_text)
        print(f"Raw predictions: {raw_predictions}")  # Debug print
        
        # Convert numeric predictions to class labels
        predicted_tags = []
        for idx, pred in enumerate(raw_predictions[0]):
            if pred == 1:
                tag_name = self.tag_classes[idx]
                predicted_tags.append(tag_name)
        
        print(f"Predicted tags: {predicted_tags}")  # Debug print
        return predicted_tags



class TextInput(BaseModel):
    text: str

# Load tag_classes
tag_classes = joblib.load('models/tag_classes.pkl')

@app.post("/predict_tags")
async def predict_tags(input_text: TextInput):
    try:
        print(f"Received text: {input_text.text[:100]}...")  # Debug print
        tagging_service = TaggingService(
            model_path='models/classifier.pkl', 
            vectorizer_path='models/vectorizer.pkl',
            tag_classes=tag_classes
        )
        tags = tagging_service.predict_tags(input_text.text)
        return {"tags": tags, "raw_text": input_text.text[:100]}  # Include input text in response
    except Exception as e:
        print(f"Error: {str(e)}")  # Debug print
        raise HTTPException(status_code=500, detail=str(e))