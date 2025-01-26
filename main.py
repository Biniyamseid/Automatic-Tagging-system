from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

class TaggingService:
    def __init__(self, model_path, vectorizer_path, tag_classes):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.tag_classes = tag_classes
    
    def predict_tags(self, text):
        # Preprocess and vectorize text
        vectorized_text = self.vectorizer.transform([text])
        predictions = self.model.predict(vectorized_text)
        
        # Get actual tag names
        predicted_tags = [
            self.tag_classes[i] 
            for i in range(len(self.tag_classes)) 
            if predictions[0][i] == 1
        ]
        
        return predicted_tags

app = FastAPI(title="Automatic Tagging System")

class TextInput(BaseModel):
    text: str

@app.post("/predict_tags")
async def predict_tags(input_text: TextInput):
    try:
        tagging_service = TaggingService(
            model_path='model.pkl', 
            vectorizer_path='vectorizer.pkl',
            tag_classes=tag_classes
        )
        tags = tagging_service.predict_tags(input_text.text)
        return {"tags": tags}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))