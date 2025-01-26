## Project Setup Instructions

### 1. Create Project Environment

```bash
# Create a new directory
mkdir automatic-tagging-system
cd automatic-tagging-system

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install required packages
pip install -U pip
pip install torch transformers scikit-learn pandas numpy fastapi uvicorn pydantic datasets
```

### 3. Fetch Dataset from Hugging Face

```python
from datasets import load_dataset

# Example: Load a text classification dataset
dataset = load_dataset("ag_news")  # Alternative: "dbpedia_14"
```

### 4. Project Files Structure

```
automatic-tagging-system/
│
├── venv/
├── src/
│   ├── data_processor.py
│   ├── feature_extractor.py
│   ├── model.py
│   └── app.py
├── notebooks/
│   └── exploration.ipynb
├── models/
│   ├── classifier.pth
│   └── vectorizer.pkl
├── requirements.txt
└── README.md
```

### 5. Training Script

```bash
# Train the model
python src/train.py
```

### 6. Run API

```bash
# Start FastAPI service
uvicorn src.app:app --reload
```

### 7. Test Endpoint

```bash
# Using curl
curl -X POST "http://localhost:8000/predict_tags" \
     -H "Content-Type: application/json" \
     -d '{"text":"Your sample text here"}'
```

![alt text](image.png)
/automatictagging/src$ python train.py
/automatictagging/src$ uvicorn app:app --reload
automatictagging/src$ uvicorn app:app --reload
# Automatic-Tagging-system
