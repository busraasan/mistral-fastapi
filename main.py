from fastapi import FastAPI
from huggingface_model import HuggingfaceModel
from request_templates import HugginfaceInferenceRequest
import json 

app = FastAPI()

mistral_model = HuggingfaceModel()

@app.head("/")
@app.get("/")

@app.post("/ai/mistral7B", name="ai")
def mistral(payload: HugginfaceInferenceRequest = None):
    print("hello")
    """Generate response from prompt for Mistral 7B."""
    return json.loads(mistral_model.get_json(payload))