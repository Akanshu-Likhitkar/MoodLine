from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from contextlib import asynccontextmanager
import pickle
import re
import os


# Model Path
model_path = "BiGRU_Model.keras"

# Tokenizer Path
tokenizer_path = "tokenizer.pkl"

#Max sequence length
max_sequence_path = 50

#Emotion Labels
emotion_labels = ["sadness" , "joy" , "love" , "anger" , "fear" , "surprise"]


# Emoji Mapping
emotion_emojis = {
    "sadness": "😢",
    "joy": "😂",
    "love": "❤️",
    "anger": "😡",
    "fear": "😨",
    "surprise": "😲"
}


# Preprocessing Function
def preprocess_text(text : str) -> str:
    text = text.lower()

    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) #[^a-zA-Z0-9\s] = letters + numbers + spaces ke alawa sab kuch remove.

    text = re.sub(r"\s+", " ", text).strip() # \s+ = multiple spaces ko single space me convert karna, strip() = leading and trailing spaces remove karna

    return text



# Schema for Input

class TextInput(BaseModel):
    text : str = Field(... , 
     min_length=1,  
     max_length=500,
     description="Input text for sentiment analysis." ,
     json_schema_extra={"example" : "I feel so happy and Exicited"}
 )



# Schema for Output
class PredictionResponse(BaseModel):
    text: str = Field(..., description="The input text for sentiment analysis.")
    predicted_emotion: str = Field(..., description="The predicted emotion label.")
    emoji: str = Field(..., description="Emoji for the predicted emotion.")
    confidence_score: float = Field(..., description="The confidence score of the prediction.")
    all_probabilities : dict[str , float]




#   Health Response
class HealthResponse(BaseModel):
    status: str = Field(..., description="The health status of the API.")
    message: str = Field(..., description="A message providing additional information about the health status.")  
    model_loaded: bool = Field(..., description="Indicates whether the model is loaded successfully.")


# Model Loading and LifeSpan Management

dl_model = {}   

# asynccontextmanager -> App start hone se pehle kya karna hai + app band hone ke baad kya cleanup karna hai, dono ek hi function mein manage kar sakte hain.

@asynccontextmanager
async def lifespan(app : FastAPI):
    print("Loading the model and tokenizer...")

    dl_model["BiGRU"] = load_model(model_path)
    with open(tokenizer_path, "rb") as file:
        dl_model["tokenizer"] = pickle.load(file)
        
    print("Model and tokenizer loaded successfully.")

    yield  # Lifespan ka use FastAPI app ke start hone se pehle/baad setup aur band hone par cleanup karne ke liye hota hai.
            # "Startup ka kaam complete ho gaya, ab FastAPI application ko run karne do."

    dl_model.clear()  # Jab FastAPI application band hoti hai, ye dictionary ke andar stored model aur tokenizer ko remove kar deta hai.


# FastAPI App
app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware , 
    allow_origins=["*"] ,
    allow_credentials=True ,
    allow_methods=["*"] ,
    allow_headers=["*"]
    
)


# **Mount ka matlab hota hai kisi path ya URL ko kisi folder ya application ke saath connect karna.**
app.mount("/static" , StaticFiles(directory = "static") , name = "static")


"""
 Server UI at Homepage
 health check endpoint at /health   
 prediction endpoint at /predict 

"""



@app.get("/" , include_in_schema = False)
def serve_homepage():
    return FileResponse("static/index.html")


@app.get("/health" , response_model = HealthResponse)
def health_check():

    model_loaded = (
        dl_model.get("BiGRU") is not None
        and dl_model.get("tokenizer") is not None
    )

    return HealthResponse(
        status = "Server is Running",
        message = "Model and tokenizer are loaded successfully."
        if model_loaded
        else "Model or tokenizer is not loaded.",
        model_loaded = model_loaded
    )





@app.post("/predict" , response_model = PredictionResponse )
def predict_emotion(input_data : TextInput):

    BiGRU_model = dl_model.get("BiGRU")
    tokenizer_model = dl_model.get("tokenizer")


    if BiGRU_model is None or tokenizer_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model or tokenizer not loaded. Please try again later."
        )


    clean_text = preprocess_text(input_data.text)


    sequence = tokenizer_model.texts_to_sequences([clean_text]) 


    padded_sequence = pad_sequences(
        sequence ,
        maxlen=max_sequence_path ,
        truncating="post" ,
        padding="post"  
    )


    # prediction
    probabilities = BiGRU_model.predict(
        padded_sequence,
        verbose=0
    )[0]


    # saari prediction me se max value ka index nikalna
    predicted_index = probabilities.argmax()


    predict_emotion = emotion_labels[predicted_index]


    confidence_score = float(probabilities[predicted_index])


    all_probabilities = {
        emotion_labels[i]: float(probabilities[i])
        for i in range(len(emotion_labels))
    }


    emoji = emotion_emojis[predict_emotion]


    return PredictionResponse(
        
        text=input_data.text,
        predicted_emotion=predict_emotion,
        emoji=emoji,
        confidence_score=confidence_score,
        all_probabilities=all_probabilities
    )