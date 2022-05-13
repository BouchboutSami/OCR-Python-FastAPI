from fastapi import FastAPI , UploadFile , File , Depends
from fastapi.middleware.cors import CORSMiddleware
import OCR
from PIL import Image
import io , cv2


app = FastAPI()


origins = [
    "http://localhost",
    "http://localhost:8000"
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict/")
async def root(file: UploadFile = File(...)):
  image =  await file.read()
  image = Image.open(io.BytesIO(image))
  return {"result":OCR.resultat_final(image)}

@app.get("/corr")
async def ocr(img_path : str):
     test = Image.open(img_path)
     chiffre , score , font = OCR.resultat_final(test)
     return  {"result":chiffre,"score":score,"font":font}





