import base64
from fastapi import FastAPI , UploadFile , File
import OCR
from PIL import Image
import io


app = FastAPI()

@app.post("/predict")
async def root(file: UploadFile = File(...)):
  image =  await file.read()
  image = Image.open(io.BytesIO(image))
  return {"result":OCR.resultat_final(image)}

@app.get("/corr")
async def ocr(imgcode : str):
     test =Image.open(io.BytesIO(base64.decodebytes(bytes(imgcode, "utf-8"))))
     chiffre , score , font = OCR.resultat_final(test)
     return  {"result":chiffre,"score":score,"font":font}





