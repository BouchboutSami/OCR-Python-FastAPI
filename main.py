from fastapi import FastAPI , UploadFile , File
import OCR
from PIL import Image
import io


app = FastAPI()

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





