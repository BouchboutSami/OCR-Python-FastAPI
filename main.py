from fastapi import FastAPI
import OCR
from PIL import Image


app = FastAPI()

@app.get("/corr")
async def ocr(img_path : str):
     test = Image.open(img_path)
     chiffre , score , font = OCR.resultat_final(test)
     return  {"result":chiffre,"score":score,"font":font}





