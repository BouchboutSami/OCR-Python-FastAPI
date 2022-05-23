from PIL import Image
import cv2
from matplotlib import pyplot as plt
import numpy as np 
from io import BytesIO

style=["Calibri","Cambria","courier new","Franklin-Gothic","futura","Helvetica"
      ,"JosefinSans","Montserrat","OpenSans","times new roman"]

def binarisation(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray_img = cv2.bitwise_not(gray_img)
    binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    data = Image.fromarray(binary_img)
    return binary_img,data

def afficher_img(image):
    plt.imshow(cv2.cvtColor(image, 3))
    plt.show()
    
def save_img(image,path):
    data = Image.fromarray(image)
    data.save(path)
    
def matrice_img(image): 
    matrix = np.zeros((image.size[1],image.size[0]))
    for i in range(image.size[1]):
        for j in range(image.size[0]):
            pixel = image.getpixel((j,i))
            if (pixel < 128) :
                matrix [i,j] = 0
            else : 
                matrix [i,j] = 1
    return matrix

def clean_index(x):
    temp=[]
    new=[]
    for i in range(len(x)-1):
        if x[i]+1==x[i+1] : 
            temp += [x[i]]
            if i+2<len(x) and x[i+1]+1!=x[i+2] : temp += [x[i+1]]
        else:
            if len(temp)>1  : 
                new += [temp[0]-2]
                new += [temp[len(temp)-1]]
                temp=[]
            else : new+=[x[i]]
    new+=[temp[0]-2]
    new=new[1:len(new)]
    return new

def index_segmentation(matrix):
    index=[]
    index2=[]
    for i in range(matrix.T.shape[0]):
        if matrix.T[i].all() == 1  : index += [i+1]
    for i in range(matrix.shape[0]):
        if matrix[i].all() == 1  : index2 += [i+1]
    index=clean_index(index)
    index2=clean_index(index2)
    return index,index2


def segmentation(image):
    crop=[]
    image_bin=binarisation(image)
    matrix = matrice_img(image_bin[1])
    index = index_segmentation(matrix)
    index0=index[0];index1=index[1];
    data = Image.fromarray(image_bin[0])
    if len(index[0])==2 and len(index[1])==2 :
        crop = [data.crop((min(index0),min(index1),max(index0),max(index1))).resize((64,64))]
    else: 
        if len(index[0])!=2 and len(index[1])==2:
            for i in range(0,len(index[0])-1,2):
                crop += [data.crop((index0[i],min(index1),index0[i+1],max(index1))).resize((64,64))]
    return crop


def print_matrix(matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            print(int(matrix[i,j]),end='')
        print()
        
def MatrixToImage(matrix):
    img = Image.new('1',(matrix.shape[1],matrix.shape[0]))
    pixels = img.load()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
                pixels[j,i] = int(matrix[i,j])
    #img = img.resize((64,64))
    return img

def corr(A,B):
    cor = 0       
    if A.any() != 0 and B.any() != 0   : cor = np.sum(A * B) / np.sqrt(np.sum(A**2) * np.sum(B**2))
    return cor


def CorrMat(A,B):
    decLig = -( B.shape[0] - 1)
    decCol = -( B.shape[1] - 1)
    C = np.full((A.shape[0] + B.shape[0] - 1 , A.shape[1] + B.shape[1] - 1  ),2,dtype=float)
    
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            C[i,j] = corr(i+decLig,j+decCol,A,B)
    return C

def OCR(image):
    A = matrice_img(image)
    result={}
    for j in range(10):
      B= Image.open("./super/"+str(j)+".png")
      B=matrice_img(B)
    #   C2 = CorrMat(A,B)
    #   maxi = C2[B.shape[0]-1][B.shape[1]-1] 
      maxi = corr(A,B)
      result.update({maxi:str(j)})
    result2={}
    for j in style:
        B = Image.open("./DataSet/" +str(result[max(result)])+ "/" +j+ ".png")
        B=np.array(B,dtype=int)
        # C2 = CorrMat(A,B)
        # maxi = C2[B.shape[0]-1][B.shape[1]-1]
        maxi = corr(A,B)
        result2.update({maxi:str(j)})
        
    return result[max(result)],result2[max(result2)],max(result2)


def App(LIST):
    ocr=[]
    if len(LIST) == 1:
        ocr+=[OCR(LIST[0])]
        return ocr
    else:
        for i in range(len(LIST)):
            ocr+=[OCR(LIST[i])]
        return ocr
      
      
def resultat_final(image):
  image=ajouter_contour(image)
  image=np.asarray(image)
  image = segmentation(image)
  liste_resultats = App(image)
  res=""
  acc=0
  font = liste_resultats[0][1]
  for i in range(len(liste_resultats)):
    res += liste_resultats[i][0]
    acc += liste_resultats[i][2]
  acc /= len(liste_resultats)
  acc *= 100
  acc = int(acc) 
  return res,acc,font

def ajouter_contour(image):
  new_size=(image.size[0]+10,image.size[1]+10)
  new_image=Image.new("RGB",new_size,(255,255,255))
  new_image.paste(image,(5,5))
  return new_image


  