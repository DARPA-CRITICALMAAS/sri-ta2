#Create python env  
conda create --name cmaas-sri-ta2 python=3.9 -y  
 
source activate cmaas-sri-ta2  
   
#Pytorch  
conda install pytorch::pytorch torchvision torchaudio -c pytorch -y   
  
#Generic packages  
pip install pandas   
  
#openai  
pip install openai backoff  
  
#OCR for PDF processing  
pip install pdf2image pytesseract opencv-python  
conda install tesseract poppler -y  
