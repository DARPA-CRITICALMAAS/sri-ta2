
# For system admin

## From scratch installation 

We assume that you'll be start from a clean Ubuntu Linux + python environment. For instance we are starting with the pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime docker.

The PDF to text step requires a fairly powerful CPU (8 cores+).
OpenAI services will be used so you'll need an OpenAI key.


### Install OCR for document processing
```
sudo apt install tesseract-ocr
sudo apt install poppler-utils
pip install backoff openai tiktoken pandas pytesseract pdf2image
```
### Configuring the system


Download text and predictions on know documents from ___

## Launch the CDR listener process


# Issues

* 10k doc limit
* Case insensitive matching
