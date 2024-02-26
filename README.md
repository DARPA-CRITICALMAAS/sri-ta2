# SRI TA2 Deposit Type Classification v0b

![System Diagram](doc/overview2.jpg)

## Installation

(Optional) It's recommended to use anaconda to make environment setup easier. Create a virtual environment in anaconda.
```
conda create --name cmaas-sri-ta2 python=3.9 -y
conda activate cmaas-sri-ta2
```

In a python>=3.6 environment, install the following packages

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install transformers accelerate tokenizers sentencepiece openpyxl pandas
```

PDF to text with OCR
```
pip install pdf2image
pip install pytesseract
pip install opencv-python
conda install tesseract
conda install poppler
```

OpenAI GPT-4 explanations
```
pip install openai
pip install accelerate
pip install backoff
```

Note that GPUs with >=24GB of combined GPU RAM is required to run the system.

## Usage (NI 43-101 reports)
Deposit type classification on a given PDF document
```
python run.py --pdf <path/to/pdf.pdf> --options labels_type.csv
```
This will run the deposit type classification pipeline on the given PDF document. By default, this process will first run OCR on the document to extract text, then determine deposit type by making calls to [NousResearch/Llama-2-7b-hf](https://huggingface.co/NousResearch/Llama-2-7b-hf) for multiple choice QA. Each run typically takes 3~4 min on a single RTX A5000. OpenAI GPT-4 justifications of the answer will be provided if you enter your openai api key as follows
```
python run.py --pdf <path/to/pdf> --options labels_type.csv --openai_key <your_openai_api_key>
```
Model for multiple choice QA can be configured with `--lm NousResearch/Llama-2-7b-hf`. 

Results for each run will be stored in `./sessions/vvvvvvv/`. `predictions.csv` stores the top-10 predicted deposit types and their predicted probability. `relevant_paragraphs.csv` stores the top 20 relevant paragraphs retrieved sorted by their relevance score. `explanation.json` stores the GPT justification. For debugging purposes, there are also `params.json` storing the command line parameters and `ocr.json` storing the text extracted from the document.

## Usage (MRDS JSON records)
```
python run_json.py --json <path/to/json.json> --options labels_type.csv --openai_key <your_openai_api_key>
```
Similar to processing PDFs, this will run the deposit type classification pipeline on the given JSON record. [NousResearch/Llama-2-7b-hf](https://huggingface.co/NousResearch/Llama-2-7b-hf) will be used by default. Each run typically takes ~5 min on a single RTX A5000. OpenAI GPT-4 justifications of the answer will be provided if you enter your openai api key.

Results for each run will be stored in the `./sessions/vvvvvvv/` folders.


## Version history
2023-Nov-04 Add initial open source pipeline 
2023-Nov-04 =>v1.1 Using LLM for retrieval instead of embeddings
2024-Feb-22 =>v0b Match algorithm version with hackathon deposit type prediction dataset version. Add MRDS JSON processing capability.
