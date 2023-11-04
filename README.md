# Long document multiple choice mineral deposit type classification

![System Diagram](doc/overview.jpg)

## Installation

In a python>=3.6 environment, install the following packages

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers==4.31
pip install accelerate
pip install tokenizers
pip install sentencepiece
pip install protobuf==3.19
pip install openpyxl
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

## Usage
Deposit type classification
```
python run.py --pdf <path/to/pdf> --options labels_type.csv
```
This will run the deposit type classification pipeline on the given PDF document. By default, this process will use [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) to find relevant paragraphs in the PDF, and use [NousResearch/Llama-2-7b-hf](https://huggingface.co/NousResearch/Llama-2-7b-hf) for multiple choice QA. 

OpenAI GPT-4 justifications of the answer will be provided if you enter your openai api key as follows
```
python run.py --pdf <path/to/pdf> --options labels_type.csv --openai_key <your_openai_api_key>
```

Model for multiple choice QA can be configured with `--lm NousResearch/Llama-2-7b-hf`.
