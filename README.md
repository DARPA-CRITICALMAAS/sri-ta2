
# SRI deposit type classification

This repository implements a backend daemon for classifying mineral site reports on [Polymer CDR](https://docs.polymer.rocks/cdr).

## Prerequisites
* Obtain a Polymer CDR key for CDR access: contact the Polymer team (Justin Garlow)
* Obtain a MinMod account for automatic knowledge graph update: contact the MinMod team (Binh Vu)
* Obtain an OpenAI API key and sufficient funding (~5 cents per document): https://platform.openai.com/signup
* Installation tested on a Ubuntu Linux machine with python installed. Other system may still work
* Hardware: CPUs with 8+ cores recommended. GPU not required. Internet access needed for OpenAI/CDR/KG

## Installation
### Clone repository
```bash
git clone https://github.com/DARPA-CRITICALMAAS/sri-ta2/ -b SRI_deptype_app
cd sri-ta2
```
### Update config
```bash
mv config_redacted.json config.json
``` 
Update cdr_key, openai_api_key, minmod_username, minmod_password in `config.json` with your credentials.

### Install dependencies
```bash
sudo apt install tesseract-ocr poppler-utils
pip install backoff openai tiktoken pandas pytesseract pdf2image
```


## Usage
Launch the CDR daemon using
```bash
python main.py
```
The system will query CDR every 3 minutes for updates, pull new documents with “[Mineral Site]” prefix, process them with OCR and deposit type classification, and generate data and push to knowledge graph.

## Understanding configurations
* Changing update frequency: `cdr_query_interval` configures update frequency in seconds.
* Changing cache/data directories
  * `dir_cache_pdf`, `dir_cache_ocr` are downloaded PDF files and OCR results. Can be deleted when low on space
  * `dir_predictions` contains the deposit classification results. Scores: logprob distribution, justification: text justification of prediction. 
  * `dir_mineral_sites` contains the mineral site data for updating the minmod knowledge graph. Follows minmod schema
* Changing language model
  * `lm` configures which OpenAI model to use, and `lm_context_window` is the context window size of the OpenAI model
* `taxonomy` configures the deposit types and their descriptions
* `confidence_threshold` configures the minimum confidence to report to KG. Only reports classified to be a deposit type with more than 0.2 probability is sent to the KG
* `ocr_num_threads` configures the number of CPU threads used for report OCR. Try to be slightly less than the number of CPU threads available for optimal speed.

## Contact

- Xiao Lin `xiao.lin at sri.com`
- Meng Ye
- Yi Yao