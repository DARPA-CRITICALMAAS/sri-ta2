
# SRI deposit type classification

This repository implements a backend daemon for classifying mineral site reports on [Polymer CDR Docs](https://docs.polymer.rocks/cdr). The following guide is for system installation and maintanence.

## Prerequisites
* Obtain a Polymer CDR key for CDR API access: contact the Polymer team (Justin Garlow)
* Obtain a MinMod account for automatic knowledge graph update: contact the MinMod team (Binh Vu)
* Obtain an OpenAI API key and sufficient funding (~10 cents per document): https://platform.openai.com/signup 
* Obtain a remotely accessible IP/URL needed for CDR process button callback. 
* Installation tested on a Ubuntu Linux machine with python installed. Other system may still work
* Hardware: CPUs with 8+ cores recommended. GPU not required.

## Running Prebuilt Dockers

Use the following command to launch the prebuilt docker available on dockerhub, with `gpt-4o` as the base language model for deposit type classification.
```bash
docker run -p 9999:9999 grrrgrrr/sri-ta2:app_latest python fast_api_registration.py       \
  --cdr_key <cdr_key>        \
  --openai_api_key <openai_api_key>      \
  --minmod_username <minmod_username>     \   
  --minmod_password <minmod_password>    \
  --cdr_callback_url http://<your ip>:9999    \ 
  --lm gpt-4o     \
  --lm_context_window 128000
```
The options are mostly self-explanatory.

## Configuring the system

The body of the system is a web app. When launched, we first send the IP:port of the app to Polymer CDR. When a new document needs to be processed, Polymer CDR will call our app with the document ID. Our app downloads the document, perform OCR, query the LLM to perform deposit type classification, and then send the results to Minmod and CDR.

In addition to the above mandatory launch parameters, additional launch parameters are available for configurating the system.

### Using Azure LLM endpoints

Set
```bash
  --azure_api_endpoint https://<your_endpoint>    \
  --azure_api_version 2024-07-01-preview    \
  --azure_lm gpt-4o
```
to use model `gpt-4o` on a custom Azure endpoint. 

You may also want to redefine the following parameters to match the Azure endpoint.
```bash
  --openai_api_key <openai_api_key>      \
  --lm_context_window <context_window_size>    \
  --lm <probably_your_azure_lm>
```
Specifically, when a custom Azure OpenAI endpoint is used, option `--lm` is still needed to tell our system which [tiktoken](https://github.com/openai/tiktoken) tokenizer to use for token count estimation. Set `--lm` to the base OpenAI model. 

### Adjusting Minmod/CDR endpoints 

To point Minmod/CDR endpoints to custom deployments, set
```bash
  --cdr_endpoint https://api.cdr.land     \
  --minmod_endpoint https://dev.minmod.isi.edu    \
  --cdr_api_version /v1
```

### Changing deposit type definitions

`taxonomy/deposit_type_descriptions.csv` contains deposit type descriptions used for classification, that defines the deposit types. The descriptions can be modified, and new rows can be added for new deposit types. They shouldn't be too short but they also shouldn't be too long, as today's LLMs don't work very well against very long descriptions. You might want to benchmark performance change after updating the deposit type descriptions.

`taxonomy/minmod/deposit_types.csv` is the deposit type schema that Minmod KG would accept. It should be made consistent with the [Minmod version of this file](https://github.com/DARPA-CRITICALMAAS/ta2-minmod-data/blob/main/data/entities/deposit_type.csv). New lines should be added for deposit classifications on a new deposit type to be visible on Minmod KG.

To modify/add deposit types, create a copy of `taxonomy/` folder with your modifications, and launch docker with

```bash
docker run -p 9999:9999  -v /path/to/new_taxonomy:/path/in/container/to/new_taxonomy grrrgrrr/sri-ta2:app_latest python fast_api_registration.py      \
  ...
  --taxonomy /path/in/container/to/new_taxonomy
```


### Changing CDR callback port

The CDR callback port is 9999 by default, and can be configured through 
```bash
docker run -p <port>:<port> grrrgrrr/sri-ta2:app_latest python fast_api_registration.py     \
  ...
  --cdr_callback_port <port>      \
  --cdr_callback_url http://<your ip>:<port> 

```

### Other launch parameters

Managing number of threads for OCR. More than 4 threads is ideal for speeding up OCR, but too many threads more than CPU cores could slow things down.
```bash
  --ocr_num_threads 12
```
Configure the minimum deposit type prediction confidence to report to Minmod KG. By default, predictions with confidence lower than 0.2 are discarded.
```bash
  --confidence_threshold 0.2
```
Managing storage of cached data (PDF, OCR result, prediction result and Minmod KG JSON data) in docker
```bash
  --dir_cache_pdf cache/docs_PDF   \
  --dir_cache_ocr cache/docs_ocr   \
  --dir_predictions predictions    \
  --dir_mineral_sites sri/mineral_sites
```
Managing system name and secret for CDR callback registration
```
  --cdr_callback_system_name DTC_APP    \
  --cdr_callback_registration_secret mysecret_dtc_app
```

## Local installation

### Docker build
With docker installed ([tutorial](https://www.digitalocean.com/community/tutorial-collections/how-to-install-and-use-docker)), simply clone repo and build Dockerfile.
```bash
git clone https://github.com/DARPA-CRITICALMAAS/sri-ta2/ -b SRI_deptype_app
cd sri-ta2
docker build -t grrrgrrr/sri-ta2:app_latest .
```

### Local build
Local installation requires Python version 3.11 and above.
First install dependencies
```bash
apt update
apt install -y htop screen git ffmpeg vim tesseract-ocr poppler-utils
pip install --upgrade pip
pip install backoff openai tiktoken pandas pytesseract pdf2image
pip install flask flask-cors httpx fastapi uvicorn ngrok cryptography
pip install -e "git+https://github.com/DARPA-CRITICALMAAS/ta2-minmod-kg.git#egg=minmodapi&subdirectory=minmodapi"
```
Then clone the repository
```bash
git clone https://github.com/DARPA-CRITICALMAAS/sri-ta2/ -b SRI_deptype_app
cd sri-ta2
```
For DOI users, install DOI certificate
```bash
cp DOIRootCA2.crt /usr/local/share/ca-certificates
chmod 644 /usr/local/share/ca-certificates/DOIRootCA2.crt && \
    update-ca-certificates
export PIP_CERT="/etc/ssl/certs/ca-certificates.crt" 
export SSL_CERT_FILE="/etc/ssl/certs/ca-certificates.crt"
export CURL_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt" 
export REQUESTS_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt"
export AWS_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt"
```
The system is now ready to use. Instead of `docker run`, launch the system with 
```bash
export OMP_THREAD_LIMIT=1
python fast_api_registration.py  \
  --cdr_key <cdr_key>   \
  --openai_api_key <openai_api_key>   \
  --minmod_username <minmod_username>  \ 
  --minmod_password <minmod_password>   \
  --cdr_callback_url http://<your ip>:9999    \ 
  --lm gpt-4o    \
  --lm_context_window 128000
```

## Contact

- Xiao Lin `xiao.lin at sri.com`
- Meng Ye
- Yi Yao