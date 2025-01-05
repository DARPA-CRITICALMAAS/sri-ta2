FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

RUN apt update
RUN apt install -y htop screen git ffmpeg vim tesseract-ocr poppler-utils
RUN pip install --upgrade pip
RUN pip install backoff openai tiktoken pandas pytesseract pdf2image
RUN pip install flask flask-cors httpx fastapi uvicorn ngrok


# copy stuff
WORKDIR /work
COPY minmod minmod
COPY taxonomy taxonomy
COPY util util
COPY *.py ./
COPY *.json ./
# CMD ["/bin/bash"]

# USGS DOI SSL
COPY DOIRootCA2.crt /usr/local/share/ca-certificates
RUN chmod 644 /usr/local/share/ca-certificates/DOIRootCA2.crt && \
    update-ca-certificates
# you probably don't need all of these, but they don't hurt 
ENV PIP_CERT="/etc/ssl/certs/ca-certificates.crt" \
    SSL_CERT_FILE="/etc/ssl/certs/ca-certificates.crt" \
    CURL_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt" \
    REQUESTS_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt" \
    AWS_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt"

ENV OMP_THREAD_LIMIT=1

ENTRYPOINT python fast_api_registration.py "$@"