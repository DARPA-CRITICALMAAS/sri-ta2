FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

RUN apt update
RUN apt install -y htop screen git ffmpeg vim tesseract-ocr poppler-utils
RUN pip install --upgrade pip
RUN pip install backoff openai tiktoken pandas pytesseract pdf2image


# copy stuff
WORKDIR /work
COPY minmod minmod
COPY taxonomy taxonomy
COPY util util
COPY *.py ./
COPY *.json ./
# CMD ["/bin/bash"]
ENTRYPOINT python main.py "$@"