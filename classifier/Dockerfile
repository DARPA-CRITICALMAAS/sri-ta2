FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

RUN apt update
RUN apt install -y htop screen git ffmpeg vim

RUN pip install --upgrade pip
RUN pip install scipy
RUN pip install matplotlib
RUN pip install scikit-learn
RUN pip install pandas
RUN pip install einops

RUN pip install transformers
RUN pip install accelerate
RUN pip install tokenizers
RUN pip install sentencepiece
RUN pip install protobuf==3.19
RUN pip install openpyxl

RUN pip install pdf2image
RUN pip install pytesseract
RUN pip install opencv-python
RUN conda install tesseract -y
RUN conda install poppler -y

RUN pip install openai
RUN pip install accelerate
RUN pip install backoff

RUN pip install peft
RUN conda install nvidia/label/cuda-11.8.0::cuda -y
RUN pip install flash-attn



# copy stuff
WORKDIR /work
COPY cache cache
COPY taxonomy taxonomy
COPY tessdata tessdata
COPY util util
COPY *.py ./
COPY *.csv ./
COPY *.pdf ./
COPY *.json ./
# CMD ["/bin/bash"]
ENTRYPOINT python run.py "$@"
