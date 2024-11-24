# Use an official Python runtime as a parent image
FROM python:3.10-slim

RUN apt-get update && apt-get install -y build-essential python-dev-is-python3

# Set the working directory in the container
WORKDIR /DS535_s7

# Copy the current directory contents into the container at /app
COPY . /DS535_s7

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt && echo "no-cache"
# RUN pip install torch==2.4.1
# RUN pip install pyyaml
# RUN pip install tensorboard
# RUN export PIP_DEFAULT_TIMEOUT=6000 && pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116


# Make port 80 available to the world outside this container
EXPOSE 80


# Run app.py when the container launches
CMD ["python", "main.py", "--model", "idea_lightgcn", "--cuda", "7"]