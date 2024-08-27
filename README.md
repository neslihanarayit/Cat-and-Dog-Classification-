# Dogs vs Cats Classifier

This project aims to classify images of cats and dogs using a Convolutional Neural Network (CNN) built with PyTorch. The project includes Docker setup, training and inference scripts, and a Gradio interface for model deployment.

## Project Structure

- `dogs-vs-cats/`: Directory containing the dataset.
  - `train.zip`: Compressed training dataset.
  - `test1.zip`: Compressed test dataset.
- `main.py`: Python script for training the model.
- `models/`: Directory containing saved models in `.pth` format.
- `runs/`: Directory containing TensorBoard logs.
- `docker/`: Directory containing Docker and Docker Compose files.
- `app.py`: Gradio interface for model inference.
- `requirements.txt`: List of Python dependencies.
- `Dockerfile`: Docker configuration for the training environment.
- `docker-compose.yml`: Compose file for setting up the entire environment.


## Setup and Installation

1. Clone the repository:


2. Preparing the Datasets:

    - Unzip the datasets and organize the data

        python main.py --organize-data ??

    

## Usage

- Training: Use main.py for training the model.
- Inference: Use app.py with Gradio to deploy the model.