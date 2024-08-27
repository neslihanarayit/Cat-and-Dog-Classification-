import gradio as gr
import torch
import pandas as pd
from torchvision import transforms
import torch
import torch.nn as nn

class TinyVGG_3(nn.Module):
    def __init__(self, input_shape: tuple, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.dropout = nn.Dropout(p=0.5)  # Dropout with 50% probability

        # Calculate the flattened size
        self.flattened_size = self._get_flattened_size(input_shape, hidden_units)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 2560),
            nn.ReLU(),
            self.dropout,  # Apply dropout after the first linear layer
            nn.Linear(2560, output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

    def _get_flattened_size(self, input_shape, hidden_units):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)  # Create a dummy input
            x = self.conv_block_1(dummy_input)
            x = self.conv_block_2(x)
            return x.numel()  # Return the number of elements in the flattened tensor

class TinyVGG_4(nn.Module):
    def __init__(self, input_shape: tuple, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(32, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.dropout = nn.Dropout(p=0.5)  # Dropout with 50% probability

        # Calculate the flattened size
        self.flattened_size = self._get_flattened_size(input_shape, hidden_units)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 2560),
            nn.ReLU(),
            self.dropout,  # Apply dropout after the first linear layer
            nn.Linear(2560, output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

    def _get_flattened_size(self, input_shape, hidden_units):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)  # Create a dummy input
            x = self.conv_block_1(dummy_input)
            x = self.conv_block_2(x)
            return x.numel()  # Return the number of elements in the flattened tensor

class TinyVGG_5(nn.Module):
    def __init__(self, input_shape: tuple, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)       
                            )
        self.dropout = nn.Dropout(p=0.3)  # Dropout with 50% probability

        # Calculate thiddenlayer-64-lrdecay-l1-0.00001he flattened size
        self.flattened_size = self._get_flattened_size(input_shape, hidden_units)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 2560),
            nn.ReLU(),
            self.dropout,  # Apply dropout after the first linear layer
            nn.Linear(2560, output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.classifier(x)
        return x

    def _get_flattened_size(self, input_shape, hidden_units):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)  # Create a dummy input
            x = self.conv_block_1(dummy_input)
            x = self.conv_block_2(x)
            x = self.conv_block_3(x)
            return x.numel()  # Return the number of elements in the flattened tensor

#set models by uploading them inside models directory
model_1 = TinyVGG_5(input_shape=(3,64,64), hidden_units= 132, output_shape= 2)

model_2 = TinyVGG_4(input_shape=(3,64,64), hidden_units= 124, output_shape= 2)

model_3 = TinyVGG_3(input_shape=(3,64,64), hidden_units= 32, output_shape= 2)

model_1.load_state_dict(torch.load("models/wd1e-1_3block_hiddenunit132_26august.pth", weights_only=True))
model_2.load_state_dict(torch.load("models/3block_lrdecay_124hdlayer_l1andl2.pth", weights_only=True))
model_3.load_state_dict(torch.load("models/l1-l1-lr_decay-wd-augmn.pth", weights_only= True))

# necessary image transform to get the results
test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


labels = ["Cat", "Dog"]

def classify_image(inp, model):
    # set the device to gpu for better evaluation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    inp = test_transforms(inp).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(inp)
        # use softmax to get the probability
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        # set the dictionary for both labels
        percentages = {labels[i]: float(probabilities[0][i]) for i in range(probabilities.size(1))}
    model.to('cpu')  # Move model back to CPU after processing
    return percentages

# Create the buttons
def classify_image_1(image):
    return classify_image(image, model_1)

def classify_image_2(image):
    return classify_image(image, model_2)

def classify_image_3(image):
    return classify_image(image, model_3)
# accuracy per epoch data frame
model_1_accuracy = pd.read_csv("model_1_accuracy.csv")  
model_2_accuracy = pd.read_csv("model_2_accuracy.csv")  
model_3_accuracy = pd.read_csv("model_3_accuracy.csv")  
model_1_losstest = pd.read_csv("model_1_losstest.csv")  
model_2_losstest = pd.read_csv("model_2_losstest.csv")  
model_3_losstest = pd.read_csv("model_3_losstest.csv")  

with gr.Blocks() as demo:
    # heading
    gr.Markdown("")
    gr.Markdown("<h1 style='text-align: center;'>Cat vs Dog Classifier</h1>")
    gr.Markdown("<h2 style='text-align: center;'>Upload an image of cat or dog to see the results</h2>")
    
    img_input = gr.Image(type="pil")
    output = gr.Label(num_top_classes=2)
    
    with gr.Row():
        btn_1 = gr.Button("Classify with Model 1")
        btn_2 = gr.Button("Classify with Model 2")
        btn_3 = gr.Button("Classify with Model 3")

    gr.Examples(
        examples=[
            ["dogs-vs-cats/test1/118.jpg"], 
            ["dogs-vs-cats/test1/20.jpg"], 
            ["tarcin.jpeg"], 
            ["dewewdwe.jpeg"],
            ['dogs-vs-cats/test1/12416.jpg']
        ],
        inputs=img_input
    )

    gr.Markdown("<h2 style='text-align: center; color: orange;'> Following plots shows the accuracy for models after each epoch in learning process </h2>")
    with gr.Row():
        gr.Markdown("<h2 style='text-align: center; color: red;'> Model 1:</h2>")
        gr.Markdown("<h2 style='text-align: center; center; color: red;'> Model 2:</h2>")
        gr.Markdown("<h2 style='text-align: center; center; color: red;'> Model 3:</h2>")
    with gr.Row():
        gr.Markdown("<h3 style='text-align: center;'>TinyVgg with 3 block and each layers have 132 hidden units. Might have overfit </h3>")
        gr.Markdown("<h3 style='text-align: center;'>TinyVgg with 3 block and each layers have 124 hidden units. Can be improved </h3>")
        gr.Markdown("<h3 style='text-align: center;'>TinyVgg with 2 block and each layers have 32 hidden units. Underfitting. </h3>")

    with gr.Row():
        gr.Markdown("<h3 style='text-align: center;'> Test Accuracy: 93.84%  </h3>")
        gr.Markdown("<h3 style='text-align: center;'> Test Accuracy: 96.13%  </h3>")
        gr.Markdown("<h3 style='text-align: center;'> Test Accuracy: 95.82% </h3>")
    with gr.Row():
        gr.Markdown("<h3 style='text-align: center;'> Train Accuracy: 98.33% </h3>")
        gr.Markdown("<h3 style='text-align: center;'> Train Accuracy: 94.33%  </h3>")
        gr.Markdown("<h3 style='text-align: center;'> Train Accuracy: 94.38% </h3>")

    with gr.Row():
        gr.Markdown("<h3 style='text-align: center;'> Validation Accuracy for each Epoch </h3>")
    with gr.Row():
        gr.LinePlot(model_1_accuracy, x= "Epoch", y= "Accuracy")
        gr.LinePlot(model_2_accuracy, x= "Epoch", y= "Accuracy")
        gr.LinePlot(model_3_accuracy, x= "Epoch", y= "Accuracy")

    with gr.Row():
        gr.Markdown("<h3 style='text-align: center;'> Validation Loss for each Epoch </h3>")
    with gr.Row():
        gr.LinePlot(model_1_losstest, x= "Epoch", y= "Loss/Validation")
        gr.LinePlot(model_2_losstest, x= "Epoch", y= "Loss/Validation")
        gr.LinePlot(model_3_losstest, x= "Epoch", y= "Loss/Validation")


    btn_1.click(fn=classify_image_1, inputs=img_input, outputs=output)
    
    btn_2.click(fn=classify_image_2, inputs=img_input, outputs=output)

    btn_3.click(fn=classify_image_3, inputs=img_input, outputs=output)

demo.launch(server_name="0.0.0.0")