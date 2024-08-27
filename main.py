import os
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from timeit import default_timer as timer


class CreateDataloaders():
    def __init__(self, train_dir: str,
                test_dir:str,
                validation_dir:str,
                train_transform: transforms.Compose,
                test_transform: transforms.Compose,
                validation_transform: transforms.Compose,
                batch_size = 32,
                seed = None,
                num_workers = None) -> None:
        
        self.test_dir = test_dir
        self.train_dir = train_dir
        self.validation_dir = validation_dir
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.validation_transform = validation_transform        
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers if num_workers else os.cpu_count()


    def load_train_data(self):
        train_dataset = datasets.ImageFolder(root=self.train_dir, transform=self.train_transform)
        class_names = train_dataset.classes
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        return train_loader, class_names
    
    def load_validation_data(self):
        valid_dataset = datasets.ImageFolder(root=self.validation_dir, transform=self.validation_transform)
        class_names = valid_dataset.classes
        valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
        return valid_loader, class_names
    
    def load_test_data(self):
        test_dataset = datasets.ImageFolder(root=self.test_dir, transform=self.test_transform)
        class_names = test_dataset.classes
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
        return test_loader, class_names
    
    def load_data(self):
        train_dataset = datasets.ImageFolder(root=self.train_dir, transform=self.train_transform)
        valid_dataset = datasets.ImageFolder(root=self.validation_dir, transform=self.validation_transform)
        test_dataset = datasets.ImageFolder(root=self.test_dir, transform=self.test_transform)

        class_names = train_dataset.classes
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

        return train_loader, valid_loader, test_loader, class_names


        

class TinyVGG_4(nn.Module):
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


class TrainModel:
    def __init__(self, model, train_dataloader, test_dataloader, validation_dataloader, optimizer, loss_fn, epochs, device, writer, l1_lambda=0.001, scheduler = None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.valid_dataloader = validation_dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.device = device
        self.writer = writer
        self.l1_lambda = l1_lambda
        self.scheduler = scheduler

    def calculate_metrics(self, data_loader):
        self.model.eval()  # Set the model to evaluation mode
        self.model.to(self.device)
        
        all_preds = []
        all_labels = []

        with torch.no_grad():  # Disable gradient computation
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)  # Move to device
                outputs = self.model(images)  # Forward pass
                
                # Convert to predicted labels
                _, preds = torch.max(outputs, 1)  # For multi-class classification
                
                # Collect predictions and labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate Precision, Recall, and F1 Score
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        return accuracy, precision, recall, f1

    def train(self):
        
        self.model.to(self.device)
        train_losses = []
        test_losses = []

        for epoch in range(self.epochs):
            self.model.train()  # Set model to training mode
            running_loss = 0.0
            
            for images, labels in self.train_dataloader:
                images, labels = images.to(self.device), labels.to(self.device)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                # Forward pass
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                
                
                # L1 regularization
                l1_reg = torch.tensor(0., device=self.device)  # Initialize on the correct device
                for param in self.model.parameters():
                    l1_reg = l1_reg + torch.sum(torch.abs(param))  # Avoid in-place operation
                loss = loss + self.l1_lambda * l1_reg
                
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item() * images.size(0)

            
            epoch_train_loss = running_loss / len(self.train_dataloader.dataset)
            train_losses.append(epoch_train_loss)
            self.writer.add_scalar("Loss/train", epoch_train_loss, epoch)

            # Evaluate on the test set
            self.model.eval()
            test_running_loss = 0.0

            with torch.no_grad():
                for images, labels in self.test_dataloader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)
                    test_running_loss += loss.item() * images.size(0)

                    

            epoch_test_loss = test_running_loss / len(self.test_dataloader.dataset)
            test_losses.append(epoch_test_loss)
            self.writer.add_scalar("Loss/test", epoch_test_loss, epoch)
            
            # Calculate and log metrics for the test set
            accuracy, precision, recall, f1 = self.calculate_metrics(self.test_dataloader)
            self.writer.add_scalar("Precision/test", precision, epoch)
            self.writer.add_scalar("Recall/test", recall, epoch)
            self.writer.add_scalar("F1/test", f1, epoch)
            self.writer.add_scalar("Accuracy/test", accuracy, epoch)


            print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {epoch_train_loss:.4f}, Test Loss: {epoch_test_loss:.4f}')
            print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

            if self.scheduler:
                self.scheduler.step(epoch_test_loss)

            # Print the learning rate after adjustment
            print(f"Learning Rate: {self.scheduler.optimizer.param_groups[0]['lr']}")

        return train_losses, test_losses


class DataTransformations():
    def __init__(self):
        # Example data transformations
        self.train_transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomRotation(0.4),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ])

        # Create testing transform (no data augmentation)
        self.test_transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ])
        # Create testing transform (no data augmentation)
        self.validation_transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ])
    def get_transforms(self):
        return self.train_transforms, self.test_transforms, self.validation_transforms

def main_part():

    image_path = Path("dogs-vs-cats/")
    train_dir = image_path / "train"
    test_dir = image_path / "test"
    validation_dir = image_path / "validation"

    # Set a seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Example data transformations

    data_transforms = DataTransformations()

    train_transforms, test_transforms, validation_transforms = data_transforms.get_transforms()

    data_loader = CreateDataloaders(train_dir= train_dir, test_dir= test_dir,
                                    validation_dir=validation_dir, train_transform= train_transforms,
                                    test_transform= test_transforms, validation_transform= validation_transforms,
                                    batch_size= 32, seed= seed)


    train_loader, validation_loader, test_loader, class_name = data_loader.load_data()

    writer = SummaryWriter()
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize model, optimizer, and loss function
    model_7 = TinyVGG_4(input_shape=(3, 64, 64), hidden_units=130, output_shape=2).to(device)
    optimizer = optim.AdamW(params=model_7.parameters(), lr=1e-3, weight_decay=1e-1)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode= 'min', factor= 0.1)
    loss_fn = nn.CrossEntropyLoss()
    # Set random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # Set number of epochs
    NUM_EPOCHS = 50
    # Start the timer
    start_time = timer()
    # Train model_2
    train_model = TrainModel(model= model_7,
                        train_dataloader= train_loader,
                        test_dataloader=test_loader,
                        validation_dataloader=validation_loader,
                        optimizer=optimizer,
                        loss_fn=loss_fn, 
                        epochs=NUM_EPOCHS,
                        device=device,
                        writer=writer,
                        l1_lambda=0.00001,
                        scheduler=scheduler
                        )
    train_losses, test_losses = train_model.train()

    writer.flush()
    writer.close()
    # End the timer and print out how long it took
    end_time = timer()
    print(f"Total training time: {end_time - start_time:.3f} seconds")
    # Plot loss curves
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Train Loss')
    plt.plot(range(1, NUM_EPOCHS + 1), test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss Curves')
    plt.legend()
    plt.show()


main_part()