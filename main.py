import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.onnx
from scipy.io import savemat
from torchinfo import summary

# evaluation is used for computing the accuracy and confusion matrix
# train is used for training the model
# plotting is used for plotting the confusion matrix and accuracy graph
from evaluation import compute_confusion_matrix
from train import train_model
from plotting import plot_confusion_matrix
from cae import model as model_1

# Initialize batch_size and number of epoch
BATCH_SIZE = 32
NUM_EPOCHS = 100
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialize the path of our data set
train_dataset_path = "C:/Users/ngoth/Documents/UMKC/Gamma_NWPU-RESISC45_01/Train"
valid_dataset_path = "C:/Users/ngoth/Documents/UMKC/Gamma_NWPU-RESISC45_01/Valid"
test_dataset_path = "C:/Users/ngoth/Documents/UMKC/Gamma_NWPU-RESISC45_01/Test"

# Transform the dataset
datasets_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.RandomCrop((224, 224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(10),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

valid_datasets_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.CenterCrop((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Build train set, valid set, and test set from their respective paths
train_set = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=datasets_transforms)
valid_set = torchvision.datasets.ImageFolder(root=valid_dataset_path, transform=valid_datasets_transforms)
test_set = torchvision.datasets.ImageFolder(root=test_dataset_path, transform=valid_datasets_transforms)

# Build data loader
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)


# Build VGG model
class VGG(torch.nn.Module):
    def __init__(self, num_classes):
        super(VGG, self).__init__()
        self.num_classes = num_classes
        # convolutional layers
        self.conv_layers_1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.concat_layers = torch.nn.Sequential(torch.nn.Conv2d(256, 256, kernel_size=3, padding=1), torch.nn.ReLU(),
                                                 torch.nn.Conv2d(256, 256, kernel_size=3, padding=1))
        self.conv_layers_2 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        height, width = 7, 7
        self.avgpool = torch.nn.AdaptiveAvgPool2d((height, width))
        # self.feature_extractor = torch.nn.Linear(512 * height * width, 4096)
        # fully connected linear layers
        self.linear_layers = torch.nn.Sequential(
            torch.nn.Linear(512 * height * width, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=4096, out_features=4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=4096, out_features=self.num_classes)
        )

    def forward(self, x, x1):
        x = self.conv_layers_1(x)
        x = x1 + x
        x = self.concat_layers(x)
        x = self.conv_layers_2(x)
        x = self.avgpool(x)
        # flatten to prepare for the fully connected layers
        x = x.view(x.size(0), -1)
        # feature = self.feature_extractor(x)
        logits = self.linear_layers(x)

        return logits


# Build the model
model = VGG(num_classes=45)
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       factor=0.1,
                                                       patience=15,
                                                       mode='max',
                                                       verbose=True)


batch_loss_list, train_accuracy_list, valid_accuracy_list = train_model(
    model=model,
    model_autoencoder=model_1,
    num_epochs=NUM_EPOCHS,
    train_loader=train_loader,
    valid_loader=valid_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    device=DEVICE,
    scheduler=scheduler,
    scheduler_on='valid_acc',
    logging_interval=100)

class_dict = {0: 'airplane',
              1: 'airport',
              2: 'baseball_diamond',
              3: 'basketball_court',
              4: 'beach',
              5: 'bridge',
              6: 'chaparral',
              7: 'church',
              8: 'circular_farmland',
              9: 'cloud',
              10: 'commercial_area',
              11: 'dense_residential',
              12: 'desert',
              13: 'forest',
              14: 'freeway',
              15: 'golf_course',
              16: 'ground_track_field',
              17: 'harbor',
              18: 'industrial_area',
              19: 'intersection',
              20: 'island',
              21: 'lake',
              22: 'meadow',
              23: 'medium_residential',
              24: 'mobile_home_park',
              25: 'mountain',
              26: 'overpass',
              27: 'palace',
              28: 'parking_lot',
              29: 'railway',
              30: 'railway_station',
              31: 'rectangular_farmland',
              32: 'river',
              33: 'roundabout',
              34: 'runway',
              35: 'sea_ice',
              36: 'ship',
              37: 'snowberg',
              38: 'sparse_residential',
              39: 'stadium',
              40: 'storage_tank',
              41: 'tennis_court',
              42: 'terrace',
              43: 'thermal_power_station',
              44: 'wetland'}

mat = compute_confusion_matrix(model=model, model_autoencoder=model_1, data_loader=test_loader,
                               device=torch.device('cuda:0'))
plot_confusion_matrix(mat, class_names=class_dict.values())
plt.savefig('confusion.png')
plt.show()

# plot_training_loss(minibatch_loss_list=batch_loss_list,
#                    num_epochs=NUM_EPOCHS,
#                    iter_per_epoch=len(train_loader),
#                    results_dir=None,
#                    averaging_iterations=200)
# plt.savefig('loss.png')
# plt.show()
#
# plot_accuracy(train_accuracy_list=train_accuracy_list,
#               valid_accuracy_list=valid_accuracy_list,
#               results_dir=None)
# plt.ylim([60, 100])
# plt.savefig('accuracy.png')
# plt.show()