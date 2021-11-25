import pandas as pd
import torch
from torchvision.io import read_image
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
from torch import optim
import datetime
import os
from os.path import dirname, abspath

#Create custom pytorch dataset
class CustomImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'imgPath']
        image = read_image(img_path).float()
        label = self.df.loc[idx, 'Class']
        if self.transform:
            image = self.transform(image)
        return image,label

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.image_features_ = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(),
            nn.Conv2d(16, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(),
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(),
            nn.Conv2d(256, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.linear_features_ = nn.Sequential(
            nn.Linear(128*6*6, 128*6*6*2),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128*6*6*2, 128*6*8),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128*6*8, 4),
        )
    def forward(self, x):
        x = self.image_features_(x)
        x=x.view(-1, 128*6*6)
        x=self.linear_features_(x)
        return x

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range(1,n_epochs+1):
        loss_train = 0.0
        for imgs,labels in train_loader:
            imgs=imgs.to(device)
            labels = labels.to(device)
            output=model(imgs)
            
            loss=loss_fn(output, torch.tensor(labels))
            
            #L2 Regularization
            l2_lambda=0.001
            l2_norm=sum(p.pow(2).sum() for p in model.parameters())
            loss=loss+l2_lambda*l2_norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch,
                                                         loss_train / len(train_loader)))

def validate(model, train_loader, val_loader):
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs=imgs.to(device)
                labels=labels.to(device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())
        print("Accuracy {}: {:.2f}".format(name , correct / total))

if __name__=='__main__':
        

    #Load Dataset
    imgPath=[]
    imgClass=[]

    for directory in os.listdir(dirname(dirname(abspath(__file__)))+'/hand_gestures_dataset'):
        for file in os.listdir(dirname(dirname(abspath(__file__)))+'/hand_gestures_dataset/'+directory):
            imgPath.append(dirname(dirname(abspath(__file__)))+'/hand_gestures_dataset/'+directory+'/'+file)
            imgClass.append(int(directory))
            
    df=pd.DataFrame(columns=['imgPath','Class'])
    df['imgPath']=imgPath
    df['Class']=imgClass

    #create Pytorch dataset 
    data=CustomImageDataset(df,transform=transforms.Compose([transforms.Resize(256),
                                                            transforms.RandomCrop((224,224)),
                                                            transforms.Normalize(
                                                                        (177.3580, 169.3307, 160.4894),
                                                                        (37.1419, 42.7550, 50.8975))
                                                            ]))
                                                            
    train_data,val_data=torch.utils.data.random_split(data,[600,223])
    train_data_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_data_loader = DataLoader(val_data, batch_size=64, shuffle=True)

    #Switching to GPU if available for training and model fitting

    if torch.cuda.is_available():
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')
        
    model = NeuralNetwork().to(device)

    #optimizer and loss function for the model
    optimizer=optim.Adam(model.parameters(),1e-4)
    loss_fn=nn.CrossEntropyLoss()

    #train the model
    training_loop(
    n_epochs = 100,
    optimizer = optimizer,
    model = model,
    loss_fn = loss_fn,
    train_loader = train_data_loader)

    #validate model accuracy
    validate(model, train_data_loader, val_data_loader)

    #Finally save the Deep Neural Network model into diretory for future predictions
    torch.save(model.state_dict(), dirname(dirname(abspath(__file__)))+'/model/'+'hand_gesture.pt')