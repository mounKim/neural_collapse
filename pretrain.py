import torchvision.models as models
from torchvision import datasets
from torchvision import transforms
import torch
import torch.nn as nn

import os
import glob
import json
import shutil



def train():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    batch_size = 16
    learning_rate = 0.05
    num_epochs = 200
    
    dataset = 'cifar100'
    inp_size = 32
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    num_classes = 100
    num_trainclasses = 10
    sep_classes = ["trout", "possum", "tulip", "caterpillar", "beetle", "rocket", "otter", "plain", "aquarium_fish", "tiger"]

    # for sep_class in sep_classes:
    #     shutil.copy(f"/home/user/mjlee/NC_CL/dataset/cifar100/train/{sep_class}", "/home/user/mjlee/NC_CL/dataset/cifar100/pretrain/train")
    
    # sigma=10
    # repeat=1
    # init_cls=100
    # rnd_seeds=3

    # sep_data_G = []
    # sep_data_F = []
    
    # for rnd_seed in range(rnd_seeds):
    #     with open(f"collections/{dataset}/{dataset}_sigma{sigma}_repeat{repeat}_init{init_cls}_seed{rnd_seed+1}.json", 'r') as fp:
    #         train_list = json.load(fp)
    #         train_stream_data = train_list['stream']
    #         for data in train_stream_data:
    #             if data['klass'] in sep_classes:
    #                 sep_data_G.append(data)
    #             else:
    #                 sep_data_F.append(data)
        

    #     with open(f"collections/{dataset}/G_{dataset}_sigma{sigma}_repeat{repeat}_init{init_cls}_seed{rnd_seed}.json", 'w') as out0_file:   
    #         json.dump(sep_data_G, out0_file)
        
    #     with open(f"collections/{dataset}/F_{dataset}_sigma{sigma}_repeat{repeat}_init{init_cls}_seed{rnd_seed}.json", 'w') as out1_file:   
    #         json.dump(sep_data_F, out1_file)

    # sep_data_G = []
    # sep_data_F = []

    # with open(f"collections/{dataset}/{dataset}_val.json",'r') as val_fp:
    #     test_stream_data = json.load(val_fp)
    #     for data in test_stream_data:
    #         if data['klass'] in sep_classes:
    #             sep_data_G.append(data)
    #         else:
    #             sep_data_F.append(data)

    # with open(f"collections/{dataset}/G_{dataset}_val.json", 'w') as val1_file:   
    #         json.dump(sep_data_G, val1_file)
    # with open(f"collections/{dataset}/F_{dataset}_val.json", 'w') as val2_file:   
    #         json.dump(sep_data_F, val2_file)

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, num_trainclasses)
    
    train_transform = transforms.Compose([
                transforms.Resize((inp_size, inp_size)),
                transforms.RandomCrop(inp_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    test_transform = transforms.Compose(
        [
            transforms.Resize((inp_size, inp_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    
    correct_train = 0
    batch_losses = []
    train_losses = []
    test_losses = []

    # rnd_seed = 1
    # with open(f"collections/{dataset}/G_{dataset}_sigma{sigma}_repeat{repeat}_init{init_cls}_seed{rnd_seed}.json", 'r') as fp1:
    #     train_list = json.load(fp)
    # train_dataset = train_list['stream']

    # with open(f"collections/{dataset}/G_{dataset}_val.json", 'r') as val_file1:  
    #     test_dataset = json.load(val_file1)


    train_dataset = datasets.ImageFolder('NC_CL/dataset/cifar100/pretrain/train', train_transform)
    test_dataset = datasets.ImageFolder('NC_CL/dataset/cifar100/pretrain/test', test_transform)

    train_loader = torch.utils.data.Dataloader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.Dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for batch, target in train_loader:
            model.train()
            batch, target = batch.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = nn.CrossEntropy(output, target)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

            _, preds = torch.max(output, 1)
            correct_train += torch.sum(preds == target.data)
        
        train_losses.append( mean(batch_losses))

        model.eval()
        y_pred = []

        correct_test = 0
        with torch.no_grad():

            for batch, targets in test_loader:
                batch = batch.to(device)
                targets = targets.to(device)

                outputs = model(batch)
                loss = nn.CrossEntropy(outputs, targets)

                y_pred.extend( outputs.argmax(dim=1).cpu().numpy() )

                _, preds = torch.max(outputs, 1)
                correct_test += torch.sum(preds == targets.data)
                

        # Calculate accuracy
        train_acc = correct_train.item() / train_dataset.data.shape[0]
        test_acc = correct_test.item() / test_dataset.data.shape[0]

        print('Training accuracy: {:.2f}%'.format(float(train_acc) * 100))
        print('Test accuracy: {:.2f}%\n'.format(float(test_acc) * 100))

    torch.save(model.state_dict(), f"{dataset}_cls{num_trainclasses}_REMIND.pt")


if __name__ == '__main__':
    train()
