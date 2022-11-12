import os
import argparse
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from time import time


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train_epoch(train_loader, optimizer, criterion, lr_scheduler, model, world_size):
        model.train()

        train_running_loss = 0.0
        train_running_acc = 0.0    
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(data)        
            preds = torch.max(output, 1)[1]

            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()
            train_running_acc += torch.eq(preds, target).sum().item()

        lr_scheduler.step()
        train_loss_value = train_running_loss/ (len(train_dataset) / world_size)
        train_acc_value = train_running_acc/ (len(train_dataset) / world_size)

        return train_loss_value, train_acc_value

def valid_epoch(valid_loader, criterion, model, world_size):
    model.eval()

    valid_running_loss = 0.0
    valid_running_acc = 0.0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(valid_loader):
           
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            outputs = model(data)
            preds = torch.max(outputs, 1)[1]

            loss = criterion(outputs, target)
            
            valid_running_loss += loss.item() 
            valid_running_acc += torch.eq(preds, target).sum().item()

    valid_loss_value = valid_running_loss/ (len(valid_dataset) / world_size)
    valid_acc_value = valid_running_acc/ (len(valid_dataset) / world_size)

    return valid_loss_value, valid_acc_value

if __name__ == '__main__':
    t1=time()
    file = open(f"logs/{time()}.csv","a+")
    parser = argparse.ArgumentParser()
    # parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    dist.init_process_group(backend='nccl')
    dist.barrier()
    rank = dist.get_rank()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    t2=time()
    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    valid_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])) # CIFAR10
    
    train_sampler = DistributedSampler(train_dataset)
    valid_sampler = DistributedSampler(valid_dataset)

    train_loader = DataLoader(train_dataset, batch_size=256,
                              sampler=train_sampler, **kwargs)#, prefetch_factor=2, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=256,
                              sampler=valid_sampler, **kwargs)#, prefetch_factor=2, num_workers=4)
    t1=time()
    print(local_rank, "data loading cost", t1-t2,"s")
    if use_cuda:
        # device = torch.device("cuda", dist.get_rank()) 
        device = torch.device("cuda", local_rank) # 
    else:
        device = torch.device("cpu")

    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18')
    # model.fc = nn.Sequential(nn.Linear(in_features=512, out_features=128), nn.LeakyReLU(),
    #                          nn.Dropout(0.5), nn.Linear(128, 10))
    # model = model.to(device)
    model = Net().to(device)
    print(local_rank, "Model created.", dist.get_rank(), local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
    #                                                   output_device=args.local_rank)
    print("Model initialized.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    lr_scheduler_values = lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.1)
    criterion = nn.CrossEntropyLoss().to(device)
    t2=time()
    print(local_rank, "model initailizing cost", t2-t1, "s")
    file.write(f"{local_rank},model initialization,{t2-t1}\n")
    num_epochs = 50
    t3=time()
    for epoch in range(num_epochs):
        t1=time()
        train_sampler.set_epoch(epoch)
        valid_sampler.set_epoch(epoch)
        
        
        print(local_rank,"epoch begin cost",t1-t3,"s")
        train_loss_value, train_acc_value = train_epoch(train_loader, optimizer, criterion, lr_scheduler_values, model, world_size)
        t2=time()
        file.write(f"{local_rank},train {epoch},{t2-t1}\n")
        print(local_rank,"train cost",t2-t1,"s")
        valid_loss_value, valid_acc_value = valid_epoch(valid_loader, criterion, model, world_size)    
        t3=time()
        # print(local_rank,"valid cost",t3-t2,"s")
        print("Train_local_rank: {} Train_Epoch: {}/{} Training_Loss: {} Training_acc: {:.2f}\
                   ".format(local_rank, epoch, num_epochs-1, train_loss_value, train_acc_value))

        print("Valid_local_rank: {} Valid_Epoch: {}/{} Valid_Loss: {} Valid_acc: {:.2f}\
                   ".format(local_rank, epoch, num_epochs-1, valid_loss_value, valid_acc_value))
        file.write(f"{local_rank},epoch {epoch},{t3-t1}\n")
        print('--------------------------------')
    file.close()
    print("finished.")
