'''Train CIFAR10 / CIFAR100 with PyTorch.'''
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from opacus.validators import ModuleValidator
from opacus.accountants.utils import get_noise_multiplier
from private_vision import PrivacyEngine
import opacus
from tqdm import tqdm
import warnings
import timm

def prepare(args):
    device=torch.device("cuda")

    # Data
    print('==> Preparing data..')

    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    if args.cifar_data=='CIFAR10':
      trainset = torchvision.datasets.CIFAR10(
        root='../../data', train=True, download=True, transform=transform_train)
      testset = torchvision.datasets.CIFAR10(
        root='../../data', train=False, download=True, transform=transform_test)
    elif args.cifar_data=='CIFAR100':
      trainset = torchvision.datasets.CIFAR100(
        root='../../data', train=True, download=True, transform=transform_train)
      testset = torchvision.datasets.CIFAR100(
        root='../../data', train=False, download=True, transform=transform_test)
 
    if 'opacus' in args.mode:
      trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.bs, shuffle=True, num_workers=2)
    else:
      trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.mini_bs, shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    print('==> Building model..', args.model, '  mode ', args.mode)
    NUM_CLASSES=10 if args.cifar_data=='CIFAR10' else 100

    net = timm.create_model(args.model,pretrained=args.pretrained,num_classes=NUM_CLASSES)
    net = ModuleValidator.fix(net)
    net = net.to(device)

    if 'convit' in args.model:
        for name,param in net.named_parameters():
            if 'attn.gating_param' in name:
                param.requires_grad=False
    if 'beit' in args.model:
        for name,param in net.named_parameters():
            if 'gamma_' in name or 'relative_position_bias_table' in name or 'attn.qkv.weight' in name or 'attn.q_bias' in name or 'attn.v_bias' in name:
                param.requires_grad=False


    for name,param in net.named_parameters():
        if 'cls_token' in name or 'pos_embed' in name:
            param.requires_grad=False


    print('number of parameters: ', sum([p.numel() for p in net.parameters()]))

        
    if "ghost" in args.mode:
        criterion = nn.CrossEntropyLoss(reduction="none")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    n_acc_steps = args.bs // args.mini_bs

    if 'ghost' in args.mode:
        sigma = get_noise_multiplier(
                target_epsilon = args.eps,
                target_delta = 1e-5,
                sample_rate = args.bs/len(trainset),
                epochs = args.epochs,
                accountant = "gdp"
            )
        privacy_engine = PrivacyEngine(
            net,
            batch_size=args.bs,
            sample_size=len(trainloader.dataset),
            noise_multiplier=sigma,
            epochs=args.epochs,
            max_grad_norm=args.grad_norm,
            ghost_clipping=True,
            mixed='mixed' in args.mode
        )
        privacy_engine.attach(optimizer)
    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            if args.mode=='non-private':
                loss.backward()
                if ((batch_idx + 1) % n_acc_steps == 0) or ((batch_idx + 1) == len(trainloader)):
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                if ((batch_idx + 1) % n_acc_steps == 0) or ((batch_idx + 1) == len(trainloader)):
                    optimizer.step(loss=loss)
                    optimizer.zero_grad()
                else:
                    optimizer.virtual_step(loss=loss)
            train_loss += loss.mean().item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print(epoch, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    def test(epoch):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                loss = loss.mean()
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print(epoch, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return args.epochs, train, test
    
def main(epochs, trainf, testf, args):
    for epoch in range(epochs):
        trainf(epoch)
        testf(epoch)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=1, type=int,
                        help='numter of epochs')
    parser.add_argument('--bs', default=1000, type=int, help='batch size')
    parser.add_argument('--eps', default=2, type=float, help='target epsilon')
    parser.add_argument('--grad_norm', '-gn', default=0.1,
                        type=float, help='max grad norm')
    parser.add_argument('--mode', default='ghost_mixed',
                        type=str, help='unfold, unfold-flex, opacus or non-private')
    parser.add_argument('--model', default='resnet18', type=str)
    parser.add_argument('--mini_bs', type=int, default=50)
    parser.add_argument('--pretrained', type=int, default=1)
    parser.add_argument('--cifar_data', type=str, default='CIFAR10')

    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    epochs, trainf, testf = prepare(args)
    main(epochs, trainf, testf, args)
