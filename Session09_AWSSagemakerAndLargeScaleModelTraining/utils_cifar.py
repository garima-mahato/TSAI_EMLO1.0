import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

classes = ('beaver','dolphin','otter','seal','whale','aquarium fish','flatfish','ray','shark','trout','orchids','poppies','roses','sunflowers','tulips','bottles','bowls','cans','cups','plates','apples','mushrooms','oranges','pears','sweet peppers','clock','computer keyboard','lamp','telephone','television','bed','chair','couch','table','wardrobe','bee','beetle','butterfly','caterpillar','cockroach','bear','leopard','lion','tiger','wolf','bridge','castle','house','road','skyscraper','cloud','forest','mountain','plain','sea','camel','cattle','chimpanzee','elephant','kangaroo','fox','porcupine','possum','raccoon','skunk','crab','lobster','snail','spider','worm','baby','boy','girl','man','woman','crocodile','dinosaur','lizard','snake','turtle','hamster','mouse','rabbit','shrew','squirrel','maple','oak','palm','pine','willow','bicycle','bus','motorcycle','pickup truck','train','lawn-mower','rocket','streetcar','tank','tractor')


def _get_transform():
    return transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    

def get_train_data_loader():
    transform = _get_transform()
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
    return torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

    
def get_test_data_loader():
    transform = _get_transform()
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)
    return torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
    

# function to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))