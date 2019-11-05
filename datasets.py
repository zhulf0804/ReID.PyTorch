from torchvision import datasets, transforms
import os
import torch


transform_train_list = [
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
        transforms.Resize((256,128), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((256,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]
transform_val_list = [
        transforms.Resize(size=(256,128),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]
data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
    'train_all': transforms.Compose(transform_train_list)
}

def get_train_datasets(data_dir, batch_size):
    train_image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'train_all']}
    train_dataloaders = {x: torch.utils.data.DataLoader(train_image_datasets[x], batch_size=batch_size,
                                             shuffle=False, num_workers=8) for x in ['train', 'val', 'train_all']}
    dataset_sizes = {x: len(train_image_datasets[x]) for x in ['train', 'val', 'train_all']}
    class_names = train_image_datasets['train'].classes
    return train_image_datasets, train_dataloaders, dataset_sizes, class_names
