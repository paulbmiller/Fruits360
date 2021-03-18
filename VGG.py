import torchvision
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16', pretrained=True)

model.classifier = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(25088, 4096),
    torch.nn.ReLU(inplace=True),
    torch.nn.Dropout(0.5, inplace=False),
    torch.nn.Linear(4096, 8),
    torch.nn.Sigmoid()
)
print(model)

train_data = torchvision.datasets.ImageFolder(
    'large_files/fruits-360-small/Training/',
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
test_data = torchvision.datasets.ImageFolder(
    'large_files/fruits-360-small/Test/',
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))

print('Using {} training images'.format(len(train_data)))
print('Using {} test images'.format(len(test_data)))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

loss_fn = torch.nn.CrossEntropyLoss()
optim = torch.optim.RMSprop(model.classifier.parameters(), lr=0.00001)
epochs = 1

model.features.requires_grad = False
model.classifier.requires_grad = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)

model.train()
for epoch in range(epochs):
    running_loss = 0.0
    len_loader = len(train_loader)
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optim.zero_grad()
        outputs = model(inputs.to(device))
        loss = loss_fn(outputs.cpu(), labels)
        loss.backward()
        optim.step()

        # print statistics
        running_loss += loss.item()

        if i % 10 == 9:
            print('Epoch {} {}/{} loss {:.2f}'.format(epoch, i + 1, len_loader, running_loss/50))
            running_loss = 0.0

print('Finished training')

model.eval()

correct = 0
total = 0
with torch.no_grad():
    get_example=True
    for images, labels in test_loader:
        if get_example:
            images_np = images.permute(0, 2, 3, 1).numpy()
        outputs = model(images.to(device))
        predicted = torch.argmax(outputs, dim=1).cpu()
        total += labels.size(0)
        if get_example:
            pred_example = predicted
            label_example = labels
            get_example=False
        correct += (predicted == labels).sum().item()

    nrows, ncols = 4, 4
    images_np = images_np[:16]
    pred_example = pred_example[:16]
    label_example = label_example[:16]
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    for i in range(len(images_np)):
        axs[i // ncols, i % ncols].imshow(images_np[i])
        axs[i // ncols, i % ncols].set_title(
            'Predicted: {}, Real: {}'.format(train_data.classes[pred_example[i]], train_data.classes[label_example[i]]))
        axs[i // ncols, i % ncols].axis('off')
    plt.show()

print('Accuracy : {:.2f}%'.format(100 * correct / total))
