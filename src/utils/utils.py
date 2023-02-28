import matplotlib.pyplot as plt
import torch 
from torch import nn

def plot_sample_image(training_data, labels_map, grid_size = (8,8)):

    figure = plt.figure(figsize=grid_size)
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label, _ = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.permute(1,2,0), cmap="gray") 
    plt.show()



def loss_fn(y_pred, y_true, model_type):
    """"
    Define loss based on model struccture
    """
    
    #define loss based on model
    if model_type=="resnet":
        bce_fn = nn.CrossEntropyLoss()
        loss = bce_fn(y_pred, y_true)
    elif model_type=="AE":
        bce_fn = nn.MSELoss()
        loss = bce_fn(y_pred, y_true)
    return loss

def train_inferencing(data_loader, model, optimizer, DEVICE):
    """"
    get the output from trained model
    
    input   : Dataloader
    Output  : Calculated loss
    """

    for image, target, _ in data_loader:
        #image = image.to(DEVICE)
        image, target = image.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = loss_fn(output, target, "resnet")
        loss.backward()
        optimizer.step()
        return  loss
    
def test_inferencing(data_loader, model, optimizer, DEVICE):
    """"
    get the output from trained model
    
    input   : Dataloader
    Output  : Calculated loss
    """

    for image, target, _ in data_loader:
        #image = image.to(DEVICE)
        image, target = image.to(DEVICE), target.to(DEVICE)
        output = model(image)
        loss = loss_fn(output, target, "resnet")
        return  loss


def loss_plot(batch_losses,vlosses, save_path):
    """
    ploting validation vs train error
    
    """
    
    fig, ax = plt.subplots(1, 1, figsize = (9, 5))
    plt.title("Train-Validation Accuracy")
    plt.plot(batch_losses, label='train')
    plt.plot(vlosses, label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    ax.legend(loc='best')
    fig.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()



# @torch.no_grad()
# def validation(model, train_dataloader, loss_fn):
#     losses = []
#     model.eval()
#     for image, __, _ in train_dataloader:
#         image = image.to(DEVICE)
#         # image_croped   = torchvision.transforms.CenterCrop([68, 68])(image)
#         output = model(image)
#         loss = loss_fn(output, image)
#         losses.append(loss.item())
        
#     return np.array(losses).mean()


