import torch 
import torch.nn as nn
import torchvision.transforms as transforms

model_perceptual = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
model_perceptual = model_perceptual.features
model_perceptual.cuda()

embedding_transform = transforms.Resize((224,224))


def extract_embedding(image):
    
    embedding = []
    #image = embedding_transform(image)
    #print(image.size())
    image = torch.cat([image, image, image], dim=1)
    for n in range(5):
        image = model_perceptual[n](image)
    embedding = image    
    return embedding

def perceptual_loss(embedding_orig, embedding_out):
    criterion = torch.nn.MSELoss().cuda()
    loss = criterion(embedding_orig, embedding_out)
    return loss
    
    