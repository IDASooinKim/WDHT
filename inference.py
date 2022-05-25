import torch
import torchvision
import torchvision.transforms as T

from PIL import Image
from utils.DataGenerator import ImageDataset
from utils.ArgParse import get_args
from model.WDHT import WDHT, LoadBackbone
from torchsummary import summary
from glob import glob


data_path = 'result_5_epoch/model_dict.pt'

def load_img(img_path):
    
    img = Image.open(img_path)
    transform = T.Resize((224,224))
    tensorize = T.ToTensor()
    resized_img = transform(img)
    tensorize_img = tensorize(resized_img)
    tensorize_img = tensorize_img[None,:]
    return tensorize_img


if __name__ == '__main__':
    
    parser = get_args()
    
    resnet50 = LoadBackbone('resnet50', True)

    model = WDHT(resnet50, parser.hash_size, parser.class_num)
    summary(model, (3,224,224), 64, 'cpu')
    model.load_state_dict(torch.load('result_5_epoch/model.pt'))
    #model  = torch.load(data_path)
    model.eval()

    for img_path in glob('test_data/*.jpg'):
        img = load_img(img_path)
        word_label, hash_tag = model(img)
        
        print(img_path)
        print("hash is : {}".format(word_label))
        print("\n")
