import torch
import torchvision
import albumentations as A
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2


class outDetector():

    def __init__(self):

        self.test_transform = A.Compose([
                A.Resize(384, 384),
                A.Normalize(),
                ToTensorV2()
            ])

        use_cuda = torch.cuda.is_available()
        self.DEVICE = torch.device("cuda" if use_cuda else "cpu")
        self.model = torchvision.models.resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(512, 3)
        self.model = self.model.to(self.DEVICE)


        model_path = '/opt/ml/project/final-project-level3-cv-12/dataset/outputs/depth_008_9122.pth'
        ckpt = torch.load(model_path, map_location=self.DEVICE)
        self.model.load_state_dict(ckpt)


    def predict(self, img, bbox):
        
        (x, y, w, h) = bbox
        cropped = img[y:y+h, x:x+w]
        cropped = self.test_transform(image=cropped)['image']
        if len(cropped.shape) == 3:
            cropped = cropped.unsqueeze(0)
        cropped = cropped.to(self.DEVICE)

        pred = self.model(cropped)
        pred = pred.argmax(dim=-1)
        
        return pred.detach().cpu()