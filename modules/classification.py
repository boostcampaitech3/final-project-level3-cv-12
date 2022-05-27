import torch
import torch.nn as nn
import torchvision
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO


class TestDataset(Dataset):
    def __init__(self, img, bboxes, transform):
        self.transform = transform
        self.img = img
        self.bboxes = bboxes

    def __getitem__(self, idx):
        (x, y, w, h) = self.bboxes[idx]
        cropped = self.img[y:y+h, x:x+w]

        if self.transform is not None:
            img_transform = self.transform(image=cropped)
            img_transform = img_transform['image']
        else:
            img_transform = cropped

        return img_transform

    def __len__(self):
        return len(self.bboxes)
        


class Classifier():
    def __init__(self):

        self.test_transform = A.Compose([
                A.Resize(384, 384),
                A.Normalize(),
                ToTensorV2()
            ])

        use_cuda = torch.cuda.is_available()
        self.DEVICE = torch.device("cuda" if use_cuda else "cpu")
        self.model = torchvision.models.resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(512, 158)
        self.model = self.model.to(self.DEVICE)

        model_path = '/opt/ml/project/final-project-level3-cv-12/weights/classification_all.pth'
        ckpt = torch.load(model_path, map_location=self.DEVICE)
        state_dict = ckpt.state_dict()
        self.model.load_state_dict(state_dict)
    

    def get_label(self, img, bboxes):
        cate_json = '/opt/ml/project/final-project-level3-cv-12/dataset/annotations/instances_Validation.json'
        test_coco = COCO(cate_json)
        categories = test_coco.dataset['categories']

        test_dataset = TestDataset(img, bboxes, transform=self.test_transform)

        test_loader = DataLoader(test_dataset,shuffle=False)

        self.model.eval()

        label_names = []
        for images in test_loader:
            with torch.no_grad():
                images = images.to(self.DEVICE)
                pred = self.model(images)
                pred = pred.argmax(dim=-1)
                name = categories[pred]['name']
                label_names.append(name)

        return label_names