import torch

class Humandetect():
  def __init__(self):
    self.model = self.get_model()

  def get_model(self):
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
      model = model.to(device)
      return model

  def detect(self,frame):
      output = self.model(frame)
      label = list(output.pandas().xyxy[0]['name'])
      if 'person' in label:
        return True
      return False
