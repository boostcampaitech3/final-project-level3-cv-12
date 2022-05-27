def get_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
    model.to(device)
    return model
  
model = get_model()  

def Human_detect(frame,model):
    output = model(frame)
    label = list(output.pandas().xyxy[0]['name'])
    if 'person' in label:
      return True
    return False
