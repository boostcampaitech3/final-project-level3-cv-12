from tokenize import Double
from urllib import request
from fastapi import FastAPI,UploadFile,File
from fastapi.param_functions import Depends
from pydantic import BaseModel,Field
from uuid import UUID, uuid4
from typing import List,Union,Optional,Dict,Any
from app.model import get_model_yolov5, predict_from_image_byte,_transform_image
from datetime import datetime
import cv2
import os
import numpy as np
from PIL import Image
import io
from fastapi.responses import FileResponse
from fastapi import Request
app = FastAPI()

orders = []

@app.get("/")
def hello_world():
    return {"hello":"fuck"}

class Product(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    #UUID: 고유 식별자
    #Field : 모델 스키마 또는 복잡한 Validation 검사를 위해 필드에 추가 정보를 제공할 때사용
    #default_factory: Product Class가 처음 만들어질 때 호출되는 함수를 uuid4로 하겠다 => product 클래스를 생성하면 uuid4를 만들어서 id에 저장 
    name: str
    price: float

class Order(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    xmin: float 
    ymin: float
    xmax: float
    ymax: float
    confidence: float
    classes: int
    name: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class OrderUpdate(BaseModel):
    products: List[Product] = Field(default_factory=list)

class InferenceImageProduct(Product):
    name:str = "inference_image_product"
    price:float = 100.0
    result:Optional[List]


@app.get("/order",description="주문 리스트를 가져옵니다.")
async def get_order() -> List[Order]:
    return orders

@app.get("/order/{order_id}", description="Order 정보를 가져옵니다.")
async def get_order(order_id:UUID) -> Union[Order,dict]:
    order = get_order_by_id(order_id=order_id)
    if not order:
        return {"message": "주문 정보를 찾을 수 없습니다."}
    return order

def get_order_by_id(order_id: UUID) -> Optional[Order]:
    return next((order for order in orders if order.id == order_id),None)

@app.post("/order",description="tasking.....")
async def make_order(data:Request):
    model = get_model_yolov5()
    products = []
    data = await data.body()
    # print(type(data))
    image = Image.open(io.BytesIO(data))
    image = image.convert("RGB")
    image_array = np.array(image)
    outputs = model(image_array)
    label = outputs.pandas().xyxy
    # for file in files:
        # image_bytes = await file.read()
        # inference_result = predict_from_image_byte(model=model, image_bytes=image_bytes)
        # print(inference_result[0].values.tolist())
        # print(type(inference_result))
    # return inference_result[0].values.tolist()[0]
    return label[0].values.tolist()[0]