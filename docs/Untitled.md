## backend_api.py
### line 1098 - BBoxes to detect labels

```python
for b in boxes:

        cls_id = int(b.cls[0])

        label = yolo_model.names[cls_id]

        bbox = tuple(map(int, b.xyxy[0].tolist()))

        detected_classes.append(label)

        if label in PERSONAL_CLASSES:

            personal.append((label, bbox))

        if label in LANDMARK_CLASSES:

            landmarks.append((label, bbox))
```

Output:
```python
Converts tensor -> list(float) -> list(int):
	b.xyxy → tensor([[100.4, 200.7, 350.2, 400.9]])  
	bbox → (100, 200, 350, 400)
	
---------------------------------------------------------------------------------

detected_classes = ["person", "chair", "laptop"]

personal = [
    ("laptop", (120, 200, 300, 380))
]

landmarks = [
    ("chair", (400, 250, 600, 500))
]
```

### Line 1140 - CV2 frames to faiss flow
```
frame (BGR image, H×W×3)
   |
   | bbox crop: frame[y1:y2, x1:x2]
   v
crop (BGR)
   |
   | BGR → RGB, convert to PIL Image
   v
PIL Image (RGB)
   |
   | preprocess (resize/normalize) + add batch dim
   v
img_t (torch tensor, 1×3×224×224)   [example size]
   |
   | CLIP encode_image()
   v
embedding (torch tensor, 1×512)
   |
   | normalize to unit vector
   v
emb (numpy float32, 1×512)
   |
   | FAISS add_with_ids(emb, [new_id])
   v
FAISS index on disk (vectors + IDs)
```

### IoU logic

```python
a = (ax1, ay1, ax2, ay2) = (10, 20, 50, 60)
b = (bx1, by1, bx2, by2) = (30, 40, 70, 80)

# So box **A** goes from (10,20) to (50,60)  
# and box **B** goes from (30,40) to (70,80)

1) Compute intersection rectangle coords
   
	inter_x1 = max(ax1, bx1) = max(10, 30) = 30
	inter_y1 = max(ay1, by1) = max(20, 40) = 40
	inter_x2 = min(ax2, bx2) = min(50, 70) = 50
	inter_y2 = min(ay2, by2) = min(60, 80) = 60

	# Overlapping rectangle would be
	(inter_x1, inter_y1) = (30, 40)
	(inter_x2, inter_y2) = (50, 60)

2) Check if overlap is invalid (returns 0)
   if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
    return 0.0

3) Compute intersection area
   inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
      = (50 - 30) * (60 - 40)
      = 20 * 20
      = 400
      
4) Compute area of each box and union
   area_a = (ax2 - ax1) * (ay2 - ay1)
       = (50 - 10) * (60 - 20)
       = 40 * 40
       = 1600
    
    area_b = (bx2 - bx1) * (by2 - by1)
       = (70 - 30) * (80 - 40)
       = 40 * 40
       = 1600
    
    union = area_a + area_b - inter
      = 1600 + 1600 - 400
      = 2800
      
3) IoU
   iou = inter / (area_a + area_b - inter + 1e-9)
    = 400 / (2800 + 1e-9)
    ≈ 0.142857...
    

```