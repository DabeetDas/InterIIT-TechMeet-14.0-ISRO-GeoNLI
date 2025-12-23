from ultralytics import YOLO
import requests
from PIL import Image
import io
import cv2

model = YOLO("yolo11l-obb.pt")

# predict obbs for swimming pool and display imgage
img_url = 'https://raw.githubusercontent.com/jugal-sac/InterIIT_dataset_sample/refs/heads/main/sample2.png'

response = requests.get(img_url)
image = Image.open(io.BytesIO(response.content)).convert('RGB')

results = model.predict(image, conf=0.25, iou=0.5, verbose=False)

# Convert PIL image to numpy array for OpenCV
import numpy as np
img_array = np.array(image)
img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

# display image with bounding boxes
for r in results:
    # For OBB models, use r.obb instead of r.boxes
    if hasattr(r, 'obb') and r.obb is not None and r.obb.xyxyxyxy is not None:
        # Get oriented bounding box corners (4 points)
        corners = r.obb.xyxyxyxy.cpu().numpy()
        for corner_set in corners:
            # Convert to integer coordinates
            pts = corner_set.astype(int)
            # Draw the oriented bounding box as a polygon
            cv2.polylines(img_cv, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
            print(f"OBB corners: {pts}")
    elif hasattr(r, 'boxes') and r.boxes is not None:
        # Fallback for regular bounding boxes (if any)
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(x1, y1, x2, y2)
            cv2.rectangle(img_cv, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    else:
        print("No detections found")

cv2.imshow('image', img_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()