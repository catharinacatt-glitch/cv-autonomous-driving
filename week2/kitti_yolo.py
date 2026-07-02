def corner_to_yolo(x_min, y_min, x_max, y_max, img_w, img_h, cls):
   
    x_center = (x_min + x_max)/2
    y_center = (y_min + y_max)/2

    
    w = x_max - x_min
    h = y_max - y_min
    
    x_center /= img_w
    y_center /= img_h
    w /= img_w
    h /= img_h
    
    return f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"

class_map = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}

def kitti_line_to_yolo(line, img_w, img_h, class_map):
    parts=line.split()
    name=parts[0]

    if name not in class_map:
        return None

    cls=class_map[name]

    x_min=float(parts[4])
    y_min=float(parts[5])
    x_max=float(parts[6])
    y_max=float(parts[7])

    return corner_to_yolo(x_min, y_min, x_max, y_max, img_w, img_h, cls)

import os
import cv2

def convert_kitti_dataset(label_dir, image_dir, output_dir, class_map):   # img_w, img_h 빼고 image_dir 추가
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(label_dir):
        if not filename.endswith('.txt'): continue

        in_path  = os.path.join(label_dir, filename)
        out_path = os.path.join(output_dir, filename)

        img_path = os.path.join(image_dir, filename.replace('.txt', '.png'))
        img = cv2.imread(img_path)
        h, w = img.shape[:2]                      # 이미지마다 실제 크기

        yolo_lines = []
        with open(in_path, 'r') as f:
            for line in f:
                result = kitti_line_to_yolo(line, w, h, class_map)   # ← img_w,img_h 가 아니라 w,h
                if result is not None:
                    yolo_lines.append(result)

        with open(out_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
  
convert_kitti_dataset(
    label_dir="/Users/hwanghyunjin/Desktop/CV-Portfolio/week2/training/label_2",
    image_dir="/Users/hwanghyunjin/Desktop/CV-Portfolio/week2/training/image_2",  # ← 추가
    output_dir="/Users/hwanghyunjin/Desktop/CV-Portfolio/week2/datasets/kitti/labels/train",
    class_map={'Car': 0, 'Pedestrian': 1, 'Cyclist': 2},
)