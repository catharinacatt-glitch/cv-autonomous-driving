import os, random, shutil

# 경로 설정
img_src   = 'data_object_image_2/training/image_2'   # 원본 이미지
label_src = 'datasets/kitti/labels_all'            # 변환된 라벨 (현재 전부 여기)
base      = 'datasets/kitti'

# 1. 라벨 파일 목록에서 '파일명(번호)'만 뽑기
#    ('000000.txt' → '000000')
ids = [f[:-4] for f in os.listdir(label_src) if f.endswith('.txt')]

# 2. 섞고 8:2로 자르기
random.seed(42)              # 재현성 (매번 같은 분할)
random.shuffle(ids)
split = int(len(ids) * 0.8)
train_ids = ids[:split]
val_ids   = ids[split:]

# 3. 목적지 폴더 4개 만들기
for sub in ['images/train', 'images/val', 'labels/train', 'labels/val']:
    os.makedirs(os.path.join(base, sub), exist_ok=True)

# 4. 파일 옮기는 함수
def place(id_list, split_name):
    for id in id_list:
        # 이미지 복사: image_2/000000.png → images/{split_name}/000000.png
        shutil.copy(
            os.path.join(img_src, id + '.png'),
            os.path.join(base, 'images', split_name, id + '.png')
        )
        # 라벨 복사: labels/train/000000.txt → labels/{split_name}/000000.txt
        shutil.copy(
            os.path.join(label_src, id + '.txt'),   # ← 빈칸 A
            os.path.join(base, 'labels', split_name, id + '.txt')   # ← 빈칸 B
        )

place(train_ids, 'train')
place(val_ids, 'val')

print(f"train: {len(train_ids)}, val: {len(val_ids)}")