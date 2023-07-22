import os
import zipfile
import shutil
import json
import numpy as np
from collections import namedtuple
from PIL import Image


KRLoadClass = namedtuple('KRLoadClass', ['name', 'id', 'color'])

classes = [
    KRLoadClass('background', 0, (0,0,0)),
    KRLoadClass('wheelchair', 1, (255, 0, 0)),
    KRLoadClass('truck', 2, (0, 255, 0)),
    KRLoadClass('traffic_sign', 3, (0, 0, 255)),
    KRLoadClass('traffic_light',4, (255, 0, 255)),
    KRLoadClass('stroller', 5, (255, 255, 0 )),
    KRLoadClass('stop', 6, (0 ,255, 255)),
    KRLoadClass('scooter', 7, (128, 128, 0)),
    KRLoadClass('pole', 8, (0, 128, 128)),
    KRLoadClass('person', 9, (128, 0, 128)),
    KRLoadClass('motorcycle', 10, (128, 0, 255)),
    KRLoadClass('dog', 11, (255, 128, 255)),
    KRLoadClass('cat', 12, (64, 0, 0)),
    KRLoadClass('carrier', 13, (0, 64, 0)),
    KRLoadClass('car', 14, (0, 0, 64)),
    KRLoadClass('bus', 15, (64, 64, 0)),
    KRLoadClass('bollard', 16, (64, 0, 64)),
    KRLoadClass('bicycle', 17, (0, 64, 64)),
    KRLoadClass('barricade', 18, (0, 192, 0)),
]

class_names = [i.name for i in classes]

cmap = []
for i in classes:
    if i.id >=0 and i.id <19:
        cmap.append(i.color)


# RLE 인코딩 함수
def rle_encode(mask):
    """
    다차원 텐서를 RLE 인코딩하는 함수

    :param tensor: 2차원 텐서 (height x width)
    :return: RLE 인코딩된 문자열 리스트
    """
    mask_flatten = mask.flatten()
    mask_flatten = np.concatenate([[0], mask_flatten, [0]])
    runs = np.where(mask_flatten[1:] != mask_flatten[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    rle = ' '.join(str(x) for x in runs)
    return rle


# RLE 디코딩 함수
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


# new_data 폴더에 새로운 zipfile이 들어왔는지 확인하는 함수
def check_zipfile_in_folder(file_path):
    file_list = os.listdir(file_path)

    if len(file_list) == 0:
        print("No New File")
        return False
    else:
        print("Files Exist")
        return True


# zipfile 압축해제
def extract_zip(zip_file_path, extract_to):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


#캠퍼ID 추출
def get_ID(file):
    file_name, file_extension = os.path.splitext(file)

    return file_name


#캠퍼 폴더 생성 and 확인
def make_folder(id):
    train_path = "/opt/ml/level3_cv_finalproject-cv-09/MLflow/data/krload"

    if not os.path.exists(os.path.join(train_path, "imgs", id)):
        os.makedirs(os.path.join(train_path, "imgs", id))
        os.makedirs(os.path.join(train_path, "labels", id))
    else:
        return


#jpg랑 json 매칭되는지 확인후 파일 이동
def check_if_match_and_send(id):
    tmp_path = "/opt/ml/level3_cv_finalproject-cv-09/MLflow/data/tmp"
    dest_img_path = "/opt/ml/level3_cv_finalproject-cv-09/MLflow/data/krload/imgs"
    dest_label_path = "/opt/ml/level3_cv_finalproject-cv-09/MLflow/data/krload/labels"

    file_list = os.listdir(tmp_path)

    s_list = sorted(file_list)

    while len(s_list) > 1:
        file1 = s_list[0]
        file2 = s_list[1]

        name1, ext1 = os.path.splitext(file1)
        name2, ext2 = os.path.splitext(file2)

        file1_full = os.path.join(tmp_path, file1)
        file2_full = os.path.join(tmp_path, file2)

        if name1 == name2:
            if ext1 == ".json":                
                shutil.move(file1_full, os.path.join(dest_label_path, id, file1))
                shutil.move(file2_full, os.path.join(dest_img_path, id, file2))
            else:
                shutil.move(file1_full, os.path.join(dest_img_path, id, file1))
                shutil.move(file2_full, os.path.join(dest_label_path, id, file2))

            s_list.pop(0)
            s_list.pop(0)
        else:
            s_list.pop(0)



#이미지파일 확장자 갖고오는 함수
def get_ext(id):
    image_path = "/opt/ml/level3_cv_finalproject-cv-09/MLflow/data/krload/imgs"

    t_path = os.path.join(image_path, id)

    file_list = os.listdir(t_path)

    name, ext = os.path.splitext(file_list[0])

    return ext

    

#tmp폴더 비우기
def refresh_tmp():
    tmp_path = "/opt/ml/level3_cv_finalproject-cv-09/MLflow/data/tmp"

    shutil.rmtree(tmp_path)

    os.mkdir(tmp_path)


#new_data 폴더 비우기
def refresh_new_data():
    new_file_path = "/opt/ml/level3_cv_finalproject-cv-09/MLflow/data/new_data"

    shutil.rmtree(new_file_path)

    os.mkdir(new_file_path)


#마스크 클래스별 컬러 지정
def mask_color(mask,cmap):
    if isinstance(mask,np.ndarray):
        r_mask = np.zeros_like(mask,dtype=np.uint8)
        g_mask = np.zeros_like(mask,dtype=np.uint8)
        b_mask = np.zeros_like(mask,dtype=np.uint8)
        for k in range(len(cmap)):
            indice = mask==k
            r_mask[indice] = cmap[k][0]
            g_mask[indice] = cmap[k][1]
            b_mask[indice] = cmap[k][2]
        return np.stack([b_mask, g_mask, r_mask], axis=2)




#json파일 이미지로 바꿔서 저장
def change_json_to_image(id, ext):
    dest_label_path = "/opt/ml/level3_cv_finalproject-cv-09/MLflow/data/krload/labels"

    id_label_path = os.path.join(dest_label_path, id)

    file_list = os.listdir(id_label_path)

    for file in file_list:
        file_path = os.path.join(id_label_path, file)

        with open(file_path, 'r') as json_file:
            data = json.load(json_file)

            shape = data["size"]

            image = np.zeros((shape[0], shape[1]), dtype=np.uint8)

            for type in data["masks"]:
                target_id = 0

                for l in classes:
                    if type == l.name:
                        target_id = l.id

                raw_mask = rle_decode(data["masks"][type], shape)

                raw_mask = raw_mask * target_id

                image = image + raw_mask
            
            file_path = file_path.replace(".json",ext)

            # image = mask_color(image, cmap)

            im = Image.fromarray(image)

            im.save(file_path)




        
        


if __name__=='__main__':
    new_file_path = "/opt/ml/level3_cv_finalproject-cv-09/MLflow/data/new_data"
    
    if not check_zipfile_in_folder(new_file_path):
        print("Job Finished")
    else:
        file_list = os.listdir(new_file_path)

        for file in file_list:
            tmp_path = "/opt/ml/level3_cv_finalproject-cv-09/MLflow/data/tmp"

            extract_zip(os.path.join(new_file_path, file), tmp_path)

            id = get_ID(file)

            make_folder(id)

            check_if_match_and_send(id)

            ext = get_ext(id)

            change_json_to_image(id, ext)

            refresh_tmp()

        refresh_new_data()







            



