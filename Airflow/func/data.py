import os
import zipfile
import shutil
import json
import numpy as np
from collections import namedtuple
from PIL import Image


KRLoadClass = namedtuple('KRLoadClass', ['name', 'id', 'color'])

classes = [
        KRLoadClass('background', 255, (0,0,0)),      # 배경
        KRLoadClass('traffic_light_controller',0,(0, 0, 255)),
        KRLoadClass('wheelchair', 1, (255, 0, 0)),  # 휠체어
        KRLoadClass('carrier', 2, (0, 64, 0)),     # 화물차
        KRLoadClass('stop', 3, (0 ,255, 255)),     # 정지선
        KRLoadClass('cat', 4, (64, 0, 0)),         # 고양이
        KRLoadClass('pole', 5, (0, 128, 128)),     # 대
        KRLoadClass('traffic_light',6, (255, 0, 255)),  # 신호등
        KRLoadClass('traffic_sign', 7, (0, 0, 255)),    # 교통 표지판
        KRLoadClass('stroller', 8, (255, 255, 0 )), # 유모차
        KRLoadClass('dog', 9, (255, 128, 255)),    # 개
        KRLoadClass('barricade', 10, (0, 192, 0)),  # 바리케이드
        KRLoadClass('person', 11, (128, 0, 128)),    # 사람 
        KRLoadClass('scooter', 12, (128, 128, 0)),  # 스쿠터
        KRLoadClass('car', 13, (0, 0, 64)),         # 차
        KRLoadClass('truck', 14, (0, 255, 0)),       # 트럭
        KRLoadClass('bus', 15, (64, 64, 0)),        # 버스 
        KRLoadClass('bollard', 16, (64, 0, 64)),    # 인도 블럭 바리케이드 비슷한거
        KRLoadClass('motorcycle', 17, (128, 0, 255)),   # 오토바이
        KRLoadClass('bicycle', 18, (0, 64, 64)),    # 자전거
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
        os.makedirs(os.path.join(train_path, "imgs/train", id))
        os.makedirs(os.path.join(train_path, "labels/train", id))
    else:
        return


#jpg랑 json 매칭되는지 확인후 파일 이동
def check_if_match_and_send(id):
    tmp_path = "/opt/ml/level3_cv_finalproject-cv-09/MLflow/data/tmp"
    dest_img_path = "/opt/ml/level3_cv_finalproject-cv-09/MLflow/data/krload/imgs/train"
    dest_label_path = "/opt/ml/level3_cv_finalproject-cv-09/MLflow/data/krload/labels/train"

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
    image_path = "/opt/ml/level3_cv_finalproject-cv-09/MLflow/data/krload/imgs/train"

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


#이름같은 파일의 확장자 알아오기
def find_files_with_same_name(folder_path, target_name):
    # 해당 폴더 안의 모든 파일들을 가져옵니다.
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 파일의 확장자를 제외한 이름을 가져옵니다.
            filename_without_extension = os.path.splitext(file)[0]

            # 확장자 제외한 파일명과 대상 이름이 같은지 확인합니다.
            if filename_without_extension == target_name:
                # 같다면 해당 파일의 확장자를 출력합니다.
                file_extension = os.path.splitext(file)[1]
                return file_extension

#json파일 이미지로 바꿔서 저장
def change_json_to_image(id):
    dest_label_path = "/opt/ml/level3_cv_finalproject-cv-09/MLflow/data/krload/labels/train"
    dest_image_path = "/opt/ml/level3_cv_finalproject-cv-09/MLflow/data/krload/imgs/train"

    id_label_path = os.path.join(dest_label_path, id)
    id_image_path = os.path.join(dest_image_path, id)

    file_list = os.listdir(id_label_path)

    s_list = sorted(file_list)

    for file in s_list:
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

            file_name = os.path.splitext(file)[0]
            
            ext = find_files_with_same_name(id_image_path, file_name)
            
            file_path = file_path.replace(".json", ext)

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

            change_json_to_image(id)

            refresh_tmp()

        refresh_new_data()