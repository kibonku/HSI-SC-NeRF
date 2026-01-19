#--- rename_script.py --->START
'''
import os
import re

# 파일이 있는 실제 폴더 경로로 변경해 주세요 (예: 'C:\\Users\\YourUser\\Desktop\\images')
folder_path = r'nerfstudio/_custom_dataset/2.pre/maize/1981x/204_hsi_WR/images_8' 
file_list = sorted(os.listdir(folder_path))
# print(file_list)

# 폴더로 이동
os.chdir(folder_path)

# 폴더 내 모든 파일 가져오기
for filename in file_list:
    # 'frame_'과 '.npy' 확장자 사이의 5자리 숫자를 찾기 위한 정규 표현식
    match = re.search(r'frame_(\d{5})\.npy', filename)
    if match:
        # 현재 숫자 추출
        current_number = int(match.group(1))
        # 숫자를 1 감소시킴
        new_number = current_number - 1
        # 새 파일 이름 생성 (5자리 숫자 형식 유지)
        new_filename = f'frame_{new_number:05d}.npy'
        
        # 이름 변경 실행
        try:
            os.rename(filename, new_filename)
            print(f'{filename} -> {new_filename}')
        except FileExistsError:
            print(f'오류: {new_filename}이(가) 이미 존재합니다. 이름 변경을 건너뜁니다.')

'''
print("이름 변경이 완료되었습니다.")
#--- rename_script.py --->END


#--- NPY WR check ---#
'''import cv2
import numpy as np

# Load the .npy file
data_array = np.load('nerfstudio/_custom_dataset/2.pre/S1/10_hsi/images/frame_00001.npy')

# Manually scale and convert your data to 8-bit unsigned integers (uint8)
# If your float data is 0.0 to 1.0, multiply by 255:
if data_array.dtype == np.float32 or data_array.dtype == np.float64:
    # Ensure values are clipped to [0, 1] before scaling
    data_array_scaled = (np.clip(data_array, 0.0, 1.0) * 255).astype(np.uint8)
# If your data is 16-bit integers (0 to 65535), you need to scale down:
elif data_array.dtype == np.uint16:
     data_array_scaled = (data_array // 256).astype(np.uint8)
else:
    # For other types, a simple astype might suffice if ranges are okay
    data_array_scaled = data_array.astype(np.uint8)

# Now save the manually converted array
cv2.imwrite('output_image_fixed.png', data_array_scaled)'''

#--- NPY WR check ---#




#--- PCD composition check---#
from plyfile import PlyData

ply = PlyData.read("nerfstudio/_custom_dataset_application-paper/4.pcd/S2/S2_t4/204_hsi_WR_mask/204_hsi_WR_mask.ply")

vertex = ply['vertex'].data   # structured array

print(vertex.dtype.names)

# 전체 출력
for i, row in enumerate(vertex):
    print(i, row)  # expect) (x,y,z, nx,ny,nz, b1,b2,...,bn)
    break
#--- PCD composition check---#
