from mmdet.apis import init_detector, inference_detector
import os
import mmcv

config_file = '/home/goeun/hair_inference/swin/htc_swin_hrfpn_dyhead_v3.py'
checkpoint_file = '/home/goeun/hair_inference/swin/epoch_240.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'
print('model 호출 ....')

data = '/home/goeun/hair_inference/swin/dataset/slicing/test/images/32-T8.jpg'
result = inference_detector(model, data)
print('result inference... : ')
bbox_result, segm_result = result
print('-------bbox result---------')
for a in bbox_result : 
    print(a)
print('---------segm_result---------')
for b in segm_result : 
    print(b)
print('------------------------------')
#전체 추론 결과 이미지 저장
out_img = model.show_result(data, result)
file_name = os.path.basename(data)
well_saved = False
while(not well_saved) : 
    well_saved = mmcv.imwrite(out_img, os.path.join('/home/goeun/client/static/result', file_name))
    result_path = os.path.join('/home/goeun/client/static/result', file_name)

print('well_saved : ', well_saved, ', path : ', result_path)
    # 1. bbox의 개수 ; 머리카락 개수
bbox_num = len(bbox_result[0])

# 2. segmentation 결과 -> thickness 얻기
# thic/
# 3. 머리카락 개수, thickness 관련 통계 정리해서 표, 그래프 표시 

