from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
import numpy as np
from tqdm import tqdm
import mmcv
import os

def get_bbox(path):
    translation_dct = {0:'ignored regions', 1:'pedestrian', 2:'people', 3:'bicycle', 4:'car', 
                      5:'van', 6:'truck', 7:'tricycle', 8:'awning-tricycle', 9:'bus', 10:'motor', 
                      11:'others'
                      }
    with open(path, 'r') as file:
        annotation = file.read().strip().split('\n')
        annotation_list=list(map(lambda x: list(map(int, x.split(',')[:6])), annotation))
        annotation_list = list(map(lambda a: [a[0], a[1], a[0]+a[2], a[1]+a[3]], list(filter(lambda a: a[5]==2, annotation_list))))
        return annotation_list

@DATASETS.register_module()
class VizdroneDataset(CustomDataset):
      CLASSES = ('Person', )

      def load_annotations(self, ann_file):
          cat2label = {k: i for i, k in enumerate(self.CLASSES)}
          image_list = os.listdir(f'{self.img_prefix}')
          data_infos = []
          for image_name in tqdm(image_list):
              img_path = f'{self.data_root}images/{image_name}'
              image = mmcv.imread(img_path)
              height, width = image.shape[:2]
    
              data_info = dict(filename=image_name, width=width, height=height)
              # load annotations
              label_prefix = f'{self.data_root}/annotations/'
              file_name = image_name.split('.')[0] + '.txt'
              bounding_boxes = np.array(get_bbox(f'{label_prefix}{file_name}'), 
                                dtype=np.float32).reshape(-1, 4)
              data_anno = dict(
                  bboxes=bounding_boxes,
                  labels=np.array([cat2label['Person'] 
                                   for _ in range(bounding_boxes.shape[0])], dtype=np.long),
                  #bboxes_ignore=np.array([], dtype=np.float32),
                  #labels_ignore=np.array([], dtype=np.long))
              )
              data_info.update(ann=data_anno)
              data_infos.append(data_info)
          return data_infos