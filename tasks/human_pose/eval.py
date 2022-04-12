#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pycocotools.coco
import pycocotools.cocoeval
import os
import torch
import PIL.Image
import torchvision
import torchvision.transforms
import trt_pose.plugins
import trt_pose.models
import trt_pose.coco
import torch2trt
import tqdm
import json
from trt_pose.parse_objects import ParseObjects
import torch2trt


# In[ ]:


@tensorrt_module
class My


# In[ ]:


model = trt_pose.models.dla34up_pose(18, 42).cuda().eval()


# In[ ]:


model.load_state_dict(torch.load('tasks/human_pose/experiments/dla34up_pose_256x256_A.json.checkpoints/epoch_249.pth'))


# In[ ]:


data = torch.zeros((1, 3, 256, 256)).cuda()


# In[ ]:


@torch2trt.tensorrt_converter('torch.split')
def convert_split_dbg(ctx):
    


# In[ ]:


backbone_trt = torch2trt.torch2trt(model.backbone, [data], fp16_mode=True, max_workspace_size=1<<25)


# In[ ]:


backbone_torch = model.backbone


# In[ ]:


model.backbone = backbone_trt


# In[ ]:


torch.max(torch.abs(model.backbone(data) - backbone_trt(data)))


# In[ ]:


import pdb
pdb.pm()


# In[ ]:


model = model.cuda().eval()


# In[ ]:


cmap, paf = model(torch.zeros((1, 3, 256, 256)).cuda())


# In[ ]:


cmap.shape


# In[ ]:


paf.shape


# In[ ]:


IMAGE_SHAPE = (256, 256)
images_dir = 'val2017'
annotation_file = 'annotations/person_keypoints_val2017_modified.json'


# In[ ]:


cocoGtTmp = pycocotools.coco.COCO('annotations/person_keypoints_val2017_modified.json')


# In[ ]:


topology = trt_pose.coco.coco_category_to_topology(cocoGtTmp.cats[1])


# In[ ]:


cocoGt = pycocotools.coco.COCO('annotations/person_keypoints_val2017.json')


# In[ ]:


catIds = cocoGt.getCatIds('person')
imgIds = cocoGt.getImgIds(catIds=catIds)


# In[ ]:


transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# In[ ]:


parse_objects = ParseObjects(topology, cmap_threshold=0.05, link_threshold=0.1, cmap_window=11, line_integral_samples=7, max_num_parts=100, max_num_objects=100)


# In[ ]:


results = []

for n, imgId in enumerate(imgIds):
    
    # read image
    img = cocoGt.imgs[imgId]
    img_path = os.path.join(images_dir, img['file_name'])

    image = PIL.Image.open(img_path).convert('RGB').resize(IMAGE_SHAPE)
    data = transform(image).cuda()[None, ...]

    cmap, paf = model(data)
    cmap, paf = cmap.cpu(), paf.cpu()

#     object_counts, objects, peaks, int_peaks = postprocess(cmap, paf, cmap_threshold=0.05, link_threshold=0.01, window=5)
#     object_counts, objects, peaks = int(object_counts[0]), objects[0], peaks[0]
    
    object_counts, objects, peaks = parse_objects(cmap, paf)
    object_counts, objects, peaks = int(object_counts[0]), objects[0], peaks[0]

    for i in range(object_counts):
        object = objects[i]
        score = 0.0
        kps = [0]*(17*3)
        x_mean = 0
        y_mean = 0
        cnt = 0
        for j in range(17):
            k = object[j]
            if k >= 0:
                peak = peaks[j][k]
                x = round(float(img['width'] * peak[1]))
                y = round(float(img['height'] * peak[0]))
                score += 1.0
                kps[j * 3 + 0] = x
                kps[j * 3 + 1] = y
                kps[j * 3 + 2] = 2
                x_mean += x
                y_mean += y
                cnt += 1

        ann = {
            'image_id': imgId,
            'category_id': 1,
            'keypoints': kps,
            'score': score / 17.0
        }
        results.append(ann)
    if n % 100 == 0:
        print('%d / %d' % (n, len(imgIds)))
#     break
        
with open('results.json', 'w') as f:
    json.dump(results, f)


# In[ ]:


with open('results.json', 'w') as f:
    json.dump(results, f)


# In[ ]:


cocoDt = cocoGt.loadRes('results.json')


# In[ ]:


cocoEval = pycocotools.cocoeval.COCOeval(cocoGt, cocoDt, 'keypoints')
cocoEval.params.imgIds = imgIds
cocoEval.params.catIds = [1]
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

