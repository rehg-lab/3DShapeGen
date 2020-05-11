import numpy
import os
import config
import numpy as np
from datetime import datetime


out_dir = config.training['out_dir']

eval_task_name = config.testing['eval_task_name']
eval_dir = os.path.join(out_dir, 'eval')
eval_task_dir = os.path.join(eval_dir, eval_task_name) 

out_path = os.path.join(eval_task_dir, 'out.npz')

save_res_path = './results'
os.makedirs(save_res_path, exist_ok=True)

out_data = np.load(out_path, allow_pickle=True)
cd = out_data['cd']
normals = out_data['normals']
fscore = out_data['fscore']
iou = out_data['iou']
obj_cat = out_data['obj_cat']
pose = out_data['pose']

mean_cd_instance = np.mean(cd)
mean_normals_instance = np.mean(normals)
if len(iou.shape) < 2:
    mean_iou_instance = np.mean(iou)
else:
    # Does not print out signed IoU
    mean_iou_instance = np.mean(iou, axis=0)[0]
mean_fscore_05_instance = np.mean(fscore, axis=0)[0]
mean_fscore_1_instance = np.mean(fscore, axis=0)[1]

cat_map = {}
for i,oc in enumerate(obj_cat):
    obj, cat = oc
    obj = obj[0]
    cat = cat[0]

    if cat not in cat_map:
        cat_map[cat] = {'cd': [], 'iou': [], 'normals': [], 'fscore05': [],\
                        'fscore1': []}
    cat_map[cat]['cd'].append(cd[i])
    cat_map[cat]['iou'].append(iou[i])
    cat_map[cat]['normals'].append(normals[i])
    cat_map[cat]['fscore05'].append(fscore[i][0])
    cat_map[cat]['fscore1'].append(fscore[i][1])
     
now = str(datetime.now().strftime("%H:%M %d-%m-%Y"))
log_string = ""
log_string += '-------------------------------------\n'
log_string += "{:25} {}\n".format('Time:', now)
log_string += "{:25} {}\n".format('Mean CD per mdl:',\
         mean_cd_instance)
log_string += "{:25} {}\n".format('Mean normals per mdl:',\
         mean_normals_instance)
log_string += "{:25} {}\n".format('Mean iou per mdl:',\
         mean_iou_instance)
log_string += "{:25} {}\n".format('Mean fscore@05 mdl:',\
         mean_fscore_05_instance)
log_string += "{:25} {}\n".format('Mean fscore@1 mdl:',\
         mean_fscore_1_instance)

log_string += "\nMean CD per category\n"
cat_avg = []
for cat in cat_map:
    cat_data = cat_map[cat]
    mean_cat = np.mean(cat_data['cd'])
    cat_avg.append(mean_cat)
    log_string += "{:25} {}\n".format(cat, mean_cat)
log_string += "{:25} {}\n".format("Avg", np.mean(cat_avg))

log_string += "\nMean Normals per category\n"
cat_avg = []
for cat in cat_map:
    cat_data = cat_map[cat]
    mean_cat = np.mean(cat_data['normals'])
    log_string += "{:25} {}\n".format(cat, mean_cat) 
    cat_avg.append(mean_cat)

log_string += "{:25} {}\n".format("Avg", np.mean(cat_avg))

log_string += "\nMean IoU per category\n"
cat_avg = []
for cat in cat_map:
    cat_data = cat_map[cat]
    if len(iou.shape) < 2:
        mean_cat = np.mean(cat_data['iou'])
    else:
        mean_cat = np.mean(cat_data['iou'],axis=0)[0]
    log_string += "{:25} {}\n".format(cat, mean_cat)
    cat_avg.append(mean_cat)
 
log_string += "{:25} {}\n".format("Avg", np.mean(cat_avg,axis=0))

log_string += "\nMean F-Score@0.5 per category\n"
cat_avg = []
for cat in cat_map:
    cat_data = cat_map[cat]
    mean_cat = np.mean(cat_data['fscore05'])
    log_string += "{:25} {}\n".format(cat, mean_cat)
    cat_avg.append(mean_cat)

log_string += "{:25} {}\n".format("Avg", np.mean(cat_avg))
log_string += "\nMean F-Score@1 per category\n"
cat_avg = []
for cat in cat_map:
    cat_data = cat_map[cat]
    mean_cat = np.mean(cat_data['fscore1'])
    log_string += "{:25} {}\n".format(cat, mean_cat)
    cat_avg.append(mean_cat)
log_string += "{:25} {}\n".format("Avg", np.mean(cat_avg))

with open(os.path.join(save_res_path, '%s.txt'%(eval_task_name)), 'a+')\
             as out_file:
    out_file.write(log_string)


        


