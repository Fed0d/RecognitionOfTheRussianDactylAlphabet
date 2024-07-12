from mmaction.apis import init_recognizer, inference_recognizer
import pandas as pd
import json
import os

# Загрузка модели
config_file = '../../work/20240416_214105/vis_data/config.py'
checkpoint_file = '../../work/best_acc_top1_epoch_78.pth'

model = init_recognizer(config_file, checkpoint_file, device='cuda:0')

ann_file_test = '../../data/rda/rda_val.csv'
data_root_val = '../../data/rda/val'

df = pd.read_csv(ann_file_test, names=['filename', 'label'], sep=' ')

y_true = []
y_pred = []
y_scores = []
y_classes = []

for index, row in df.iterrows():
    video_path = os.path.join(data_root_val, row['filename'])

    result = inference_recognizer(model, video_path)

    y_pred_label = result.pred_label.item()
    y_pred_score = result.pred_score.cpu()[y_pred_label].item()

    y_scores.append(y_pred_score)
    if y_pred_label == row['label']:
        y_classes.append(1)
    else:
        y_classes.append(0)

    y_true.append(row['label'])
    y_pred.append(y_pred_label)

with open('cm.json', 'w') as f:
    json.dump({
        'y_true': y_true,
        'y_pred': y_pred
    }, f)
with open('roc.json', 'w') as f:
    json.dump({
        'y_scores': y_scores,
        'y_classes': y_classes
    }, f)

