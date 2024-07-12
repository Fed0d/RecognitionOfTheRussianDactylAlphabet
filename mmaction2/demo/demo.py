from mmaction.apis import init_recognizer, inference_recognizer
import os

# Загрузка модели
config_file = '../../work/20240416_214105/vis_data/config.py'
checkpoint_file = '../../work/best_acc_top1_epoch_78.pth'  # замените на путь к вашему файлу .pth

model = init_recognizer(config_file, checkpoint_file, device='cuda:0')

# Путь к папке с видео
video_folder = '../../data/NewTest'

# Получаем список всех видеофайлов в папке
video_files = sorted([f for f in os.listdir(video_folder) if f.startswith('test_') and f.endswith('.mp4')],
                     key=lambda x: int(x.split('_')[1].split('.')[0]))

# Для каждого видеофайла выполняем предсказание
for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)

    # Предсказание
    result = inference_recognizer(model, video_path)

    # Вывод результатов
    print(f'Video file: {video_file}')
    print(f'Predicted label: {result.pred_label}')
