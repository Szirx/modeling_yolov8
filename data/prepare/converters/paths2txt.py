import os

# Указываем путь к папке с изображениями
folder_path = '../../YOLO-MS/data/objs_v11_filtered_val/val/images'

# Получаем список файлов в папке
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

# Открываем текстовый файл для записи путей
with open('annotations_val.txt', 'w') as file:
    for image in image_files:
        # Формируем полный путь к файлу
        full_path = os.path.join(folder_path, image)
        # Записываем путь в файл
        file.write(full_path + '\n')

print("Пути к изображениям записаны в файл image_paths.txt")
