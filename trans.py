import os

dir_path_35 = r'F:\新建文件夹\35-36\35'
counter_35 = 1

for file in os.listdir(dir_path_35):
    if file.endswith('.png'):
        new_name = f'35-{counter_35}.png'
        os.rename(os.path.join(dir_path_35, file), os.path.join(dir_path_35, new_name))
        counter_35 += 1

dir_path_36 = r'F:\新建文件夹\35-36\36'
counter_36 = 1

for file in os.listdir(dir_path_36):
    if file.endswith('.png'):
        new_name = f'36-{counter_36}.png'
        os.rename(os.path.join(dir_path_36, file), os.path.join(dir_path_36, new_name))
        counter_36 += 1
        
