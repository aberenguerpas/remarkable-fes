import os

# List directory
print(os.listdir('dataset_raw'))

for folder in os.listdir('dataset_raw'):
    if not folder.startswith('.'):
        for file in os.listdir('dataset_raw/'+folder):
            # Rename file
            os.rename(f'dataset_raw/{folder}/{file}', f'dataset_raw/{folder}/{folder}_{file}')