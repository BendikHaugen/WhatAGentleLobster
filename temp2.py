import os

def filter_files(directory_path, min_id):
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"): 
            file_path = os.path.join(directory_path, filename)
            
            with open(file_path, 'r') as file:
                lines = file.readlines()
                filtered_lines = [line for line in lines if int(line.split(',')[0]) >= min_id]
            
            with open(file_path, 'w') as file:
                file.writelines(filtered_lines)

def list_subdirectories(directory_path):
    paths = os.listdir(directory_path)
    
    subdirectories = [path for path in paths if os.path.isdir(os.path.join(directory_path, path))]
    
    return subdirectories

directory_path = 'data/tracker_results/LOBSTER-test'
dirs = ['YOLOv8 + BoT-SORT', 'YOLOv8 + ByteTrack']

for dir in dirs:
    path = directory_path + "/" + dir + "/data"
    filter_files(path, 30)
