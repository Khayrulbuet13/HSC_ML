import os 

print(os.getcwd())
folder_dir = "./Output/TBCells_crops/Train_data/BCells_Old_crops/"
files = os.listdir(os.getcwd())
remove_list = [os.path.join(folder_dir, filename)  for filename in files[10:12]]
#print(remove_list)

for remove_file in remove_list:   
    try:
        os.remove(remove_file)
        print("% s removed successfully" % remove_file)
    except OSError as error:
        print(error)
        print("File path can not be removed")