import os

folder_path = "./data/foggy_cityscapes/images/leftImg8bit_foggy"

# check if the folder exists if not error
if not os.path.exists(folder_path):
    raise FileNotFoundError(f"The folder '{folder_path}' does not exist. Have you already ran the preprocessing? \n")

# Iterate through the folder structure
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if "_foggy_beta_" in file:
            file_name, ext = os.path.splitext(file)
            beta_value = file_name.split("_foggy_beta_")[-1]
            if beta_value != "0.02":
                # Remove images with beta_value not equal to 0.02
                os.remove(os.path.join(root, file))
            else:
                # Rename images with beta_value equal to 0.02
                new_name = file_name.replace("_foggy_beta_0.02", "")
                new_file = new_name + ext
                os.rename(
                    os.path.join(root, file),
                    os.path.join(root, new_file)
                )

new_folder_name = "leftImg8bit"
new_folder_path = os.path.join(os.path.dirname(folder_path), new_folder_name)

os.rename(folder_path, new_folder_path)

print('Preprocessing complete; Only foggy images with beta=0.02 remain. \n')