import argparse
import csv
import os
import cv2

def extract_metadata(root_path: str, directories: list, output_files: list):
    """
    Extracts metadata of images in the specified directories and stores them in CSV files.
    :param root_path: str, root path of the train and validation directories
    :param directories: list of strings, subdirectory names containing the images
    :param output_files: list of strings, output CSV filenames to store the metadata
    :return: None
    """
    # check if number of directories is equal to number of output files
    if len(directories) != len(output_files):
        print("Error: number of directories must be equal to the number of output files")
        return

    # loop through each directory and extract metadata of each image
    for i, directory in enumerate(directories):
        print(f"Processing directory {directory}...")
        images_path = os.path.join(root_path, directory)

        # create output CSV file and write header row
        output_file = output_files[i]
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['filename', 'file_path', 'bytes_size', 'resolution', 'aspect_ratio', 'type', 'label'])

            # loop through each image in directory and extract metadata
            for subdir, _, files in os.walk(images_path):
                for file in files:
                    file_path = os.path.join(subdir, file)
                    if file_path.endswith(".jpg") or file_path.endswith(".jpeg") or file_path.endswith(".png"):
                        # read image using opencv
                        img = cv2.imread(file_path)

                        # get image file name
                        filename = os.path.basename(file_path)

                        # get image size in bytes
                        bytes_size = os.path.getsize(file_path)

                        # get image resolution and aspect ratio
                        resolution = f"{img.shape[1]}x{img.shape[0]}"
                        aspect_ratio = round(img.shape[1] / img.shape[0], 2)

                        # get dataset type
                        type = directory

                        # Get the label based on the folder directory
                        label = 'unknown'
                        if 'dog' in subdir:
                            label = 'dog'
                        elif 'cat' in subdir:
                            label = 'cat'
                        elif 'wild' in subdir:
                            label = 'wildlife'                        

                        # write to csv the metadata generated
                        writer.writerow([filename, file_path, bytes_size, resolution, aspect_ratio, type, label])


if __name__ == '__main__':
    # setup argparse
    parser = argparse.ArgumentParser(description='Extracts metadata of images and stores them in CSV files')
    parser.add_argument('--root_path', type=str, required=True, help='Root path of the train and validation directories')
    parser.add_argument('--directories', type=str, nargs='+', required=True,
                        help='Subdirectory names containing the images')
    parser.add_argument('--output_files', type=str, nargs='+', required=True,
                        help='Output CSV filenames to store the metadata')
    args = parser.parse_args()

    # call the function with the parsed arguments
    extract_metadata(args.root_path, args.directories, args.output_files)