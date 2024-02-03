import csv
import os

def generate_csv(output_base_dir, csv_filename):
    image_records = []

    # Walk through the output directory and collect all image records
    for root, dirs, files in os.walk(output_base_dir):
        for file in sorted(files):  # Sort files to ensure frames of the same video are in sequence
            if file.endswith('.jpg'):
                # Class is the parent directory of the current root
                class_name = os.path.basename(os.path.dirname(root))
                # Collect the path and the class name
                image_records.append((os.path.join(root, file), class_name))

    # Sort image records by path
    image_records.sort(key=lambda x: x[0])

    # Write sorted records to the CSV file
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['path', 'class']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for path, class_name in image_records:
            writer.writerow({'path': path, 'class': class_name})

if __name__ == "__main__":
    output_base_dir = ''  # Replace with your dir to save query-target candidate images
    csv_filename = ''  # Replace with your annotation-file's path for query-target candidate images

    generate_csv(output_base_dir, csv_filename)