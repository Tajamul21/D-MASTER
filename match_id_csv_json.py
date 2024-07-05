import json
import csv

# Load the JSON file and extract image IDs and file names
# with open('/home/tajamul/scratch/DA/DATA/Dmaster_Data/IRCH/annotations/image_info_test-dev2017.json', 'r') as json_file:
with open('/home/kaustubh/scratch/Mammo_Datasets_negroni/Dmaster_Data/c_view_data/c_view.json', 'r') as json_file:
    coco_data = json.load(json_file)
    image_id_to_name = {image['id']: image['file_name'] for image in coco_data['images']}

# Load the existing CSV file and extract image IDs, confidence scores, and bounding boxes
csv_data = []
with open('/home/kaustubh/scratch/D-MASTER/outputs_krb/teaching/csv_preds_kaustubh/1_d_master_adap_cview_post_training.csv', 'r') as csv_file:
    
    reader = csv.DictReader(csv_file)
    for row in reader:
        image_id = int(row['image_name'])  # Corrected column name
        if image_id in image_id_to_name:  # Match image ID with JSON data
            file_name_image_id = image_id_to_name[image_id]
            confidence_score = float(row['confidence_score'])
            bounding_box = row['bounding_box']
            csv_data.append({
                'file_name_image_id': file_name_image_id,
                'confidence_score': confidence_score,
                'bounding_box': bounding_box
            })
        else:
            print(f"Image ID '{image_id}' not found in JSON data.")

# Write the new CSV file with file name (image name) and image ID, confidence score, and bounding box
new_csv_filename = '/home/kaustubh/scratch/D-MASTER/outputs_krb/teaching/csv_preds_kaustubh/1_d_master_adap_cview_post_training_with_names.csv'
with open(new_csv_filename, mode='w', newline='') as new_csv_file:
    fieldnames = ['file_name_image_id', 'confidence_score', 'bounding_box']
    writer = csv.DictWriter(new_csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_data)

print(f"New CSV file '{new_csv_filename}' has been created.")
