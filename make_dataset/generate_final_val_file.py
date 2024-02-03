import os
import csv
import random
from collections import defaultdict
import pandas as pd


if __name__ == '__main__':
    candidates_file = '' # Replace with your annotation-file's path for query-target candidate images
    db_file = '' # Replace with your dir to save distraction images
    anno_file_path = '' # Replace with your final annotation-file's path

    db_image_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(db_file) for f in filenames if f.endswith('.jpg')]
    print(len(db_image_files))
    df_db = pd.DataFrame({
        'path': db_image_files,
        'label': [1000000] * len(db_image_files),
        'type': ['db'] * len(db_image_files)
    })

    df = pd.read_csv(candidates_file)
    df_valid = df[(df['contains_person'] == True) & (df['blip2_result'].isin(['Yes', 'yes', 'Both', 'both']))]

    folder_dict = defaultdict(list)
    for index, row in df_valid.iterrows():
        folder_name = row['path'].split('/')[-2]
        folder_dict[folder_name].append(row)

    output_df = pd.DataFrame(columns=['path', 'class', 'label', 'type'])
    type_id = 1
    for folder, rows in folder_dict.items():
        if len(rows) >= 2:
            # If there are at least two images, randomly select two
            selected_rows = random.sample(rows, 2)
            # Add the first selected row to the output DataFrame with type 'query'
            output_df = output_df.append({
                'path': selected_rows[0]['path'],
                'class': selected_rows[0]['class'],
                'label': type_id,
                'type': 'query'
            }, ignore_index=True)
            # Add the second selected row to the output DataFrame with type 'gt'
            output_df = output_df.append({
                'path': selected_rows[1]['path'],
                'class': selected_rows[1]['class'],                
                'label': type_id,
                'type': 'gt',
            }, ignore_index=True)
            # Increment the type_id for uniqueness
            type_id += 1

    df_anno = pd.concat([output_df, df_db], ignore_index=True)
    df_anno.to_csv(anno_file_path, index=False)