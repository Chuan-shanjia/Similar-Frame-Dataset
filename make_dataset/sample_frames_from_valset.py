import cv2
import os
import numpy as np
import concurrent.futures
import albumentations as A

class VideoFrameExtractor:
    def __init__(self, output_size=(224, 224)):
        self.output_size = output_size
        # Define data augmentation transformations
        self.transform = A.Compose([
            A.OneOf([
                A.HorizontalFlip(p=0.9),
                A.RandomResizedCrop(height=self.output_size[0], width=self.output_size[1],
                                    scale=(0.08, 1.0), ratio=(1. / 2., 2. / 1.), p=0.9),
                A.GaussianBlur(p=0.9),
                A.ColorJitter(p=0.9),
            ], p=0.5),
            # Resize after augmentation
            A.Resize(height=self.output_size[0], width=self.output_size[1]),
        ])

    def apply_transforms(self, frame):
        # Apply data augmentation
        augmented = self.transform(image=frame)
        return augmented['image']

    def extract_random_frames(self, video_path, output_base_dir):
        video_name = os.path.basename(video_path).split('.')[0]
        video_class = os.path.basename(os.path.dirname(video_path))
        output_dir = os.path.join(output_base_dir, video_class, video_name)

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, 8, dtype=int)
        for i, frame_index in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Could not read frame {frame_index} from video {video_path}.")
                continue

            # Apply transformations to the frame
            frame_transformed = self.apply_transforms(frame)

            # Save the transformed frame
            frame_filename = os.path.join(output_dir, f'{i+1}.jpg')
            cv2.imwrite(frame_filename, frame_transformed)

        cap.release()

def process_video(video_path, output_base_dir):
    extractor = VideoFrameExtractor()
    extractor.extract_random_frames(video_path, output_base_dir)

def process_videos_in_directory(video_dir, output_base_dir):
    # Get all the .mp4 files in the directory
    video_classes = os.listdir(video_dir)
    
    for video_class in video_classes:
        class_path = os.path.join(video_dir, video_class)
        video_files = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.mp4')]
        # Run the frame extraction function for each video file
        for video_file in video_files:
            process_video(video_file, output_base_dir)
        print(f'{video_class} finished')

if __name__ == "__main__":
    video_dir_all = ''  # Replace with your dir of K700's validation set
    output_base_dir = ''  # Replace with your dir to save query-target candidate images

    # Ensure the output directory exists
    os.makedirs(output_base_dir, exist_ok=True)

    # Process the videos and save the frames in the designated output directory
    process_videos_in_directory(video_dir_all, output_base_dir) 