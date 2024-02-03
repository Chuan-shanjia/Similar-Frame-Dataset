import cv2
import os
import random
import albumentations as A

class VideoFrameExtractor:
    def __init__(self, output_size=(224, 224)):
        self.output_size = output_size
        # define data augmentation
        self.transform = A.Compose([
            A.OneOf([
                A.HorizontalFlip(p=0.9),
                A.RandomResizedCrop(height=self.output_size[0], width=self.output_size[1],
                scale=(0.08, 1.0), ratio=(1. / 2., 2. / 1.), p=0.9),
                A.GaussianBlur(p=0.9),
                A.ColorJitter(p=0.9),
            ], p=0.5),
            # resize
            A.Resize(height=self.output_size[0], width=self.output_size[1]),
        ])

    def apply_transforms(self, frame):
        # augmentation
        augmented = self.transform(image=frame)
        return augmented['image']

    def extract_random_frames(self, video_path, output_dir_base):
        video_name = os.path.basename(video_path).split('.')[0]
        os.makedirs(output_dir_base, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_name}.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = random.sample(range(total_frames), 2)

        for i, frame_index in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Could not read frame {frame_index} from video {video_name}.")
                continue

            # resize
            frame_resized = cv2.resize(frame, self.output_size)
            
            # data augmentation
            frame_augmented = self.apply_transforms(frame_resized)

            # save frames
            if i == 0:
                frame_filename = os.path.join(output_dir_base, f'{video_name}_1.jpg')
            else:
                frame_filename = os.path.join(output_dir_base, f'{video_name}_2.jpg')

            cv2.imwrite(frame_filename, frame_augmented)

        cap.release()

def process_videos_in_directory(video_dir, output_dir):
    extractor = VideoFrameExtractor()
    # get mp4 files in video_dir
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    
    # sample frames from each video
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        extractor.extract_random_frames(video_path, output_dir)
        print(f"Finished processing {video_file}")

if __name__ == "__main__":
    video_dir = ''  # Replace with your dir of K700's test set
    output_dir = ''  # Replace with your dir to save distraction images
    process_videos_in_directory(video_dir, output_dir)
