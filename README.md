# Download Kinetics Dataset

Kinetics is a collection of large-scale, high-quality datasets of URL links of up to 650,000 video clips that cover 400/600/700 human action classes, depending on the dataset version. The videos include human-object interactions such as playing instruments, as well as human-human interactions such as shaking hands and hugging. Each action class has at least 400/600/700 video clips. Each clip is human annotated with a single action class and lasts around 10 seconds.

The Kinetics project publications can be found here: https://deepmind.com/research/open-source/kinetics.

The Kinetics dataset can be downloaded here: https://github.com/cvdfoundation/kinetics-dataset.

# SFD generator

Following the next steps to generate Similar Frame Datasets from Kinetics-700!

### Clone repo and enter directory

```
git clone https://github.com/cvdfoundation/kinetics-dataset.git
cd make_dataset
```

Randomly sample 2 frames from each video in the test test of Kinetics-700 to get distraction images for SFD.
```
python sample_frame_from_testset.py
```

Uniformly sample 8 frames from each video in the validation set of Kinetics-700 to get query-target candidates for SFD.
```
python sample_frames_from_valset.py
```

Generate annotation file (.csv format) of query-target candidates.
```
python get_img_path_and_class.py
```

Check whether human is in the image using Faster R-CNN
```
python get_img_path_and_class.py
```

Ask blip2 whether key information is included in the frame
```
python blip2_answer_for_sfd.py
```

Generate the final annotation file for SFD.
```
python generate_final_val_file.py
```

# Train a model under AFCL framework

change your config here
```
afcl_config.py
```

start training
```
bash train_afcl.sh
```





