**Super Resolution Model Evaluation**
**Dataset and Preprocessing**
The original image dimensions were 512x512. To better observe the effect of super resolution, the images were downsampled to 256x256 using a downsampling technique. All models were trained and evaluated on these downsampled images.

**Trained Models**
The following models were trained for 50 epochs:

SRDF

SRCNN

EDSR

ESPCN

The trained models are saved and available for use.

**Pre-trained Models**
In addition to the trained models, two pre-trained models—LapSRN and EDSR—were also tested.

**Results**
The EDSR model provided the best results among all tested models. Both the OpenCV pre-trained EDSR model and the custom-trained EDSR model produced highly successful outcomes.

**Usage**
All models are available and ready to use in the ready_whole_models directory.
This directory also includes example usage scripts to help you get started quickly.
