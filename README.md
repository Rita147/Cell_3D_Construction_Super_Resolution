🧪 Super-Resolution Methods Used
We applied several SR methods to enhance histopathology images before segmentation and 3D reconstruction:

✅ Pretrained Models

EDSR (OpenCV): Used OpenCV’s built-in LapSRN model with pretrained weights.


LapSRN (OpenCV): Used OpenCV’s built-in LapSRN model with pretrained weights.

🔧 Custom-Trained Models
SRCNN: Implemented and trained from scratch.

ESPCN: Implemented from scratch based on the original paper.

EDSR: Tried both custom-trained and OpenCV pretrained versions.

SRDFN: Fully implemented and trained for this project.

All models were evaluated using PSNR/SSIM. EDSR and SRDFN gave the best results for structure preservation.
