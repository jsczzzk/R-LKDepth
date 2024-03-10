





# R-LKDepth: Recurrent Depth Learning With Larger Kernel (SPL)
This is the official implementation for testing depth estimation using the model proposed in 
>R-LKDepth: Recurrent Depth Learning With Larger Kernel


R-LKDepth can estimate a depth map from a single image.

![image](https://github.com/jsczzzk/R-LKDepth/assets/32475718/ae6e3bc1-889a-4b9d-b6cc-5ce973931661)

![image](https://github.com/jsczzzk/R-LKDepth/assets/32475718/42aa5f99-31bd-4540-80f5-b3cd0ee8bf09)
![image](https://github.com/jsczzzk/R-LKDepth/assets/32475718/b66d3720-5a24-4854-a690-5da53a687f2c)




## Pretrained Models
We have updated all the results as follows:
[models](https://drive.google.com/drive/folders/13C2A0yZMEg0pirw96glach_FTauEFDBN?usp=sharing)

## KITTI Evaluation
You can predict scaled disparity for a single image used R-LKDepth with:
```shell
python test_simple.py --image_path='path_to_image' --model_path='path_to_model' 



