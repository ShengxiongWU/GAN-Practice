# GAN-Practice

### How to train:
1. Download `Anime-Face.ipynb` file
2. Find your own dataset and load it in this section:
    ```python
    # Load the dataset
    image_dir1 = os.path.join('anime-faces/anime-faces')
    dataset1 = CustomDataset(image_dir1, transform=transform)
    
    # Load the dataset
    image_dir2 = os.path.join('faces/faces')
    dataset2 = CustomDataset(image_dir2, transform=transform)
    ```
3. Run it, the result will be in the output folder.

### How to inference:
1. Download `inference.py` file and `generator_epoch_50.pth` weight file in weight folder.
2. Run it, the result will be in the `inferenced_images` folder.

### Sample inference output:
![](inferenced_images/generated_image_0.png)
![](inferenced_images/generated_image_1.png)
![](inferenced_images/generated_image_2.png)
![](inferenced_images/generated_image_3.png)
![](inferenced_images/generated_image_4.png)
![](inferenced_images/generated_image_5.png)
![](inferenced_images/generated_image_6.png)
![](inferenced_images/generated_image_7.png)
![](inferenced_images/generated_image_8.png)
![](inferenced_images/generated_image_9.png)
![](inferenced_images/generated_image_10.png)
![](inferenced_images/generated_image_11.png)
![](inferenced_images/generated_image_12.png)
![](inferenced_images/generated_image_13.png)
![](inferenced_images/generated_image_14.png)
![](inferenced_images/generated_image_15.png)
![](inferenced_images/generated_image_16.png)
![](inferenced_images/generated_image_17.png)
![](inferenced_images/generated_image_18.png)
![](inferenced_images/generated_image_19.png)
![](inferenced_images/generated_image_20.png)
![](inferenced_images/generated_image_21.png)
![](inferenced_images/generated_image_22.png)
![](inferenced_images/generated_image_23.png)
![](inferenced_images/generated_image_24.png)
![](inferenced_images/generated_image_25.png)
![](inferenced_images/generated_image_26.png)
![](inferenced_images/generated_image_27.png)
![](inferenced_images/generated_image_28.png)
![](inferenced_images/generated_image_29.png)
![](inferenced_images/generated_image_30.png)
![](inferenced_images/generated_image_31.png)
![](inferenced_images/generated_image_32.png)
![](inferenced_images/generated_image_33.png)
![](inferenced_images/generated_image_34.png)
![](inferenced_images/generated_image_35.png)
![](inferenced_images/generated_image_36.png)
![](inferenced_images/generated_image_37.png)
![](inferenced_images/generated_image_38.png)
![](inferenced_images/generated_image_39.png)
![](inferenced_images/generated_image_40.png)
![](inferenced_images/generated_image_41.png)


