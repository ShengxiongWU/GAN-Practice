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
1. Download `inference.py` file and `generator_epoch_50.pth` weight file.
2. Run it, the result will be in the `inferenced_images` folder.
