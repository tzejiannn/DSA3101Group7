The data used in this project comes from publically available large-scale Amazon Reviews dataset, collected in 2023 by McAuley Lab, specifically the Amazon_Fashion review data. 

Given that the dataset we used is >100 mb and exceeds the git allowed limit, we are unable to upload the data in this folder. As such, please download the dataset we used via the link below: 

https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Amazon_Fashion.jsonl.gz


After downloading, you can run the Data_Cleaning_and_Preparation.py script or Data_Cleaning_and_Preparation.ipynb notebook. Remeber to change the file path to "/path/to/your/Amazon_Fashion.jsonl"


When the scirpt is run succesfully, 3 separate folders 
(red_top_images, blue_top_images, green_top_images) containing ~20 resized images will be created and downloaded. This is the dataset we used to train our models.


Due to the noise in Amazon dataset images, we also created a separate mini dataset consisting of 25 images in the mini_dataset folder. These images were also used to train our models. 

Due to the size of the trained models being too big, to download the trained models head to the google drive through the link below
https://drive.google.com/drive/folders/1XP5F8W9oHcGiUUgvAoYZLYVqA9ls3e2V?usp=sharing
