#!/usr/bin/env python
# coding: utf-8

# Import libraries

# In[2]:


import pandas as pd


# In[3]:


file = "/Users/kc/Desktop/DSA3101/project/Amazon_fashion/Amazon_Fashion.jsonl"
# Converting datatype to match the fields specified in Amazon Reviews'23
dtype = {"rating": float, 
         "title": str, 
         "text": str, 
         "images": list, 
         "asin": str, 
         "parent_asin": str, 
         "user_id": str}
reviews = pd.read_json(file, lines = True, dtype = dtype)


# In[4]:


print(reviews.shape)
print(reviews.dtypes)


# ![image.png](attachment:image.png)

# In[5]:


print(reviews.tail())


# In[6]:


print(reviews["timestamp"].head())
print(type(reviews["timestamp"][1]))


# In[7]:


print(reviews["user_id"].head())
print(type(reviews["user_id"][1])) # correct type: str


# In[8]:


print(type(reviews["images"][1])) # correct type: list


# In[9]:


renamed_reviews = reviews.rename(columns={"asin":"product_id"})
print(renamed_reviews.dtypes)


# In[38]:


print(reviews["asin"].head())
print(renamed_reviews["product_id"].head())


# In[39]:


renamed_reviews = renamed_reviews.rename(columns={"timestamp":"date"})
renamed_reviews['date'] = pd.to_datetime(renamed_reviews['date']).dt.date


# In[40]:


renamed_reviews = renamed_reviews.rename(columns={"parent_asin":"group_id"})


# In[41]:


renamed_reviews = renamed_reviews[renamed_reviews['verified_purchase'] == True]
renamed_reviews = renamed_reviews.dropna()


# In[42]:


import json 
pprint = lambda x: print(json.dumps(x, indent=2)) if isinstance(x, dict) else display(x)

r_renamed_reviews = renamed_reviews[renamed_reviews['images'].apply(lambda x: len(x) > 0)] #To remove rows without image links


# In[52]:


print(r_renamed_reviews.head())
row_count = len(r_renamed_reviews)
print(f"Number of rows in the DataFrame: {row_count}")


# In[54]:


# Filter the DataFrame to only include 'product_id', 'title', and 'images' columns
filtered_df = r_renamed_reviews[['product_id', 'title', 'images']]

# Display the filtered DataFrame
print(filtered_df.head(10))


# In[55]:


# Function to extract the 'medium_image_url' from the 'images' column
def extract_medium_image_url(images):
    if isinstance(images, list) and len(images) > 0:
        return images[0].get('medium_image_url')  # Assuming we only need the first image's medium URL
    return None

# Apply the function to the DataFrame
filtered_df['medium_image_url'] = filtered_df['images'].apply(extract_medium_image_url)

# Keep only the necessary columns: 'product_id', 'title', 'medium_image_url'
final_df = filtered_df[['product_id', 'title', 'medium_image_url']]

# Display the first few rows of the updated DataFrame
print(final_df.head(10))

# Alternatively, count using len()
num_entries = len(final_df)
print(f'The number of entries in final_df is: {num_entries}')


# In[62]:


import pandas as pd
import re

# Assuming final_df is already defined and contains the necessary data

# Define a function to filter for red shirts
def filter_red_tops(df):
    # Create a regex pattern for various forms of 'red' related to shirts
    pattern = r'\b(red|crimson|burgundy|scarlet|ruby|cardinal|cherry|wine)\b.*\b(shirt|t-shirt|tee|top|sweatshirt|blouse|dress)\b'
    return df[df['title'].str.contains(pattern, case=False, na=False, regex=True)]

# Apply the filter function
red_tops_df = filter_red_shirts(final_df)

# Display the first few rows of the filtered DataFrame
print(red_tops_df.head(20))

# Count the number of entries for red shirts
num_red_tops = len(red_tops_df)
print(f'The number of red tops in the dataset is: {num_red_tops}')


# In[64]:


import pandas as pd
import re

# Assuming final_df is already defined and contains the necessary data

# Define a function to filter for blue tops
def filter_blue_tops(df):
    # Create a regex pattern for various forms of 'blue' related to tops
    pattern = r'\b(blue|navy|azure|cobalt|sky|teal|cerulean|sapphire)\b.*\b(top|t-shirt|tee|sweatshirt|blouse|dress)\b'
    return df[df['title'].str.contains(pattern, case=False, na=False, regex=True)]

# Apply the filter function
blue_tops_df = filter_blue_tops(final_df)

# Display the first few rows of the filtered DataFrame
print(blue_tops_df.head(20))

# Count the number of entries for blue tops
num_blue_tops = len(blue_tops_df)
print(f'The number of blue tops in the dataset is: {num_blue_tops}')


# In[66]:


import pandas as pd
import re

# Assuming final_df is already defined and contains the necessary data

# Define a function to filter for green tops
def filter_green_tops(df):
    # Create a regex pattern for various forms of 'green' related to tops
    pattern = r'\b(green|lime|olive|emerald|forest|mint|jade|teal)\b.*\b(top|t-shirt|tee|sweatshirt|blouse|dress)\b'
    return df[df['title'].str.contains(pattern, case=False, na=False, regex=True)]

# Apply the filter function
green_tops_df = filter_green_tops(final_df)

# Display the first few rows of the filtered DataFrame
print(green_tops_df.head(20))

# Count the number of entries for green tops
num_green_tops = len(green_tops_df)
print(f'The number of green tops in the dataset is: {num_green_tops}')


# In[82]:


# Create mini dataset for red tops
red_tops_mini_dataset = red_tops_df.head(20)
print("Red Tops Mini Dataset:")
print(red_tops_mini_dataset)

# Create mini dataset for green tops
green_tops_mini_dataset = green_tops_df.head(20)
print("\nGreen Tops Mini Dataset:")
print(green_tops_mini_dataset)

# Create mini dataset for blue tops
blue_tops_mini_dataset = blue_tops_df.head(20)
print("\nBlue Tops Mini Dataset:")
print(blue_tops_mini_dataset)


# In[83]:


import os
import requests

def download_images(df, folder_name):
    # Create a folder to save images
    os.makedirs(folder_name, exist_ok=True)
    
    for index, row in df.iterrows():
        image_url = row['medium_image_url']
        try:
            # Send a GET request to the image URL
            response = requests.get(image_url)
            if response.status_code == 200:  # Check if the request was successful
                # Save the image
                image_path = os.path.join(folder_name, f"{row['product_id']}.jpg")
                with open(image_path, 'wb') as f:
                    f.write(response.content)
            else:
                print(f"Failed to download image from {image_url} with status code {response.status_code}")
        except Exception as e:
            print(f"Error downloading {image_url}: {e}")

# Example usage
download_images(red_tops_mini_dataset, "red_tops_images")
download_images(blue_tops_mini_dataset, "blue_tops_images")
download_images(green_tops_mini_dataset, "green_tops_images")


# In[90]:


from PIL import Image
import numpy as np
import glob
import os

def preprocess_images(folder_name, target_size=(128, 128)):
    # Check if there are any images in the folder
    image_paths = glob.glob(os.path.join(folder_name, "*.jpg"))
    if not image_paths:
        print(f"No images found in {folder_name}. Please check the folder path and file extensions.")
        return

    for image_path in image_paths:
        try:
            # Open an image file
            with Image.open(image_path) as img:
                print(f"Processing image: {image_path}")  # Debug: Track image being processed
                
                # Resize image with high-quality interpolation to reduce blurriness
                img = img.resize(target_size, Image.LANCZOS)
                
                # Convert image to NumPy array and normalize (if needed)
                img_array = np.array(img) / 255.0
                
                # Convert back to image format for saving
                img = Image.fromarray((img_array * 255).astype('uint8'))
                
                # Save the processed image (overwriting original)
                img.save(image_path)
                print(f"Overwritten image: {image_path}")  # Debug: Confirm image overwrite

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

# Example usage for each color folder with a higher quality interpolation
preprocess_images("red_tops_images", target_size=(512, 512))  # Choose (128, 128) or (256, 256) as needed
preprocess_images("green_tops_images", target_size=(512, 512))
preprocess_images("blue_tops_images", target_size=(512, 512))



