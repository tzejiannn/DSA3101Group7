#!/usr/bin/env python
# coding: utf-8

# Import libraries

# In[1]:


import pandas as pd


# In[ ]:


file = "/Users/kc/Desktop/DSA3101/project/Amazon_fashion/Amazon_Fashion.jsonl"  # change to /path/to/your/Amazon_Fashion.jsonl
# Converting datatype to match the fields specified in Amazon Reviews'23
dtype = {"rating": float, 
         "title": str, 
         "text": str, 
         "images": list, 
         "asin": str, 
         "parent_asin": str, 
         "user_id": str}
reviews = pd.read_json(file, lines = True, dtype = dtype)


# In[ ]:


print(reviews["user_id"].head())
print(type(reviews["user_id"][1])) # correct type: str


# In[ ]:


print(type(reviews["images"][1])) # correct type: list


# In[ ]:


renamed_reviews = reviews.rename(columns={"asin":"product_id"})
print(renamed_reviews.dtypes)


# In[ ]:


print(reviews["asin"].head())
print(renamed_reviews["product_id"].head())


# In[ ]:


renamed_reviews = renamed_reviews.rename(columns={"timestamp":"date"})
renamed_reviews['date'] = pd.to_datetime(renamed_reviews['date']).dt.date


# In[ ]:


renamed_reviews = renamed_reviews.rename(columns={"parent_asin":"group_id"})


# In[ ]:


renamed_reviews = renamed_reviews[renamed_reviews['verified_purchase'] == True]
renamed_reviews = renamed_reviews.dropna()
print(renamed_reviews.dtypes)
print(renamed_reviews.head())









# In[ ]:


def extract_medium_image_url(images):
    if isinstance(images, list) and len(images) > 0:
        return images[0].get('medium_image_url')  # Assuming we only need the first image's medium URL
    return None

# Apply the function to the DataFrame
renamed_reviews['medium_image_url'] = renamed_reviews['images'].apply(extract_medium_image_url)
renamed_reviews = renamed_reviews.drop(columns=['images'])
print(renamed_reviews.head())


# In[ ]:


from sqlalchemy import create_engine
import pandas as pd

# Define your MySQL database credentials
username = 'root'
password = 'Powq09210939z'
host = 'localhost'  # usually 'localhost' if running locally
port = '3306'  # default MySQL port
database = 'Amazon_Fashion_Data'

# Create the database connection string
engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}')



# Export the DataFrame to the MySQL database
renamed_reviews.to_sql(name='reviews_table', con=engine, if_exists='replace', index=False)

print("DataFrame successfully saved to MySQL database.")


# In[ ]:


row_count = len(renamed_reviews)
print(f"Number of rows in the DataFrame: {row_count}")


# In[ ]:




