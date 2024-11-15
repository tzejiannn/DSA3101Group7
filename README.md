# Synergy Squad
## Overview
- E-commerce platforms face numerous challenges in optimizing their operations, understanding
customer behavior, and maximizing profitability. This project aims to leverage data analysis
techniques to gain insights into various aspects of an e-commerce business, including sales
trends, customer segmentation, inventory management, and pricing strategies. The goal is to develop a comprehensive analysis framework that can help e-commerce
businesses make data-driven decisions to improve their performance, enhance customer
satisfaction, and increase revenue.

## Subgroup A
### **AI Image Synthesis and User Experience**

1. How can we develop an AI model to generate realistic product images based on user customization choices?
- Evaluate different AI image synthesis techniques (e.g., GANs, VAEs, Diffusion Models).
- Develop a prototype model for generating customised product images.
- Assess the quality and realism of generated images.(Ray)

2. What are the key factors influencing user satisfaction with the product customization Experience?
- Analyse user interaction data to identify pain points and preferences in the customization Process. 
- Propose metrics for tracking user satisfaction and engagement. 

3. How can we optimise the user interface to make the customization process intuitive and Enjoyable?
- Develop A/B testing strategies for different UI/UX designs.
- Analyse user behaviour data to identify the most effective customization workflows.

## Subgroup B
### **Inventory Management and Pricing Optimisation**

1. How can we accurately predict demand for customized products? 
- Develop a demand forecasting model using historical data and user customization choices. 
- Identify key factors influencing demand fluctuations for personalized merchandise.

2. What inventory management strategies would optimize stock levels while minimizing costs?
- Create an algorithm to optimize inventory based on predicted demand and customization
trends.
- Simulate the impact of proposed inventory strategies on costs and product availability.

3. How can we implement a dynamic pricing model for customized products?
- Develop a pricing algorithm that accounts for customization complexity, material costs, and
 demand.
- Analyze the potential impact of dynamic pricing on sales and profit margins.

## Setting up
### Setting up UI for AI Image Synthesis
1. Set up React development environment
* Install Node.js
> Go to the Node.js download page and download the LTS version
2. Open UI code. UI code can be downloaded at the code section UI Codes -> UI Codes Subgroup A -> src
3. Open src using editor (eg. Visual Studio Code)
4. Run UI environment
* Type "npm start" in project directory
5. Run app.py to ensure product Recommendation function works

## Repository Structure
- **API/API Subgroup A**: Contains API-related code and resources developed by Subgroup A.
- **Dashboard SubgroupA**: Directory for the dashboard for Subgroup A
- **Data**: Directory for datasets used across the project. 
- **Images Subgroup B**: Backup of image files for reference or documentation.
- **Models**: Trained models and scripts for model development.
- **Subgroup B Streamlit app**: Streamlit application files specific to Subgroup B.
- **UI Codes/UI Codes Subgroup A**: User interface code and assets developed by Subgroup A.
- **Wiki_Images**: Images used for documentation or Wiki resources.
- **sql/sql Subgroup A**: SQL scripts and database-related files for Subgroup A.
- **.DS_Store**: metadata
- **README.md**: Documentation and overview of the project.
- **requirements.txt**: List of dependencies needed for the project.

## Data Sources
### Subgroup A

The data used comes from publically available large-scale Amazon Reviews dataset, collected in 2023 by McAuley Lab, specifically the [Amazon_Fashion](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Amazon_Fashion.jsonl.gz) review data. 

- After downloading, you can run the `Data_Cleaning_and_Preparation.py` script or `Data_Cleaning_and_Preparation.ipynb` notebook under 'Data Subgroup A'. Remeber to change the file path to "/path/to/your/Amazon_Fashion.jsonl"

- When the scirpt is run succesfully, 3 separate folders 
(red_top_images, blue_top_images, green_top_images) containing ~20 resized images will be created and downloaded. This is the dataset we used to train our models.

- Due to the noise in Amazon dataset images, we also created a separate mini dataset consisting of 25 images in the mini_dataset folder. These images were also used to train our models. 

- Due to the size of the trained models being too big, to download the trained models head to the google drive through the link below
https://drive.google.com/drive/folders/1XP5F8W9oHcGiUUgvAoYZLYVqA9ls3e2V?usp=sharing

### Subgroup B
- The data used by Subgroup A to for AI Image Synthesis and User Experience is the [Amazon Product Dataset](https://jmcauley.ucsd.edu/data/amazon/)

- The data used by Subgroup B for Inventory Management and Pricing Optimisation originates from the Inventory and Sales Data from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II). The original dataset has been processed and the transformed data used is in the `store_data.db`. For the processing steps, refer to `data_processing.ipynb` under 'Data Subgroup B'.

## Instructions for Docker
To run the Store Manager streamlit app using Docker, you will need:
- [Docker](https://www.docker.com/products/docker-desktop) installed on your machine.

### Installation and Setup
1. **Clone the repository**
   ```sh
   git clone https://github.com/tzejiannn/DSA3101Group7.git
   cd DSA3101Group7/'Subgroup B Streamlit app'
   ```
2. **Build the Docker image**
   ```sh
   docker build -t streamlit-app .
   ```
3. **Run the Docker container**
   ```sh
   docker run -p 8501:8501 streamlit-app
   ```
4. **Access the Application**

   Open a web browser and go to http://localhost:8501 to view the app.


## API Documentation
API codes can be found under the API folder in the repository.
### Product Recommmendation API
**Technologies Used**
* Flask
* Python
* Cosine Similarity
* Pandas
* Scikit-learn

### VAE Model API
**Technologies Used**
* Flask
* PyTorch
* Torchvision
* NumPy
* PIL

### Web Application API
**Technologies Used**
*tensorflow
*streamlit
*sqlalchemy
*Pandas
*NumPy
*Cosine Similarity
*Scikit-learn

