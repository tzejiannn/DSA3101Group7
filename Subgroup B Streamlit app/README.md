# Store Manager Application

## Product Overview
The Store Manager Dashboard is an AI-powered application designed to streamline store management through data-driven insights.
It integrates forecasting, inventory management, and pricing optimisation to assist store managers in making informed decisions 
for better operational efficiency and profitability.

## Folder Structure

- **Dockerfile**: The configuration file for building the Docker container.
- **README.md**: Provides an overview and instructions for the application.
- **requirements.txt**: Contains a list of required Python packages to run the application.
- **store_data.db**: The database used by the application.
- **store_manager.py**: The main file for the application.
- **store_manager_functions.py**: Helper functions used by `store_manager.py`.

## Data Sources
- The data used originates from the Inventory and Sales Data from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II).
The original dataset has been processed and the transformed data used is in the `store_data.db`.

## Requirements

To run this project using Docker, you will need:
- [Docker](https://www.docker.com/products/docker-desktop) installed on your machine.

## Installation and Setup
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

## Usage
Once the app is running, you can use the following features:

1. **Product Selection**:
   - Use the dropdown menu labeled **Select Product** to choose the product you want to analyse.
   - After selecting a product, use the **Select Variation Detail** dropdown to choose specific variations related to the product.

3. **Forecasted Demand**:
   - The **Forecasted Demand** section displays a table showing the predicted quantity for the selected product and variation over time.

4. **Optimised Pricing**:
   - The **Optimised Pricing** section provides recommendations for pricing adjustments based on forecasted demand.
   - This table includes:
     - **Date**: The date of the forecast.
     - **Predicted Quantity**: Forecasted demand quantity.
     - **Optimised Price**: Suggested price for the product to optimise revenue.
     - **Optimised Predicted Revenue**: Predicted revenue based on the optimised price.

5. **Demand Forecast**:
   - This line chart visualizes the **forecasted demand** over time, showing units sold.
   - It helps identify trends in demand for better inventory planning.

6. **Revenue Forecast**:
   - This chart displays the **forecasted revenue** over time, helping to assess the potential financial performance.

7. **Price Adjustment**:
   - The **Price Adjustment** chart shows adjustments in the recommended unit price over time.
   - This visualization helps in understanding pricing strategies and their effects on demand and revenue.
