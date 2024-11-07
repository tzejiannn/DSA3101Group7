-- Create the database if it doesn't exist
CREATE DATABASE IF NOT EXISTS Amazon_Fashion_Data;

-- Select the database to use
USE Amazon_Fashion_Data;

-- Create the table to store reviews
CREATE TABLE IF NOT EXISTS reviews_table (
    id INT AUTO_INCREMENT PRIMARY KEY,
    rating DECIMAL(2, 1),
    title VARCHAR(255),
    text TEXT,
    product_id VARCHAR(20),
    group_id VARCHAR(20),
    user_id VARCHAR(100),
    date DATE,
    helpful_vote INT,
    verified_purchase BOOLEAN,
    medium_image_url VARCHAR(255)
);

