-- Query 1: Retrieve All Reviews for a Specific Product
SELECT rating, title, text, user_id, date, helpful_vote, verified_purchase
FROM reviews_table
WHERE product_id = 'B091GMMYPS';


-- Query 2: Filter Reviews with High Ratings (4 and above)
SELECT rating, title, text, product_id, date
FROM reviews_table
WHERE rating >= 4;


-- Query 3: Find Most Helpful Reviews (those with more than 5 helpful votes)
SELECT product_id, rating, title, text, user_id, helpful_vote
FROM reviews_table
WHERE helpful_vote > 5
ORDER BY helpful_vote DESC;


-- Query 4: Count Verified Purchases by Product
SELECT product_id, COUNT(*) AS verified_purchase_count
FROM reviews_table
WHERE verified_purchase = TRUE
GROUP BY product_id
ORDER BY verified_purchase_count DESC;


-- Query 5: Summarize Ratings by Product (Average Rating and Review Count)
SELECT product_id, AVG(rating) AS avg_rating, COUNT(*) AS review_count
FROM reviews_table
GROUP BY product_id
ORDER BY avg_rating DESC;


-- Query 6: Extract Reviews within a Specific Date Range (2023)
SELECT product_id, rating, title, text, date
FROM reviews_table
WHERE date BETWEEN '2023-01-01' AND '2023-12-31';


-- Query 7: Find Top-Rated Products in Each Group
SELECT group_id, product_id, AVG(rating) AS avg_rating
FROM reviews_table
GROUP BY group_id, product_id
ORDER BY group_id, avg_rating DESC;


-- Query 8: Retrieve Images for Products with High Ratings (4 and above)
SELECT product_id, medium_image_url
FROM reviews_table
WHERE rating >= 4;


-- Query 9: Identify Active Reviewers (Users who have written more than 5 reviews)
SELECT user_id, COUNT(*) AS review_count
FROM reviews_table
GROUP BY user_id
HAVING review_count > 5
ORDER BY review_count DESC;


-- Query 10: Analyze Rating Distribution by Product
SELECT product_id, rating, COUNT(*) AS count
FROM reviews_table
GROUP BY product_id, rating
ORDER BY product_id, rating;
