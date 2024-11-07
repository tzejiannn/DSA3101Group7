#Verify the data is being loaded fully/correctly 

SELECT COUNT(*) FROM reviews_table; #should be 2337702


SELECT COUNT(*) FROM reviews_table WHERE medium_image_url IS NOT NULL; #should be 136265