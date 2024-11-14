We moved the data in the Amazon_Fashion review from a pandas dataframe to a SQL database. 

To replicate, follow the below steps:
1. Execute the Create_Database_and_Table.sql script to create the Amazon_Fashion_Data Database and reviews_table
2. Execute the Populate_DataBase.py script to populate the newly created Database and Table. (Remember to change the MySQL database credentials)
3. Execute the Verify_Data_Load.sql script to verify successful load of thhe data



Why Create a SQL Database to store our data?

We have also wrote a Example_Queries_for_Future_Business_Uses.sql script that contains 10 example queries that extract data for many different business related purposes/ that can be used to answer many different business questions. This example script is to show the benefits of having stored our data in a SQL database. 

Unlike in-memory DataFrames, SQL databases can scale to accommodate growing datasets while maintaining performance. This setup ensures data consistency, reduces redundancy, and leverages powerful SQL features to streamline data analysis tasks, making it easier to extract valuable insights for model training and ongoing analytics directly from a robust, centralized source.  