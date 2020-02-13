
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3

# Connect to the MIMIC database
conn = sqlite3.connect('data/mimicdata.sqlite')

# Create our test query
test_query = """
SELECT subject_id, hadm_id, admittime, dischtime, admission_type, diagnosis
FROM admissions
"""
## Run the query and assign the results to a variable
test = pd.read_sql_query(test_query,conn)
#
## Display the first few rows
test.head()