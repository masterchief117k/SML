import pandas as pd

# Read the existing data from person_details.csv
existing_data = pd.read_csv('person_details.csv')

# Add data for my friend
friend_data = {
    'Name': ['John'],
    'Gender': ['Male'],
    'Age': [28],
    'City': ['Seattle'],
    'Living Expenses': [3200]
}
friend_df = pd.DataFrame(friend_data)

# Add data for myself
my_data = {
    'Name': ['diloKaShooter'],
    'Gender': ['same as your dad'],
    'Age': [1000],  # Just for fun!
    'City': ['your moms house '],
    'Living Expenses': [1000000000000000]  # I don't need money, I'm digital! 😉
}
my_df = pd.DataFrame(my_data)

# Concatenate the new data with the existing data
updated_data = pd.concat([existing_data, friend_df, my_df], ignore_index=True)

# Print the last five rows
print(updated_data.tail())