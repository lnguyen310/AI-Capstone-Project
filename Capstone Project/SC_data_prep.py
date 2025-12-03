# Capstone Project: ScrappyChef
# Data Preparation
#####

# importing libraries 
import pandas as pd
import random

#####
# part 1: loading & cleaning datasets
#####
print("-" * 60)

# loading dataset
raw_recipes = pd.read_csv("RAW_recipes.csv")

# getting dataset shape
print("Dataset Shape:")
print("Food.com recipes:", raw_recipes.shape)

# removing duplicates
raw_recipes.drop_duplicates(inplace=True)

# lowercase for consistency
raw_recipes['ingredients'] = raw_recipes['ingredients'].fillna("").str.lower().str.strip()
raw_recipes['name'] = raw_recipes['name'].fillna("").str.lower().str.strip()

#####
# function to normalize names
#####
def normalize_name(name):
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    for char in [',', '.', '(', ')', '[', ']', '-']:
        name = name.replace(char, '')
    for word in ["fresh", "chopped", "sliced", "diced", "optional", "small", "large"]:
        name = name.replace(word, '')
    return name.strip()

#####
# part 2: building master recipe dataset 
#####
print("-" * 60)

master_recipes = []
count = 0
skipped_count = 0 

# looping through each recipe in raw_recipes dataset
for idx, row in raw_recipes.iterrows():
    # progress update every 1000 rows
    if idx % 1000 == 0:
        print(f"Processed {idx} recipes...")

    # filter out recipes that take longer than 30 minutes
    if pd.notna(row['minutes']) and row['minutes'] > 30:
        skipped_count += 1
        continue  

    # normalize title safely
    title_clean = normalize_name(row['name'])

    # normalize ingredients safely
    ingredients_clean = ""
    if pd.notna(row['ingredients']):
        ingredients_list = [normalize_name(i) for i in row['ingredients'].split(',')]
        ingredients_clean = ', '.join(ingredients_list)

    # creating new recipe dictionary
    recipe_info = {
        'recipe_id': row['id'],
        'title': title_clean,
        'ingredients': ingredients_clean,
        'instructions': row['steps'] if pd.notna(row['steps']) else "",
        'time': row['minutes'] if pd.notna(row['minutes']) else None,
    }

    # adding to recipe dictionary to list 
    master_recipes.append(recipe_info)
    count += 1

print("Finished processing:", count, "recipes.")
print("Skipped recipes >30 mins:", skipped_count)

# create dataframe
master_df = pd.DataFrame(master_recipes)

# save to CSV
master_df.to_csv("master_recipes.csv", index=False)

# quick dataset summary
print("-" * 60)
print("The number of master recipes:", len(master_df))
print("Sample of master dataset:")
print(master_df.head(5))
