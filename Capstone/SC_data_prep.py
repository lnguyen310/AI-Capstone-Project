#####
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

# loading datasets
# RAW_recipes.csv : basic recipe info (name, ingredients, steps, time, etc.)
raw_recipes = pd.read_csv("RAW_recipes.csv")
# RAW_interactions.csv : user interactions (ratings, reviews, etc.)
raw_interactions = pd.read_csv("RAW_interactions.csv")
# epi_r.csv : nutrition info
epi_recipes = pd.read_csv("epi_r.csv")

# getting dataset shapes
print("Datasets Shape:")
print("Food.com recipes:", raw_recipes.shape)
print("Interactions:", raw_interactions.shape)
print("Epicurious recipes:", epi_recipes.shape)

# removing duplicates
raw_recipes.drop_duplicates(inplace=True)
epi_recipes.drop_duplicates(inplace=True)

# lowercase for consistency
raw_recipes['ingredients'] = raw_recipes['ingredients'].str.lower().str.strip()
epi_recipes['title'] = epi_recipes['title'].str.lower().str.strip()

#####
# part 2: nutrition lookup (using only epi_recipes)
#####
print("-" * 60)

# function to normalize ingredient names
# removes punctuation and common descriptors
def normalize_name(name):
    name = name.lower().strip()
    for char in [',', '.', '(', ')', '[', ']', '-']:
        name = name.replace(char, '')
    for word in ["fresh", "chopped", "sliced", "diced", "optional", "small", "large"]:
        name = name.replace(word, '')
    return name.strip()

# normalize titles 
title_norm_list = []
for title in epi_recipes['title']:
    clean_title = normalize_name(title)
    title_norm_list.append(clean_title)

epi_recipes['title_norm'] = title_norm_list

# keep only nutrition-related columns that excist
nutrition_list = ['calories', 'protein', 'fat', 'fiber', 'sodium', 'cholesterol', 'potassium']
epi_nutrition_cols = []

for col in nutrition_list:
    if col in epi_recipes.columns:
        epi_nutrition_cols.append(col)

print("Nutrition columns found:", epi_nutrition_cols)

#####
# part 3: average ratings
#####
print("-" * 60)

# make sure IDs match
raw_interactions['recipe_id'] = raw_interactions['recipe_id'].astype(raw_recipes['id'].dtype)

# getting the average rating per recipe
rating_sum = {}
rating_count = {}

for i, row in raw_interactions.iterrows():
    rid = row['recipe_id']
    rate = row['rating']
    # initialize if not present
    if rid not in rating_sum:
        rating_sum[rid] = 0
        rating_count[rid] = 0
    # adding current rating to total
    rating_sum[rid] += rate
    rating_count[rid] += 1

# getting average ratings
ratings_avg = {}
for rid in rating_sum:
    avg = rating_sum[rid] / rating_count[rid]
    ratings_avg[rid] = avg

print("Average ratings calculated for", len(ratings_avg), "recipes.")

#####
# part 4: building master recipe dataset 
#####
print("-" * 60)

master_recipes = []
count = 0

# looping through each recipe in raw_recipes dataset
for idx, row in raw_recipes.iterrows():
    # progress update every 1000 rows so I know it's working
    if idx % 1000 == 0:
        print(f"Processed {idx} recipes...")

    # creating new recipe dictionary
    recipe_id = row['id']
    recipe_info = {
        'recipe_id': recipe_id,
        'title': row['name'],
        'ingredients': row['ingredients'],
        'instructions': row['steps'],
        'time': row['minutes'],
    }

    # find average rating 
    if recipe_id in ratings_avg:
        recipe_info['rating'] = ratings_avg[recipe_id]
    else:
        recipe_info['rating'] = 0

    # set up blank nutrition dictionary
    total_nutrition = {}
    for col in epi_nutrition_cols:
        total_nutrition[col] = 0

    # finding nutrition match in epi_recipes
    for ingredient in row['ingredients'].split(','):
        ingredient_norm = normalize_name(ingredient)
        # chekcing if ingredients are also in epi_recipes titles
        matches = epi_recipes[epi_recipes['title_norm'].str.contains(ingredient_norm, na=False)]
        
        # ingredient match found, take average nutrition value
        if not matches.empty:
            for nut_key in epi_nutrition_cols:
                avg_value = matches[nut_key].mean()
                total_nutrition[nut_key] += avg_value

    # add nutrition info to recipe info
    for nut_key in total_nutrition:
        recipe_info[nut_key] = total_nutrition[nut_key]

    # adding to recipe dictionary to list 
    master_recipes.append(recipe_info)
    count += 1

print("Finished processing:", count, "recipes.")

# create dataframe
master_df = pd.DataFrame(master_recipes)

# save to CSV
master_df.to_csv("master_recipes.csv", index=False)

# quick dataset summary
print("-" * 60)
print("The number of master recipes:", len(master_df))
print("Sample of master dataset:")
print(master_df.head(5))