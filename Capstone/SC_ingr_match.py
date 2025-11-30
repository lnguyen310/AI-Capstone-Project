#####
# Capstone Project: ScrappyChef
# Recipe & Nutrition Recommendations
#####

# importing libraries
import pandas as pd
from rapidfuzz import fuzz, process
import nltk
from nltk.corpus import wordnet

# downloading WordNet data
nltk.download('wordnet')
nltk.download('omw-1.4')

# loading datasets
# recipe dataset
master_df = pd.read_csv("master_recipes.csv")
# nutrition dataset
nutrition_df = pd.read_csv("cleaned_nutrition_dataset_per100g.csv")

####
# part 1: cleaning columns
####
# making columns lowercase, removing spaces and brackets 
nutrition_df.columns = [
    col.strip().lower().replace(" ", "_").replace("(", "").replace(")", "")
    for col in nutrition_df.columns
]

# cleaning food names with lowercase and removing whitespaces from beginning and end
nutrition_df['food_normalized'] = nutrition_df['food_normalized'].str.lower().str.strip()

####
# part 2: normalize ingredients
####
# normalizing function to remove extra words and symbols
def normalize_name(name):
    name = name.lower().strip()
    for char in [',', '.', '(', ')', '[', ']', '-']:
        name = name.replace(char, '')
    for word in ["fresh", "chopped", "sliced", "diced", "optional", "small", "large", "raw", "cooked"]:
        name = name.replace(word, '')
    if name.endswith('s') and len(name) > 3:
        name = name[:-1]
    name = name.strip()

    # synonym dictionary lookup
    if name in manual_synonyms:
        return manual_synonyms[name]
    return name

# creating synonyms dictionary
manual_synonyms = {
    "bean sprout": "mung bean",
    "bean sprouts": "mung bean",
    "mung bean sprout": "mung bean",
    "scallion": "green onion",
    "spring onion": "green onion",
    "cilantro": "coriander",
    "powdered sugar": "powdered sugar",
    "confectioners sugar": "powdered sugar",
}

# updating dataset 
nutrition_df['food_normalized'] = nutrition_df['food_normalized'].apply(normalize_name)

####
# part 3: WordNet synonyms, hyponyms, hypernyms
####
# getting synonyms from wordnet
def get_wordnet_synonyms(term):
    synonyms = set()
    for syn in wordnet.synsets(term, pos='n'):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_',' ').lower())
    return list(synonyms) if synonyms else [term]

# getting hyponyms (specific) from wordnet 
def get_hyponyms(term):
    hypos = set()
    for syn in wordnet.synsets(term, pos='n'):
        for hypo in syn.hyponyms():
            for lemma in hypo.lemmas():
                hypos.add(lemma.name().replace('_',' ').lower())
    return list(hypos)

# getting hypernym (general) from wordnet 
def get_wordnet_hypernyms(term):
    hypernyms = set()
    for syn in wordnet.synsets(term, pos='n'):
        for hyper in syn.hypernyms():
            for lemma in hyper.lemmas():
                hypernyms.add(lemma.name().replace("_", " ").lower())
    return list(hypernyms)

# expanding term to get relationship from wordnet 
def expand_ingredient_with_wordnet(term):
    term = term.lower().strip()
    synonyms = get_wordnet_synonyms(term)
    hyponyms = get_hyponyms(term)
    hypernyms = get_wordnet_hypernyms(term)

    expanded_set = set([term])
    expanded_set.update(synonyms)
    expanded_set.update(hyponyms)
    expanded_set.update(hypernyms)

    return list(expanded_set)

####
# part 4: ingredient families
####
ingredient_families = {
    "vegetables": ["zucchini", "tomato", "potato", "pumpkin", "carrot", "spinach", "lettuce"],
    "dairy": ["milk", "cheese", "butter", "yogurt", "cream"],
    "legumes": ["lentils", "chickpeas", "beans", "peas"],
    "nuts_seeds": ["peanut", "almond", "sesame", "walnut", "cashew"],
    "eggs": ["egg", "egg white", "egg yolk"],
    "oils": ["olive oil", "sesame oil", "vegetable oil"],
    "sweeteners": ["honey", "sugar", "maple syrup"],
}

####
# part 5: get nutrition for ingredient
####
def get_nutrition_for_ingredient(ingredient):
    ingredient_norm = normalize_name(ingredient)

    # check if ingredients match with family first
    for family, items in ingredient_families.items():
        if ingredient_norm == family or ingredient_norm in items:
            nutrition_list = []
            for item in items:
                if item in nutrition_df['food_normalized'].values:
                    match_row = nutrition_df[nutrition_df['food_normalized'] == item].iloc[0]
                    nutrition_list.append({
                        'ingredient': item,
                        'calories': round(match_row.get('calories_kcal_per_100g', 0), 2),
                        'protein': round(match_row.get('protein_g_per_100g', 0), 2),
                        'fat': round(match_row.get('fat_g_per_100g', 0), 2),
                        'carbohydrates': round(match_row.get('carbohydrates_g_per_100g', 0), 2),
                        'fiber': round(match_row.get('dietary_fiber_g_per_100g', 0), 2),
                        'sodium': round(match_row.get('sodium_mg_per_100g', 0), 2),
                        'vitamin_c': round(match_row.get('vitamin_c_mg_per_100g', 0), 2),
                        'vitamin_b11': round(match_row.get('vitamin_b11_mg_per_100g', 0), 2),
                        'iron': round(match_row.get('iron_mg_per_100g', 0), 2),
                    })
            if nutrition_list:
                # adding up all nutrition in family 
                total = {k: sum(d[k] for d in nutrition_list) for k in nutrition_list[0] if k != 'ingredient'}
                return {'ingredient': ingredient, 'match': f"{ingredient_norm} family", **total}

    # if not family match, expand using WordNet
    expanded_terms = expand_ingredient_with_wordnet(ingredient_norm)

    best_match = None
    best_score = 0
    for term in expanded_terms:
        match = process.extractOne(term, nutrition_df['food_normalized'], scorer=fuzz.token_set_ratio)
        if match and match[1] > best_score:
            best_match = match
            best_score = match[1]
    
    # yes to match, getting nutrition info 
    if best_match and best_score >= 80:
        match_row = nutrition_df[nutrition_df['food_normalized'] == best_match[0]].iloc[0]
        return {
            'ingredient': ingredient,
            'match': best_match[0],
            'calories': round(match_row.get('calories_kcal_per_100g', 0), 2),
            'protein': round(match_row.get('protein_g_per_100g', 0), 2),
            'fat': round(match_row.get('fat_g_per_100g', 0), 2),
            'carbohydrates': round(match_row.get('carbohydrates_g_per_100g', 0), 2),
            'fiber': round(match_row.get('dietary_fiber_g_per_100g', 0), 2),
            'sodium': round(match_row.get('sodium_mg_per_100g', 0), 2),
            'vitamin_c': round(match_row.get('vitamin_c_mg_per_100g', 0), 2),
            'vitamin_b11': round(match_row.get('vitamin_b11_mg_per_100g', 0), 2),
            'iron': round(match_row.get('iron_mg_per_100g', 0), 2),
        }

    # if no match, return none
    return {'ingredient': ingredient, 'match': None}

####
# part 6: parse ingredients in recipes
####
# converting recipes ingredients string to lists 
def parse_ingredients(ingredients_str):
    # skipping empty ingredients
    if pd.isna(ingredients_str):
        return []
    # removing brackets, quotes, commas, bascially normalize the names
    clean = ingredients_str.strip("[]").replace("'", "").replace('"', "")
    return [normalize_name(i) for i in clean.split(",") if i.strip()]

master_df["ingredients_list"] = master_df["ingredients"].apply(parse_ingredients)

####
# part 7: full matching pipeline using WordNet expansion
####
def full_match_pipeline(user_ingredients, recipe_ingredients):
    # expanding user ingredients with hyponyms and synonyms 
    all_user_terms = set()
    for u_ing in user_ingredients:
        normalized = normalize_name(u_ing)
        expanded_terms = expand_ingredient_with_wordnet(normalized)
        all_user_terms.update(expanded_terms)
   
    # only include recipe ingredients that literally match any expanded term
    matched_recipe_ingredients = [r for r in recipe_ingredients if normalize_name(r) in all_user_terms]
    return matched_recipe_ingredients

# score recipes by how many ingredients match
def score_recipe(recipe_ingredients, user_ingredients):
    matched_ings = full_match_pipeline(user_ingredients, recipe_ingredients)
    score = len(matched_ings) / len(recipe_ingredients) if recipe_ingredients else 0
    return score, matched_ings

####
# part 8: recommend recipes
####
def recommend_recipes(user_input, top_n=5):
    # converting user input into list of ingredients
    user_ingredients = [normalize_name(i) for i in user_input.split(",")]

    # scoring each recipe by comparing recipe's ingredients wtih user's ingredients 
    def score_wrapper(recipe_ingredients):
        # score_recipe() returns the score & list of ingredients that matched
        score, matched_ings = score_recipe(recipe_ingredients, user_ingredients)
        return pd.Series([score, matched_ings])

    # updating to the master recipe dataset with match score and matched ingredients 
    master_df[['match_score', 'matched_ingredients']] = master_df["ingredients_list"].apply(score_wrapper)

    # keeping only recipes that have match score greater than 0
    recommended = master_df[master_df["match_score"] > 0].sort_values(by="match_score", ascending=False)
    
    # in case there is no match, give user update 
    if recommended.empty:
        print("No matching recipes found. Try fewer or simpler ingredients.")
        return pd.DataFrame()

    # display certain colums to user
    cols_to_show = ["title", "ingredients_list", "time", "instructions", "match_score", "matched_ingredients"]
    return recommended.head(top_n)[cols_to_show]

####
# part 9: user input & output
####
# asking for user ingredient input
user_input = input("Enter your ingredients (comma-separated): ")
# converting user input text to a list & removing spaces 
user_ingredients = [i.strip() for i in user_input.split(",") if i.strip()]

# nutrition info for each of user ingredient input
nutrition_results = [get_nutrition_for_ingredient(ing) for ing in user_ingredients]
print("\nNUTRITION PER INGREDIENT")
print("-"*50)

# if user's ingredient found in nutrition database, give nutrition details
for info in nutrition_results:
    if info['match']:
        print(f"{info['ingredient'].title()} (Matched: {info['match']})")
        print(f"  Calories (kcal per 100g): {info['calories']} kcal")
        print(f"  Protein (g per 100g): {info['protein']} g, Fat (g per 100g): {info['fat']} g, Carbs (g per 100g): {info['carbohydrates']} g, Fiber: {info['fiber']} g, Sodium: {info['sodium']} mg")
        print(f"  Vitamin C (mg per 100g): {info['vitamin_c']} mg, Vitamin B11 (mg per 100g): {info['vitamin_b11']} mg, Iron (mg per 100g): {info['iron']} mg")
    else:
        print(f"{info['ingredient'].title()}: No match found")
    print("-"*50)

# adding up total nutrition for all of user's ingredient input 
total_nutrition = {key: sum(info.get(key,0) for info in nutrition_results if info['match']) 
                   for key in ['calories','protein','fat','carbohydrates','fiber','sodium','vitamin_c','vitamin_b11','iron']}

# displaying nutrition results 
print("\nTOTAL NUTRITION (per 100g each ingredient)")
for k, v in total_nutrition.items():
    print(f"{k.title()}: {round(v,2)}")

# getting top 5 recipe recommendations
top_recipes = recommend_recipes(user_input, top_n=5)

# found recipes with detailed informations 
if not top_recipes.empty:
    print("\nTOP 5 RECIPE RECOMMENDATIONS")
    print("-"*50)
    for idx, row in top_recipes.iterrows():
        matched_count = len(row['matched_ingredients'])
        total_count = len(row['ingredients_list'])
        print(f"Recipe: {row['title']}")
        print(f"Ingredients: {', '.join(row['ingredients_list'])}")
        print(f"Time (minutes): {row['time']}")
        print(f"Instructions: {row['instructions']}")
        print(f"Matched Ingredients: {matched_count}/{total_count} ({', '.join(row['matched_ingredients'])})")
        print(f"Match Score: {row['match_score']:.2f}")
        print("-"*50)
else:
    print("No recipe matches found.")
