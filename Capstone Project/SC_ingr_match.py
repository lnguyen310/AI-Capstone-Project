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

# loading datasetsx
master_df = pd.read_csv("master_recipes.csv")
nutrition_df = pd.read_csv("cleaned_nutrition_dataset_per100g.csv")

####
# part 1: cleaning columns
####
# making columns lowercase, removing spaces and brackets 
nutrition_df.columns = [
    col.strip().lower().replace(" ", "_").replace("(", "").replace(")", "")
    for col in nutrition_df.columns
]
nutrition_df['food_normalized'] = nutrition_df['food_normalized'].str.lower().str.strip()

####
# part 2: normalize ingredients
####
# normalizing function to remove extra words and symbols
# creating synonym dictionary
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

def normalize_name(name):
    # lowercase and removing spaces
    name = name.lower().strip()
    # removing punctutions 
    for char in [',', '.', '(', ')', '[', ']', '-']:
        name = name.replace(char, '')
    # removing extra descriptive words 
    for word in ["fresh", "chopped", "sliced", "diced", "optional", "small", "large", "raw", "cooked"]:
        name = name.replace(word, '')
    # removing plural
    if name.endswith('s') and len(name) > 3:
        name = name[:-1]
    # applying manual synonym dictionary 
    name = name.strip()
    if name in manual_synonyms:
        return manual_synonyms[name]
    return name

# updating dataset 
nutrition_df['food_normalized'] = nutrition_df['food_normalized'].apply(normalize_name)

####
# part 3: WordNet synonyms, hyponyms, hypernyms
####
# expanding ingredient into multiuiple related words
def expand_ingredient_with_relationships(term):

    term = term.lower().strip()
    relationship_map = {term: "original"} 

    # synonyms (same meaning)
    synonyms_set = set()
    for syn in wordnet.synsets(term, pos='n'):
        for lemma in syn.lemmas():
            s = lemma.name().replace("_", " ").lower()
            if s != term:
                synonyms_set.add(s)
    for s in synonyms_set:
        if s not in relationship_map:
            relationship_map[s] = "synonym"

    # hyponyms (more specific)
    hyponyms_set = set()
    for syn in wordnet.synsets(term, pos='n'):
        for hypo in syn.hyponyms():
            for lemma in hypo.lemmas():
                h = lemma.name().replace("_", " ").lower()
                if h != term:
                    hyponyms_set.add(h)
    for h in hyponyms_set:
        if h not in relationship_map:
            relationship_map[h] = "hyponym"

    # hypernyms (general category)
    hypernyms_set = set()
    for syn in wordnet.synsets(term, pos='n'):
        for hyper in syn.hypernyms():
            for lemma in hyper.lemmas():
                h2 = lemma.name().replace("_", " ").lower()
                if h2 != term:
                    if h2 not in relationship_map:
                        relationship_map[h2] = "hypernym"

    return relationship_map

####
# part 4: ingredient families
####
# created ingredient family dictionary for common categories
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
# part 5: nutrition info
####
# getting nutrition info for ingredient
def get_nutrition_for_ingredient(ingredient):
    ingredient_norm = normalize_name(ingredient)

    # ingredient family check
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
            
            # adding together nutrition info 
            if nutrition_list:
                total = {k: sum(d[k] for d in nutrition_list) for k in nutrition_list[0] if k != 'ingredient'}
                return {'ingredient': ingredient, 'match': f"{ingredient_norm} family", **total}

    # WordNet expansion matching
    expanded_terms = expand_ingredient_with_relationships(ingredient_norm)

    best_match = None
    best_score = 0
    # fuzzy match expanding terms to dataset 
    for term in expanded_terms:
        match = process.extractOne(term, nutrition_df['food_normalized'], scorer=fuzz.token_set_ratio)
        if match and match[1] > best_score:
            best_match = match
            best_score = match[1]

    # returning nutrition from best match 
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

    return {'ingredient': ingredient, 'match': None}

####
# part 6: parse ingredients
####
# converting recipe ingredients from string to list 
def parse_ingredients(ingredients_str):
    if pd.isna(ingredients_str):
        return []
    clean = ingredients_str.strip("[]").replace("'", "").replace('"', "")
    return [normalize_name(i) for i in clean.split(",") if i.strip()]

# updating dataset 
master_df["ingredients_list"] = master_df["ingredients"].apply(parse_ingredients)

####
# part 7: matching with relationship mapping
####
# matching ingredients to their semantic relationship 
def full_match_pipeline_with_relationship(user_ingredients, recipe_ingredients):
    expanded_dict = {}
    # priority mapping
    priority = {"original": 4, "synonym": 3, "hyponym": 2, "hypernym": 1}

    for u_ing in user_ingredients:
        normalized = normalize_name(u_ing)
        expanded = expand_ingredient_with_relationships(normalized)
        # storing best match for each expanded term
        for term, relation in expanded.items():
            if term not in expanded_dict:
                expanded_dict[term] = (u_ing, relation)
            else:
                _, existing_rel = expanded_dict[term]
                # choosing highest priority relationship
                if priority.get(relation, 0) > priority.get(existing_rel, 0):
                    expanded_dict[term] = (u_ing, relation)

    matched_recipe_ings = []
    matched_relationships = {}

    # matching recipe ingredients to expanded user's input
    for r in recipe_ingredients:
        r_norm = normalize_name(r)
        if r_norm in expanded_dict:
            matched_recipe_ings.append(r)
            matched_relationships[r] = expanded_dict[r_norm]

    return matched_recipe_ings, matched_relationships

# getting user's input score to recipe ingreidents 
def score_recipe(recipe_ingredients, user_ingredients):
    matched_ings, matched_details = full_match_pipeline_with_relationship(user_ingredients, recipe_ingredients)
    score = len(matched_ings) / len(recipe_ingredients) if recipe_ingredients else 0
    return score, matched_ings, matched_details

####
# part 8: recommend recipes
####
# ranking recipes based on number of user matches 
# only getting teh top 5 recipes with match info and relationship details
def recommend_recipes(user_input, top_n=5):
    user_ingredients = [normalize_name(i) for i in user_input.split(",")]

    def score_wrapper(recipe_ingredients):
        score, matched_ings, matched_details = score_recipe(recipe_ingredients, user_ingredients)
        return pd.Series([score, matched_ings, matched_details])

    master_df[['match_score', 'matched_ingredients', 'matched_relationships']] = \
        master_df["ingredients_list"].apply(score_wrapper)

    recommended = master_df[master_df["match_score"] > 0].sort_values(by="match_score", ascending=False)

    if recommended.empty:
        print("No matching recipes found. Try fewer or simpler ingredients.")
        return pd.DataFrame()

    cols_to_show = ["title", "ingredients_list", "time", "instructions",
                    "match_score", "matched_ingredients", "matched_relationships"]

    return recommended.head(top_n)[cols_to_show]

####
# part 9: user interaction
####
# asking for user's input

user_input = input("Enter your ingredients (comma-separated): ")
user_ingredients = [i.strip() for i in user_input.split(",") if i.strip()]

####
# part 10 : printing results 
####
# printing nutrition results 
nutrition_results = [get_nutrition_for_ingredient(ing) for ing in user_ingredients]
print("\nNUTRITION PER INGREDIENT")
print("-" * 50)

for info in nutrition_results:
    if info['match']:
        print(f"{info['ingredient'].title()} (Matched: {info['match']})")
        print(f"  Calories: {info['calories']} kcal")
        print(f"  Protein: {info['protein']} g | Fat: {info['fat']} g | Carbs: {info['carbohydrates']} g | Fiber: {info['fiber']} g")
        print(f"  Sodium: {info['sodium']} mg | Vitamin C: {info['vitamin_c']} mg | B11: {info['vitamin_b11']} mg | Iron: {info['iron']} mg")
    else:
        print(f"{info['ingredient'].title()}: No match found")
    print("-"*50)

total_nutrition = {key: sum(info.get(key,0) for info in nutrition_results if info['match'])
                   for key in ['calories','protein','fat','carbohydrates','fiber','sodium','vitamin_c','vitamin_b11','iron']}

# printing total nutrition, all of user's input 
print("\nTOTAL NUTRITION (per 100g each ingredient)")
for k, v in total_nutrition.items():
    print(f"{k.title()}: {round(v,2)}")

# getting top 5 recipe recommendations 
top_recipes = recommend_recipes(user_input, top_n=5)

# printing top 5 recipe recommendations & query expansions 
if not top_recipes.empty:
    print("\nTOP 5 RECIPE RECOMMENDATIONS")
    print("-"*50)
    for idx, row in top_recipes.iterrows():
        matched_count = len(row['matched_ingredients'])
        total_count = len(row['ingredients_list'])
        print(f"** Recipe: {row['title']} **")
        print(f"- Ingredients: {', '.join(row['ingredients_list'])}")
        print(f"- Time: {row['time']} minutes")
        print(f"- Instructions: {row['instructions']}")

        # showing synonyms, hyponyms, hypernyms relationships
        if row['matched_relationships']:
            print("\n - Matched Ingredients with Relationships (User Input → Recipe Ingredient [Relationship]):")
            for recipe_ing, (user_ing, relation) in row['matched_relationships'].items():
                print(f"  {user_ing} → {recipe_ing} ({relation})")
        else:
            print("\nNo synonym/hyponym/hypernym matches found.")

        print(f"- Matched Ingredients: {matched_count}/{total_count} ({', '.join(row['matched_ingredients'])})")
        print(f"- Match Score: {row['match_score']:.2f}")
        print("-"*50)
else:
    print("No recipe matches found.")
