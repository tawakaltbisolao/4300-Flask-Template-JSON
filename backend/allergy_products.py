import json

with open('dataset/allergies.json') as f:
    allergies = json.load(f)
sulfates = allergies['sulfates']
parabens = allergies['parabens']
fragrances = allergies['fragrances']
seeds = allergies['seeds']
nuts = allergies['nuts']
alcohols = allergies['alcohols']

sulfates = set(sulfates)
parabens = set(parabens)
fragrances = set(fragrances)
seeds = set(seeds)
nuts = set(nuts)
alcohols = set(alcohols)

allergies = dict()
allergies['sulfates'] = set()
allergies['parabens'] = set()
allergies['fragrances'] = set()
allergies['seeds'] = set()
allergies['nuts'] = set()
allergies['alcohols'] = set()

with open('dataset/products_ulta.json') as f:
    products_info = json.load(f)
products = products_info['shampoo']
products.update(products_info['conditioner'])
products.update(products_info['oil'])

with open('dataset/ulta_product_ids.json') as f:
    prod_ids = json.load(f)

id_to_index = {v:i for i,v in enumerate(prod_ids)}

for id in prod_ids:
    ingredients = products[id]['ingredients']
    for ing in ingredients:
        if(ing in sulfates):
            allergies['sulfates'].add(id_to_index[id])
        if(ing in parabens):
            allergies['parabens'].add(id_to_index[id])
        if(ing in fragrances):
            allergies['fragrances'].add(id_to_index[id])
        if(ing in seeds):
            allergies['seeds'].add(id_to_index[id])
        if(ing in nuts):
            allergies['nuts'].add(id_to_index[id])
        if(ing in alcohols):
            allergies['alcohols'].add(id_to_index[id])

allergies['sulfates'] = sorted(list(allergies['sulfates']))
allergies['parabens'] = sorted(list(allergies['parabens']))
allergies['fragrances'] = sorted(list(allergies['fragrances']))
allergies['seeds'] = sorted(list(allergies['seeds']))
allergies['nuts'] = sorted(list(allergies['nuts']))
allergies['alcohols'] = sorted(list(allergies['alcohols']))
with open('dataset/allergy_map.json', 'w') as f:
    json.dump(allergies, f)

