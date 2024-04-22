import json

with open("./dataset/AQUA/train/dataset.json") as fp:
    aqua = json.load(fp)
    
with open("./dataset/HRVQA-1.0/train/dataset.json") as fp:
    hrvqa = json.load(fp)
    
with open("./dataset/VQA-MED/train/dataset.json") as fp:
    vqamed = json.load(fp)

print(len(aqua))
print(len(hrvqa))
print(len(vqamed))
# breakpoint()

with open("./scenarios/AQUA-0.json", 'w') as json_file:
    json.dump(aqua[:4000], json_file, indent=4)

with open("./scenarios/AQUA-1.json", 'w') as json_file:
    json.dump(aqua[10000:14000], json_file, indent=4)
    
with open("./scenarios/AQUA-2.json", 'w') as json_file:
    json.dump(aqua[20000:24000], json_file, indent=4)

with open("./scenarios/HRVQA-1.0-0.json", 'w') as json_file:
    json.dump(hrvqa[:4000], json_file, indent=4)

with open("./scenarios/HRVQA-1.0-1.json", 'w') as json_file:
    json.dump(hrvqa[10000:14000], json_file, indent=4)
    
with open("./scenarios/HRVQA-1.0-2.json", 'w') as json_file:
    json.dump(hrvqa[20000:24000], json_file, indent=4)

with open("./scenarios/HRVQA-1.0-3.json", 'w') as json_file:
    json.dump(hrvqa[30000:34000], json_file, indent=4)
    
with open("./scenarios/VQA-MED-0.json", 'w') as json_file:
    json.dump(vqamed[:4000], json_file, indent=4)

with open("./scenarios/VQA-MED-1.json", 'w') as json_file:
    json.dump(vqamed[4000:8000], json_file, indent=4)
    
with open("./scenarios/VQA-MED-2.json", 'w') as json_file:
    json.dump(vqamed[8000:12000], json_file, indent=4)



# with open("./scenarios/AQUA-0.json", 'w') as json_file:
#     json.dump(aqua[:10000], json_file, indent=4)

# with open("./scenarios/AQUA-1.json", 'w') as json_file:
#     json.dump(aqua[10000:20000], json_file, indent=4)
    
# with open("./scenarios/AQUA-2.json", 'w') as json_file:
#     json.dump(aqua[20000:30000], json_file, indent=4)
    
# with open("./scenarios/AQUA-3.json", 'w') as json_file:
#     json.dump(aqua[30000:40000], json_file, indent=4)
    
# with open("./scenarios/AQUA-4.json", 'w') as json_file:
#     json.dump(aqua[40000:50000], json_file, indent=4)
    

# with open("./scenarios/HRVQA-1.0-0.json", 'w') as json_file:
#     json.dump(hrvqa[:10000], json_file, indent=4)

# with open("./scenarios/HRVQA-1.0-1.json", 'w') as json_file:
#     json.dump(hrvqa[10000:20000], json_file, indent=4)
    
# with open("./scenarios/HRVQA-1.0-2.json", 'w') as json_file:
#     json.dump(hrvqa[20000:30000], json_file, indent=4)
    
# with open("./scenarios/HRVQA-1.0-3.json", 'w') as json_file:
#     json.dump(hrvqa[30000:40000], json_file, indent=4)
    
# with open("./scenarios/HRVQA-1.0-4.json", 'w') as json_file:
#     json.dump(hrvqa[40000:50000], json_file, indent=4)