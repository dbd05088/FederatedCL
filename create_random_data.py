import json

with open("./dataset/AQUA/train/dataset.json") as fp:
    aqua = json.load(fp)
    
with open("./dataset/HRVQA-1.0/train/dataset.json") as fp:
    hrvqa = json.load(fp)

print(len(aqua))
print(len(hrvqa))

with open("./scenarios/AQUA-0.json", 'w') as json_file:
    json.dump(aqua[:10000], json_file, indent=4)

with open("./scenarios/AQUA-1.json", 'w') as json_file:
    json.dump(aqua[10000:20000], json_file, indent=4)
    
with open("./scenarios/AQUA-2.json", 'w') as json_file:
    json.dump(aqua[20000:30000], json_file, indent=4)
    
with open("./scenarios/AQUA-3.json", 'w') as json_file:
    json.dump(aqua[30000:40000], json_file, indent=4)
    
with open("./scenarios/AQUA-4.json", 'w') as json_file:
    json.dump(aqua[40000:50000], json_file, indent=4)
    

with open("./scenarios/HRVQA-1.0-0.json", 'w') as json_file:
    json.dump(hrvqa[:10000], json_file, indent=4)

with open("./scenarios/HRVQA-1.0-1.json", 'w') as json_file:
    json.dump(hrvqa[10000:20000], json_file, indent=4)
    
with open("./scenarios/HRVQA-1.0-2.json", 'w') as json_file:
    json.dump(hrvqa[20000:30000], json_file, indent=4)
    
with open("./scenarios/HRVQA-1.0-3.json", 'w') as json_file:
    json.dump(hrvqa[30000:40000], json_file, indent=4)
    
with open("./scenarios/HRVQA-1.0-4.json", 'w') as json_file:
    json.dump(hrvqa[40000:50000], json_file, indent=4)