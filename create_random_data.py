import json
import numpy as np
import random

with open("./dataset/AQUA/train/dataset.json") as fp:
    aqua = json.load(fp)

    
with open("./dataset/HRVQA-1.0/train/dataset.json") as fp:
    hrvqa = json.load(fp)
    
# with open("./dataset/VQA-MED/train/dataset.json") as fp:
#     vqamed = json.load(fp)

print(len(aqua))
print(len(hrvqa))
# print(len(vqamed))
# breakpoint()

size = 60000
indices = np.arange(len(aqua)).tolist()
selection = sorted(random.sample(indices, size))
new_list = []
for idx in selection:
    new_list.append(aqua[idx])
with open("./scenarios/AQUA-0.json", 'w') as json_file:
    json.dump(new_list[:20000], json_file, indent=4)

with open("./scenarios/AQUA-1.json", 'w') as json_file:
    json.dump(new_list[20000:40000], json_file, indent=4)
    
with open("./scenarios/AQUA-2.json", 'w') as json_file:
    json.dump(new_list[40000:60000], json_file, indent=4)


size = 120000
indices = np.arange(len(hrvqa)).tolist()
selection = sorted(random.sample(indices, size))
new_list = []
for idx in selection:
    new_list.append(hrvqa[idx])

with open("./scenarios/HRVQA-1.0-0.json", 'w') as json_file:
    json.dump(new_list[:20000], json_file, indent=4)

with open("./scenarios/HRVQA-1.0-1.json", 'w') as json_file:
    json.dump(new_list[40000:60000], json_file, indent=4)
    
with open("./scenarios/HRVQA-1.0-2.json", 'w') as json_file:
    json.dump(new_list[60000:80000], json_file, indent=4)

with open("./scenarios/HRVQA-1.0-3.json", 'w') as json_file:
    json.dump(new_list[80000:100000], json_file, indent=4)

with open("./scenarios/HRVQA-1.0-4.json", 'w') as json_file:
    json.dump(new_list[100000:120000], json_file, indent=4)
    
# size = 12000
# indices = np.arange(len(vqamed)).tolist()
# selection = sorted(random.sample(indices, size))
# new_list = []
# for idx in selection:
#     new_list.append(vqamed[idx])
    
# with open("./scenarios/VQA-MED-0.json", 'w') as json_file:
#     json.dump(new_list[:4000], json_file, indent=4)

# with open("./scenarios/VQA-MED-1.json", 'w') as json_file:
#     json.dump(new_list[4000:8000], json_file, indent=4)
    
# with open("./scenarios/VQA-MED-2.json", 'w') as json_file:
#     json.dump(new_list[8000:12000], json_file, indent=4)



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