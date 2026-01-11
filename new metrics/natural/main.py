"""****Generalizability of Our Approach to Real-World Problems****

We aim to demonstrate that our evaluation approach for XAI methods informs us how well XAI methods will perform when solving real-world problems that involve natural data sets and trained models. To this end, we compared the XAI method evaluation results obtained on our synthetic dataset using our manually constructed ResNet-50 model with those obtained on a natural dataset using a trained ResNet-50 model. We show that the metric scores of XAI methods when they are evaluated on our synthetic dataset and manually constructed model are close to the metric scores of these XAI methods when they are evaluated on a natural dataset and trained model.

**How to Replicate results**

1.   If you will run the code on Google Colab, then create a folder on Google Drive entitled "Colab_daan_coco". Or, if you do not want to run the code on Google Colab, a local folder with the name "colab_daan_coco" can be mounted instead in the code block below. Copy the provided files in this folder. Download and copy the COCO 2017 dataset (https://cocodataset.org) inside the folder called "DatasetCoco". Also, download and copy COCO-Stuff dataset (https://github.com/nightrome/cocostuff) inside the folder called "DatasetCoco".

2. Pre-compute which concepts are present in which image. This is stored in the file DatasetCoco/concepts_in_images_val2017.npy. To compute this, set
```
settings_global['calc_concepts_in_images'] = True
```

2.   Pre-compute which input examples contain the selected concepts. This is stored in the file DatasetCoco/examples_items_RESNET50.npy. To compute this, set
```
settings_global['determine_concept_exanmples'] = True
```
Run the code until the code block entitled "Compute explanations using XAI methods on the data computed above, and compute XAI metrics of these explanations". Then, set
```
settings_global['calc_concepts_in_images'] = False
settings_global['determine_concept_exanmples'] = False
```
3.   Run the code after changing the following line in the code block below
```
settings_global['run_id'] = 0
```

4.   Run the code after changing the following line in the code block below
```
settings_global['run_id'] = 1
```

5.   Run the code after changing the following line in the code block below
```
settings_global['run_id'] = 2
```

6.   The results are presented by the code blocks entitled "Results on Natural dataset" and "Final Results".
"""

### Global Settings ###

settings_global = {}
settings_global['run_id'] = 2                            # ID of the current run. Used to store results in a separate folder with this ID.  See result replication instructions above. Mean+/-std results over all runs are computed and displayed below.
settings_global['num_runs'] = 3                          # Total number of runs.
settings_global['determine_concept_exanmples'] = False   # Determine concept exanmples, or else load earlier computed to save time. See result replication instructions above.
settings_global['calc_metrics'] = True                   # Whether or not to calculate XAI metrics on generated explanations.
settings_global['saveSelectedImages'] = True             # Whether or not to save 8 example input images, explanations, GT explanations as images.
settings_global['calc_concepts_in_images'] = False       # Determine which concepts are in each image, or load a file that stores this information to save time. See result replication instructions above.



import sys

### Install Dependencies ###

# Labels of the COCO 2017 dataset. Taken from: https://github.com/nightrome/cocostuff/blob/master/labels.txt
coco = {0: 'unlabeled',1: 'person',2: 'bicycle',3: 'car',4: 'motorcycle',5: 'airplane',6: 'bus',7: 'train',8: 'truck',9: 'boat',10: 'traffic light',11: 'fire hydrant',12: 'street sign',13: 'stop sign',14: 'parking meter',15: 'bench',16: 'bird',17: 'cat',18: 'dog',19: 'horse',20: 'sheep',21: 'cow',22: 'elephant',23: 'bear',24: 'zebra',25: 'giraffe',26: 'hat',27: 'backpack',28: 'umbrella',29: 'shoe',30: 'eye glasses',31: 'handbag',32: 'tie',33: 'suitcase',34: 'frisbee',35: 'skis',36: 'snowboard',37: 'sports ball',38: 'kite',39: 'baseball bat',40: 'baseball glove',41: 'skateboard',42: 'surfboard',43: 'tennis racket',44: 'bottle',45: 'plate',46: 'wine glass',47: 'cup',48: 'fork',49: 'knife',50: 'spoon',51: 'bowl',52: 'banana',53: 'apple',54: 'sandwich',55: 'orange',56: 'broccoli',57: 'carrot',58: 'hot dog',59: 'pizza',60: 'donut',61: 'cake',62: 'chair',63: 'couch',64: 'potted plant',65: 'bed',66: 'mirror',67: 'dining table',68: 'window',69: 'desk',70: 'toilet',71: 'door',72: 'tv',73: 'laptop',74: 'mouse',75: 'remote',76: 'keyboard',77: 'cell phone',78: 'microwave',79: 'oven',80: 'toaster',81: 'sink',82: 'refrigerator',83: 'blender',84: 'book',85: 'clock',86: 'vase',87: 'scissors',88: 'teddy bear',89: 'hair drier',90: 'toothbrush',91: 'hair brush',92: 'banner',93: 'blanket',94: 'branch',95: 'bridge',96: 'building-other',97: 'bush',98: 'cabinet',99: 'cage',100: 'cardboard',101: 'carpet',102: 'ceiling-other',103: 'ceiling-tile',104: 'cloth',105: 'clothes',106: 'clouds',107: 'counter',108: 'cupboard',109: 'curtain',110: 'desk-stuff',111: 'dirt',112: 'door-stuff',113: 'fence',114: 'floor-marble',115: 'floor-other',116: 'floor-stone',117: 'floor-tile',118: 'floor-wood',119: 'flower',120: 'fog',121: 'food-other',122: 'fruit',123: 'furniture-other',124: 'grass',125: 'gravel',126: 'ground-other',127: 'hill',128: 'house',129: 'leaves',130: 'light',131: 'mat',132: 'metal',133: 'mirror-stuff',134: 'moss',135: 'mountain',136: 'mud',137: 'napkin',138: 'net',139: 'paper',140: 'pavement',141: 'pillow',142: 'plant-other',143: 'plastic',144: 'platform',145: 'playingfield',146: 'railing',147: 'railroad',148: 'river',149: 'road',150: 'rock',151: 'roof',152: 'rug',153: 'salad',154: 'sand',155: 'sea',156: 'shelf',157: 'sky-other',158: 'skyscraper',159: 'snow',160: 'solid-other',161: 'stairs',162: 'stone',163: 'straw',164: 'structural-other',165: 'table',166: 'tent',167: 'textile-other',168: 'towel',169: 'tree',170: 'vegetable',171: 'wall-brick',172: 'wall-concrete',173: 'wall-other',174: 'wall-panel',175: 'wall-stone',176: 'wall-tile',177: 'wall-wood',178: 'water-other',179: 'waterdrops',180: 'window-blind',181: 'window-other',182: 'wood'}

# Labels of the Imagenet dataset. Taken from: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
imagenet = {0: 'tench, Tinca tinca', 1: 'goldfish, Carassius auratus', 2: 'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias', 3: 'tiger shark, Galeocerdo cuvieri', 4: 'hammerhead, hammerhead shark', 5: 'electric ray, crampfish, numbfish, torpedo', 6: 'stingray', 7: 'cock', 8: 'hen', 9: 'ostrich, Struthio camelus', 10: 'brambling, Fringilla montifringilla', 11: 'goldfinch, Carduelis carduelis', 12: 'house finch, linnet, Carpodacus mexicanus', 13: 'junco, snowbird', 14: 'indigo bunting, indigo finch, indigo bird, Passerina cyanea', 15: 'robin, American robin, Turdus migratorius', 16: 'bulbul', 17: 'jay', 18: 'magpie', 19: 'chickadee', 20: 'water ouzel, dipper', 21: 'kite', 22: 'bald eagle, American eagle, Haliaeetus leucocephalus', 23: 'vulture', 24: 'great grey owl, great gray owl, Strix nebulosa', 25: 'European fire salamander, Salamandra salamandra', 26: 'common newt, Triturus vulgaris', 27: 'eft', 28: 'spotted salamander, Ambystoma maculatum', 29: 'axolotl, mud puppy, Ambystoma mexicanum', 30: 'bullfrog, Rana catesbeiana', 31: 'tree frog, tree-frog', 32: 'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui', 33: 'loggerhead, loggerhead turtle, Caretta caretta', 34: 'leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea', 35: 'mud turtle', 36: 'terrapin', 37: 'box turtle, box tortoise', 38: 'banded gecko', 39: 'common iguana, iguana, Iguana iguana', 40: 'American chameleon, anole, Anolis carolinensis', 41: 'whiptail, whiptail lizard', 42: 'agama', 43: 'frilled lizard, Chlamydosaurus kingi', 44: 'alligator lizard', 45: 'Gila monster, Heloderma suspectum', 46: 'green lizard, Lacerta viridis', 47: 'African chameleon, Chamaeleo chamaeleon', 48: 'Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis', 49: 'African crocodile, Nile crocodile, Crocodylus niloticus', 50: 'American alligator, Alligator mississipiensis', 51: 'triceratops', 52: 'thunder snake, worm snake, Carphophis amoenus', 53: 'ringneck snake, ring-necked snake, ring snake', 54: 'hognose snake, puff adder, sand viper', 55: 'green snake, grass snake', 56: 'king snake, kingsnake', 57: 'garter snake, grass snake', 58: 'water snake', 59: 'vine snake', 60: 'night snake, Hypsiglena torquata', 61: 'boa constrictor, Constrictor constrictor', 62: 'rock python, rock snake, Python sebae', 63: 'Indian cobra, Naja naja', 64: 'green mamba', 65: 'sea snake', 66: 'horned viper, cerastes, sand viper, horned asp, Cerastes cornutus', 67: 'diamondback, diamondback rattlesnake, Crotalus adamanteus', 68: 'sidewinder, horned rattlesnake, Crotalus cerastes', 69: 'trilobite', 70: 'harvestman, daddy longlegs, Phalangium opilio', 71: 'scorpion', 72: 'black and gold garden spider, Argiope aurantia', 73: 'barn spider, Araneus cavaticus', 74: 'garden spider, Aranea diademata', 75: 'black widow, Latrodectus mactans', 76: 'tarantula', 77: 'wolf spider, hunting spider', 78: 'tick', 79: 'centipede', 80: 'black grouse', 81: 'ptarmigan', 82: 'ruffed grouse, partridge, Bonasa umbellus', 83: 'prairie chicken, prairie grouse, prairie fowl', 84: 'peacock', 85: 'quail', 86: 'partridge', 87: 'African grey, African gray, Psittacus erithacus', 88: 'macaw', 89: 'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita', 90: 'lorikeet', 91: 'coucal', 92: 'bee eater', 93: 'hornbill', 94: 'hummingbird', 95: 'jacamar', 96: 'toucan', 97: 'drake', 98: 'red-breasted merganser, Mergus serrator', 99: 'goose', 100: 'black swan, Cygnus atratus', 101: 'tusker', 102: 'echidna, spiny anteater, anteater', 103: 'platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus', 104: 'wallaby, brush kangaroo', 105: 'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus', 106: 'wombat', 107: 'jellyfish', 108: 'sea anemone, anemone', 109: 'brain coral', 110: 'flatworm, platyhelminth', 111: 'nematode, nematode worm, roundworm', 112: 'conch', 113: 'snail', 114: 'slug', 115: 'sea slug, nudibranch', 116: 'chiton, coat-of-mail shell, sea cradle, polyplacophore', 117: 'chambered nautilus, pearly nautilus, nautilus', 118: 'Dungeness crab, Cancer magister', 119: 'rock crab, Cancer irroratus', 120: 'fiddler crab', 121: 'king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica', 122: 'American lobster, Northern lobster, Maine lobster, Homarus americanus', 123: 'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish', 124: 'crayfish, crawfish, crawdad, crawdaddy', 125: 'hermit crab', 126: 'isopod', 127: 'white stork, Ciconia ciconia', 128: 'black stork, Ciconia nigra', 129: 'spoonbill', 130: 'flamingo', 131: 'little blue heron, Egretta caerulea', 132: 'American egret, great white heron, Egretta albus', 133: 'bittern', 134: 'crane', 135: 'limpkin, Aramus pictus', 136: 'European gallinule, Porphyrio porphyrio', 137: 'American coot, marsh hen, mud hen, water hen, Fulica americana', 138: 'bustard', 139: 'ruddy turnstone, Arenaria interpres', 140: 'red-backed sandpiper, dunlin, Erolia alpina', 141: 'redshank, Tringa totanus', 142: 'dowitcher', 143: 'oystercatcher, oyster catcher', 144: 'pelican', 145: 'king penguin, Aptenodytes patagonica', 146: 'albatross, mollymawk', 147: 'grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus', 148: 'killer whale, killer, orca, grampus, sea wolf, Orcinus orca', 149: 'dugong, Dugong dugon', 150: 'sea lion', 151: 'Chihuahua', 152: 'Japanese spaniel', 153: 'Maltese dog, Maltese terrier, Maltese', 154: 'Pekinese, Pekingese, Peke', 155: 'Shih-Tzu', 156: 'Blenheim spaniel', 157: 'papillon', 158: 'toy terrier', 159: 'Rhodesian ridgeback', 160: 'Afghan hound, Afghan', 161: 'basset, basset hound', 162: 'beagle', 163: 'bloodhound, sleuthhound', 164: 'bluetick', 165: 'black-and-tan coonhound', 166: 'Walker hound, Walker foxhound', 167: 'English foxhound', 168: 'redbone', 169: 'borzoi, Russian wolfhound', 170: 'Irish wolfhound', 171: 'Italian greyhound', 172: 'whippet', 173: 'Ibizan hound, Ibizan Podenco', 174: 'Norwegian elkhound, elkhound', 175: 'otterhound, otter hound', 176: 'Saluki, gazelle hound', 177: 'Scottish deerhound, deerhound', 178: 'Weimaraner', 179: 'Staffordshire bullterrier, Staffordshire bull terrier', 180: 'American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier', 181: 'Bedlington terrier', 182: 'Border terrier', 183: 'Kerry blue terrier', 184: 'Irish terrier', 185: 'Norfolk terrier', 186: 'Norwich terrier', 187: 'Yorkshire terrier', 188: 'wire-haired fox terrier', 189: 'Lakeland terrier', 190: 'Sealyham terrier, Sealyham', 191: 'Airedale, Airedale terrier', 192: 'cairn, cairn terrier', 193: 'Australian terrier', 194: 'Dandie Dinmont, Dandie Dinmont terrier', 195: 'Boston bull, Boston terrier', 196: 'miniature schnauzer', 197: 'giant schnauzer', 198: 'standard schnauzer', 199: 'Scotch terrier, Scottish terrier, Scottie', 200: 'Tibetan terrier, chrysanthemum dog', 201: 'silky terrier, Sydney silky', 202: 'soft-coated wheaten terrier', 203: 'West Highland white terrier', 204: 'Lhasa, Lhasa apso', 205: 'flat-coated retriever', 206: 'curly-coated retriever', 207: 'golden retriever', 208: 'Labrador retriever', 209: 'Chesapeake Bay retriever', 210: 'German short-haired pointer', 211: 'vizsla, Hungarian pointer', 212: 'English setter', 213: 'Irish setter, red setter', 214: 'Gordon setter', 215: 'Brittany spaniel', 216: 'clumber, clumber spaniel', 217: 'English springer, English springer spaniel', 218: 'Welsh springer spaniel', 219: 'cocker spaniel, English cocker spaniel, cocker', 220: 'Sussex spaniel', 221: 'Irish water spaniel', 222: 'kuvasz', 223: 'schipperke', 224: 'groenendael', 225: 'malinois', 226: 'briard', 227: 'kelpie', 228: 'komondor', 229: 'Old English sheepdog, bobtail', 230: 'Shetland sheepdog, Shetland sheep dog, Shetland', 231: 'collie', 232: 'Border collie', 233: 'Bouvier des Flandres, Bouviers des Flandres', 234: 'Rottweiler', 235: 'German shepherd, German shepherd dog, German police dog, alsatian', 236: 'Doberman, Doberman pinscher', 237: 'miniature pinscher', 238: 'Greater Swiss Mountain dog', 239: 'Bernese mountain dog', 240: 'Appenzeller', 241: 'EntleBucher', 242: 'boxer', 243: 'bull mastiff', 244: 'Tibetan mastiff', 245: 'French bulldog', 246: 'Great Dane', 247: 'Saint Bernard, St Bernard', 248: 'Eskimo dog, husky', 249: 'malamute, malemute, Alaskan malamute', 250: 'Siberian husky', 251: 'dalmatian, coach dog, carriage dog', 252: 'affenpinscher, monkey pinscher, monkey dog', 253: 'basenji', 254: 'pug, pug-dog', 255: 'Leonberg', 256: 'Newfoundland, Newfoundland dog', 257: 'Great Pyrenees', 258: 'Samoyed, Samoyede', 259: 'Pomeranian', 260: 'chow, chow chow', 261: 'keeshond', 262: 'Brabancon griffon', 263: 'Pembroke, Pembroke Welsh corgi', 264: 'Cardigan, Cardigan Welsh corgi', 265: 'toy poodle', 266: 'miniature poodle', 267: 'standard poodle', 268: 'Mexican hairless', 269: 'timber wolf, grey wolf, gray wolf, Canis lupus', 270: 'white wolf, Arctic wolf, Canis lupus tundrarum', 271: 'red wolf, maned wolf, Canis rufus, Canis niger', 272: 'coyote, prairie wolf, brush wolf, Canis latrans', 273: 'dingo, warrigal, warragal, Canis dingo', 274: 'dhole, Cuon alpinus', 275: 'African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus', 276: 'hyena, hyaena', 277: 'red fox, Vulpes vulpes', 278: 'kit fox, Vulpes macrotis', 279: 'Arctic fox, white fox, Alopex lagopus', 280: 'grey fox, gray fox, Urocyon cinereoargenteus', 281: 'tabby, tabby cat', 282: 'tiger cat', 283: 'Persian cat', 284: 'Siamese cat, Siamese', 285: 'Egyptian cat', 286: 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor', 287: 'lynx, catamount', 288: 'leopard, Panthera pardus', 289: 'snow leopard, ounce, Panthera uncia', 290: 'jaguar, panther, Panthera onca, Felis onca', 291: 'lion, king of beasts, Panthera leo', 292: 'tiger, Panthera tigris', 293: 'cheetah, chetah, Acinonyx jubatus', 294: 'brown bear, bruin, Ursus arctos', 295: 'American black bear, black bear, Ursus americanus, Euarctos americanus', 296: 'ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus', 297: 'sloth bear, Melursus ursinus, Ursus ursinus', 298: 'mongoose', 299: 'meerkat, mierkat', 300: 'tiger beetle', 301: 'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle', 302: 'ground beetle, carabid beetle', 303: 'long-horned beetle, longicorn, longicorn beetle', 304: 'leaf beetle, chrysomelid', 305: 'dung beetle', 306: 'rhinoceros beetle', 307: 'weevil', 308: 'fly', 309: 'bee', 310: 'ant, emmet, pismire', 311: 'grasshopper, hopper', 312: 'cricket', 313: 'walking stick, walkingstick, stick insect', 314: 'cockroach, roach', 315: 'mantis, mantid', 316: 'cicada, cicala', 317: 'leafhopper', 318: 'lacewing, lacewing fly', 319: "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk", 320: 'damselfly', 321: 'admiral', 322: 'ringlet, ringlet butterfly', 323: 'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus', 324: 'cabbage butterfly', 325: 'sulphur butterfly, sulfur butterfly', 326: 'lycaenid, lycaenid butterfly', 327: 'starfish, sea star', 328: 'sea urchin', 329: 'sea cucumber, holothurian', 330: 'wood rabbit, cottontail, cottontail rabbit', 331: 'hare', 332: 'Angora, Angora rabbit', 333: 'hamster', 334: 'porcupine, hedgehog', 335: 'fox squirrel, eastern fox squirrel, Sciurus niger', 336: 'marmot', 337: 'beaver', 338: 'guinea pig, Cavia cobaya', 339: 'sorrel', 340: 'zebra', 341: 'hog, pig, grunter, squealer, Sus scrofa', 342: 'wild boar, boar, Sus scrofa', 343: 'warthog', 344: 'hippopotamus, hippo, river horse, Hippopotamus amphibius', 345: 'ox', 346: 'water buffalo, water ox, Asiatic buffalo, Bubalus bubalis', 347: 'bison', 348: 'ram, tup', 349: 'bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis', 350: 'ibex, Capra ibex', 351: 'hartebeest', 352: 'impala, Aepyceros melampus', 353: 'gazelle', 354: 'Arabian camel, dromedary, Camelus dromedarius', 355: 'llama', 356: 'weasel', 357: 'mink', 358: 'polecat, fitch, foulmart, foumart, Mustela putorius', 359: 'black-footed ferret, ferret, Mustela nigripes', 360: 'otter', 361: 'skunk, polecat, wood pussy', 362: 'badger', 363: 'armadillo', 364: 'three-toed sloth, ai, Bradypus tridactylus', 365: 'orangutan, orang, orangutang, Pongo pygmaeus', 366: 'gorilla, Gorilla gorilla', 367: 'chimpanzee, chimp, Pan troglodytes', 368: 'gibbon, Hylobates lar', 369: 'siamang, Hylobates syndactylus, Symphalangus syndactylus', 370: 'guenon, guenon monkey', 371: 'patas, hussar monkey, Erythrocebus patas', 372: 'baboon', 373: 'macaque', 374: 'langur', 375: 'colobus, colobus monkey', 376: 'proboscis monkey, Nasalis larvatus', 377: 'marmoset', 378: 'capuchin, ringtail, Cebus capucinus', 379: 'howler monkey, howler', 380: 'titi, titi monkey', 381: 'spider monkey, Ateles geoffroyi', 382: 'squirrel monkey, Saimiri sciureus', 383: 'Madagascar cat, ring-tailed lemur, Lemur catta', 384: 'indri, indris, Indri indri, Indri brevicaudatus', 385: 'Indian elephant, Elephas maximus', 386: 'African elephant, Loxodonta africana', 387: 'lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens', 388: 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca', 389: 'barracouta, snoek', 390: 'eel', 391: 'coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch', 392: 'rock beauty, Holocanthus tricolor', 393: 'anemone fish', 394: 'sturgeon', 395: 'gar, garfish, garpike, billfish, Lepisosteus osseus', 396: 'lionfish', 397: 'puffer, pufferfish, blowfish, globefish', 398: 'abacus', 399: 'abaya', 400: "academic gown, academic robe, judge's robe", 401: 'accordion, piano accordion, squeeze box', 402: 'acoustic guitar', 403: 'aircraft carrier, carrier, flattop, attack aircraft carrier', 404: 'airliner', 405: 'airship, dirigible', 406: 'altar', 407: 'ambulance', 408: 'amphibian, amphibious vehicle', 409: 'analog clock', 410: 'apiary, bee house', 411: 'apron', 412: 'ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin', 413: 'assault rifle, assault gun', 414: 'backpack, back pack, knapsack, packsack, rucksack, haversack', 415: 'bakery, bakeshop, bakehouse', 416: 'balance beam, beam', 417: 'balloon', 418: 'ballpoint, ballpoint pen, ballpen, Biro', 419: 'Band Aid', 420: 'banjo', 421: 'bannister, banister, balustrade, balusters, handrail', 422: 'barbell', 423: 'barber chair', 424: 'barbershop', 425: 'barn', 426: 'barometer', 427: 'barrel, cask', 428: 'barrow, garden cart, lawn cart, wheelbarrow', 429: 'baseball', 430: 'basketball', 431: 'bassinet', 432: 'bassoon', 433: 'bathing cap, swimming cap', 434: 'bath towel', 435: 'bathtub, bathing tub, bath, tub', 436: 'beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon', 437: 'beacon, lighthouse, beacon light, pharos', 438: 'beaker', 439: 'bearskin, busby, shako', 440: 'beer bottle', 441: 'beer glass', 442: 'bell cote, bell cot', 443: 'bib', 444: 'bicycle-built-for-two, tandem bicycle, tandem', 445: 'bikini, two-piece', 446: 'binder, ring-binder', 447: 'binoculars, field glasses, opera glasses', 448: 'birdhouse', 449: 'boathouse', 450: 'bobsled, bobsleigh, bob', 451: 'bolo tie, bolo, bola tie, bola', 452: 'bonnet, poke bonnet', 453: 'bookcase', 454: 'bookshop, bookstore, bookstall', 455: 'bottlecap', 456: 'bow', 457: 'bow tie, bow-tie, bowtie', 458: 'brass, memorial tablet, plaque', 459: 'brassiere, bra, bandeau', 460: 'breakwater, groin, groyne, mole, bulwark, seawall, jetty', 461: 'breastplate, aegis, egis', 462: 'broom', 463: 'bucket, pail', 464: 'buckle', 465: 'bulletproof vest', 466: 'bullet train, bullet', 467: 'butcher shop, meat market', 468: 'cab, hack, taxi, taxicab', 469: 'caldron, cauldron', 470: 'candle, taper, wax light', 471: 'cannon', 472: 'canoe', 473: 'can opener, tin opener', 474: 'cardigan', 475: 'car mirror', 476: 'carousel, carrousel, merry-go-round, roundabout, whirligig', 477: "carpenter's kit, tool kit", 478: 'carton', 479: 'car wheel', 480: 'cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM', 481: 'cassette', 482: 'cassette player', 483: 'castle', 484: 'catamaran', 485: 'CD player', 486: 'cello, violoncello', 487: 'cellular telephone, cellular phone, cellphone, cell, mobile phone', 488: 'chain', 489: 'chainlink fence', 490: 'chain mail, ring mail, mail, chain armor, chain armour, ring armor, ring armour', 491: 'chain saw, chainsaw', 492: 'chest', 493: 'chiffonier, commode', 494: 'chime, bell, gong', 495: 'china cabinet, china closet', 496: 'Christmas stocking', 497: 'church, church building', 498: 'cinema, movie theater, movie theatre, movie house, picture palace', 499: 'cleaver, meat cleaver, chopper', 500: 'cliff dwelling', 501: 'cloak', 502: 'clog, geta, patten, sabot', 503: 'cocktail shaker', 504: 'coffee mug', 505: 'coffeepot', 506: 'coil, spiral, volute, whorl, helix', 507: 'combination lock', 508: 'computer keyboard, keypad', 509: 'confectionery, confectionary, candy store', 510: 'container ship, containership, container vessel', 511: 'convertible', 512: 'corkscrew, bottle screw', 513: 'cornet, horn, trumpet, trump', 514: 'cowboy boot', 515: 'cowboy hat, ten-gallon hat', 516: 'cradle', 517: 'crane', 518: 'crash helmet', 519: 'crate', 520: 'crib, cot', 521: 'Crock Pot', 522: 'croquet ball', 523: 'crutch', 524: 'cuirass', 525: 'dam, dike, dyke', 526: 'desk', 527: 'desktop computer', 528: 'dial telephone, dial phone', 529: 'diaper, nappy, napkin', 530: 'digital clock', 531: 'digital watch', 532: 'dining table, board', 533: 'dishrag, dishcloth', 534: 'dishwasher, dish washer, dishwashing machine', 535: 'disk brake, disc brake', 536: 'dock, dockage, docking facility', 537: 'dogsled, dog sled, dog sleigh', 538: 'dome', 539: 'doormat, welcome mat', 540: 'drilling platform, offshore rig', 541: 'drum, membranophone, tympan', 542: 'drumstick', 543: 'dumbbell', 544: 'Dutch oven', 545: 'electric fan, blower', 546: 'electric guitar', 547: 'electric locomotive', 548: 'entertainment center', 549: 'envelope', 550: 'espresso maker', 551: 'face powder', 552: 'feather boa, boa', 553: 'file, file cabinet, filing cabinet', 554: 'fireboat', 555: 'fire engine, fire truck', 556: 'fire screen, fireguard', 557: 'flagpole, flagstaff', 558: 'flute, transverse flute', 559: 'folding chair', 560: 'football helmet', 561: 'forklift', 562: 'fountain', 563: 'fountain pen', 564: 'four-poster', 565: 'freight car', 566: 'French horn, horn', 567: 'frying pan, frypan, skillet', 568: 'fur coat', 569: 'garbage truck, dustcart', 570: 'gasmask, respirator, gas helmet', 571: 'gas pump, gasoline pump, petrol pump, island dispenser', 572: 'goblet', 573: 'go-kart', 574: 'golf ball', 575: 'golfcart, golf cart', 576: 'gondola', 577: 'gong, tam-tam', 578: 'gown', 579: 'grand piano, grand', 580: 'greenhouse, nursery, glasshouse', 581: 'grille, radiator grille', 582: 'grocery store, grocery, food market, market', 583: 'guillotine', 584: 'hair slide', 585: 'hair spray', 586: 'half track', 587: 'hammer', 588: 'hamper', 589: 'hand blower, blow dryer, blow drier, hair dryer, hair drier', 590: 'hand-held computer, hand-held microcomputer', 591: 'handkerchief, hankie, hanky, hankey', 592: 'hard disc, hard disk, fixed disk', 593: 'harmonica, mouth organ, harp, mouth harp', 594: 'harp', 595: 'harvester, reaper', 596: 'hatchet', 597: 'holster', 598: 'home theater, home theatre', 599: 'honeycomb', 600: 'hook, claw', 601: 'hoopskirt, crinoline', 602: 'horizontal bar, high bar', 603: 'horse cart, horse-cart', 604: 'hourglass', 605: 'iPod', 606: 'iron, smoothing iron', 607: "jack-o'-lantern", 608: 'jean, blue jean, denim', 609: 'jeep, landrover', 610: 'jersey, T-shirt, tee shirt', 611: 'jigsaw puzzle', 612: 'jinrikisha, ricksha, rickshaw', 613: 'joystick', 614: 'kimono', 615: 'knee pad', 616: 'knot', 617: 'lab coat, laboratory coat', 618: 'ladle', 619: 'lampshade, lamp shade', 620: 'laptop, laptop computer', 621: 'lawn mower, mower', 622: 'lens cap, lens cover', 623: 'letter opener, paper knife, paperknife', 624: 'library', 625: 'lifeboat', 626: 'lighter, light, igniter, ignitor', 627: 'limousine, limo', 628: 'liner, ocean liner', 629: 'lipstick, lip rouge', 630: 'Loafer', 631: 'lotion', 632: 'loudspeaker, speaker, speaker unit, loudspeaker system, speaker system', 633: "loupe, jeweler's loupe", 634: 'lumbermill, sawmill', 635: 'magnetic compass', 636: 'mailbag, postbag', 637: 'mailbox, letter box', 638: 'maillot', 639: 'maillot, tank suit', 640: 'manhole cover', 641: 'maraca', 642: 'marimba, xylophone', 643: 'mask', 644: 'matchstick', 645: 'maypole', 646: 'maze, labyrinth', 647: 'measuring cup', 648: 'medicine chest, medicine cabinet', 649: 'megalith, megalithic structure', 650: 'microphone, mike', 651: 'microwave, microwave oven', 652: 'military uniform', 653: 'milk can', 654: 'minibus', 655: 'miniskirt, mini', 656: 'minivan', 657: 'missile', 658: 'mitten', 659: 'mixing bowl', 660: 'mobile home, manufactured home', 661: 'Model T', 662: 'modem', 663: 'monastery', 664: 'monitor', 665: 'moped', 666: 'mortar', 667: 'mortarboard', 668: 'mosque', 669: 'mosquito net', 670: 'motor scooter, scooter', 671: 'mountain bike, all-terrain bike, off-roader', 672: 'mountain tent', 673: 'mouse, computer mouse', 674: 'mousetrap', 675: 'moving van', 676: 'muzzle', 677: 'nail', 678: 'neck brace', 679: 'necklace', 680: 'nipple', 681: 'notebook, notebook computer', 682: 'obelisk', 683: 'oboe, hautboy, hautbois', 684: 'ocarina, sweet potato', 685: 'odometer, hodometer, mileometer, milometer', 686: 'oil filter', 687: 'organ, pipe organ', 688: 'oscilloscope, scope, cathode-ray oscilloscope, CRO', 689: 'overskirt', 690: 'oxcart', 691: 'oxygen mask', 692: 'packet', 693: 'paddle, boat paddle', 694: 'paddlewheel, paddle wheel', 695: 'padlock', 696: 'paintbrush', 697: "pajama, pyjama, pj's, jammies", 698: 'palace', 699: 'panpipe, pandean pipe, syrinx', 700: 'paper towel', 701: 'parachute, chute', 702: 'parallel bars, bars', 703: 'park bench', 704: 'parking meter', 705: 'passenger car, coach, carriage', 706: 'patio, terrace', 707: 'pay-phone, pay-station', 708: 'pedestal, plinth, footstall', 709: 'pencil box, pencil case', 710: 'pencil sharpener', 711: 'perfume, essence', 712: 'Petri dish', 713: 'photocopier', 714: 'pick, plectrum, plectron', 715: 'pickelhaube', 716: 'picket fence, paling', 717: 'pickup, pickup truck', 718: 'pier', 719: 'piggy bank, penny bank', 720: 'pill bottle', 721: 'pillow', 722: 'ping-pong ball', 723: 'pinwheel', 724: 'pirate, pirate ship', 725: 'pitcher, ewer', 726: "plane, carpenter's plane, woodworking plane", 727: 'planetarium', 728: 'plastic bag', 729: 'plate rack', 730: 'plow, plough', 731: "plunger, plumber's helper", 732: 'Polaroid camera, Polaroid Land camera', 733: 'pole', 734: 'police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria', 735: 'poncho', 736: 'pool table, billiard table, snooker table', 737: 'pop bottle, soda bottle', 738: 'pot, flowerpot', 739: "potter's wheel", 740: 'power drill', 741: 'prayer rug, prayer mat', 742: 'printer', 743: 'prison, prison house', 744: 'projectile, missile', 745: 'projector', 746: 'puck, hockey puck', 747: 'punching bag, punch bag, punching ball, punchball', 748: 'purse', 749: 'quill, quill pen', 750: 'quilt, comforter, comfort, puff', 751: 'racer, race car, racing car', 752: 'racket, racquet', 753: 'radiator', 754: 'radio, wireless', 755: 'radio telescope, radio reflector', 756: 'rain barrel', 757: 'recreational vehicle, RV, R.V.', 758: 'reel', 759: 'reflex camera', 760: 'refrigerator, icebox', 761: 'remote control, remote', 762: 'restaurant, eating house, eating place, eatery', 763: 'revolver, six-gun, six-shooter', 764: 'rifle', 765: 'rocking chair, rocker', 766: 'rotisserie', 767: 'rubber eraser, rubber, pencil eraser', 768: 'rugby ball', 769: 'rule, ruler', 770: 'running shoe', 771: 'safe', 772: 'safety pin', 773: 'saltshaker, salt shaker', 774: 'sandal', 775: 'sarong', 776: 'sax, saxophone', 777: 'scabbard', 778: 'scale, weighing machine', 779: 'school bus', 780: 'schooner', 781: 'scoreboard', 782: 'screen, CRT screen', 783: 'screw', 784: 'screwdriver', 785: 'seat belt, seatbelt', 786: 'sewing machine', 787: 'shield, buckler', 788: 'shoe shop, shoe-shop, shoe store', 789: 'shoji', 790: 'shopping basket', 791: 'shopping cart', 792: 'shovel', 793: 'shower cap', 794: 'shower curtain', 795: 'ski', 796: 'ski mask', 797: 'sleeping bag', 798: 'slide rule, slipstick', 799: 'sliding door', 800: 'slot, one-armed bandit', 801: 'snorkel', 802: 'snowmobile', 803: 'snowplow, snowplough', 804: 'soap dispenser', 805: 'soccer ball', 806: 'sock', 807: 'solar dish, solar collector, solar furnace', 808: 'sombrero', 809: 'soup bowl', 810: 'space bar', 811: 'space heater', 812: 'space shuttle', 813: 'spatula', 814: 'speedboat', 815: "spider web, spider's web", 816: 'spindle', 817: 'sports car, sport car', 818: 'spotlight, spot', 819: 'stage', 820: 'steam locomotive', 821: 'steel arch bridge', 822: 'steel drum', 823: 'stethoscope', 824: 'stole', 825: 'stone wall', 826: 'stopwatch, stop watch', 827: 'stove', 828: 'strainer', 829: 'streetcar, tram, tramcar, trolley, trolley car', 830: 'stretcher', 831: 'studio couch, day bed', 832: 'stupa, tope', 833: 'submarine, pigboat, sub, U-boat', 834: 'suit, suit of clothes', 835: 'sundial', 836: 'sunglass', 837: 'sunglasses, dark glasses, shades', 838: 'sunscreen, sunblock, sun blocker', 839: 'suspension bridge', 840: 'swab, swob, mop', 841: 'sweatshirt', 842: 'swimming trunks, bathing trunks', 843: 'swing', 844: 'switch, electric switch, electrical switch', 845: 'syringe', 846: 'table lamp', 847: 'tank, army tank, armored combat vehicle, armoured combat vehicle', 848: 'tape player', 849: 'teapot', 850: 'teddy, teddy bear', 851: 'television, television system', 852: 'tennis ball', 853: 'thatch, thatched roof', 854: 'theater curtain, theatre curtain', 855: 'thimble', 856: 'thresher, thrasher, threshing machine', 857: 'throne', 858: 'tile roof', 859: 'toaster', 860: 'tobacco shop, tobacconist shop, tobacconist', 861: 'toilet seat', 862: 'torch', 863: 'totem pole', 864: 'tow truck, tow car, wrecker', 865: 'toyshop', 866: 'tractor', 867: 'trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi', 868: 'tray', 869: 'trench coat', 870: 'tricycle, trike, velocipede', 871: 'trimaran', 872: 'tripod', 873: 'triumphal arch', 874: 'trolleybus, trolley coach, trackless trolley', 875: 'trombone', 876: 'tub, vat', 877: 'turnstile', 878: 'typewriter keyboard', 879: 'umbrella', 880: 'unicycle, monocycle', 881: 'upright, upright piano', 882: 'vacuum, vacuum cleaner', 883: 'vase', 884: 'vault', 885: 'velvet', 886: 'vending machine', 887: 'vestment', 888: 'viaduct', 889: 'violin, fiddle', 890: 'volleyball', 891: 'waffle iron', 892: 'wall clock', 893: 'wallet, billfold, notecase, pocketbook', 894: 'wardrobe, closet, press', 895: 'warplane, military plane', 896: 'washbasin, handbasin, washbowl, lavabo, wash-hand basin', 897: 'washer, automatic washer, washing machine', 898: 'water bottle', 899: 'water jug', 900: 'water tower', 901: 'whiskey jug', 902: 'whistle', 903: 'wig', 904: 'window screen', 905: 'window shade', 906: 'Windsor tie', 907: 'wine bottle', 908: 'wing', 909: 'wok', 910: 'wooden spoon', 911: 'wool, woolen, woollen', 912: 'worm fence, snake fence, snake-rail fence, Virginia fence', 913: 'wreck', 914: 'yawl', 915: 'yurt', 916: 'web site, website, internet site, site', 917: 'comic book', 918: 'crossword puzzle, crossword', 919: 'street sign', 920: 'traffic light, traffic signal, stoplight', 921: 'book jacket, dust cover, dust jacket, dust wrapper', 922: 'menu', 923: 'plate', 924: 'guacamole', 925: 'consomme', 926: 'hot pot, hotpot', 927: 'trifle', 928: 'ice cream, icecream', 929: 'ice lolly, lolly, lollipop, popsicle', 930: 'French loaf', 931: 'bagel, beigel', 932: 'pretzel', 933: 'cheeseburger', 934: 'hotdog, hot dog, red hot', 935: 'mashed potato', 936: 'head cabbage', 937: 'broccoli', 938: 'cauliflower', 939: 'zucchini, courgette', 940: 'spaghetti squash', 941: 'acorn squash', 942: 'butternut squash', 943: 'cucumber, cuke', 944: 'artichoke, globe artichoke', 945: 'bell pepper', 946: 'cardoon', 947: 'mushroom', 948: 'Granny Smith', 949: 'strawberry', 950: 'orange', 951: 'lemon', 952: 'fig', 953: 'pineapple, ananas', 954: 'banana', 955: 'jackfruit, jak, jack', 956: 'custard apple', 957: 'pomegranate', 958: 'hay', 959: 'carbonara', 960: 'chocolate sauce, chocolate syrup', 961: 'dough', 962: 'meat loaf, meatloaf', 963: 'pizza, pizza pie', 964: 'potpie', 965: 'burrito', 966: 'red wine', 967: 'espresso', 968: 'cup', 969: 'eggnog', 970: 'alp', 971: 'bubble', 972: 'cliff, drop, drop-off', 973: 'coral reef', 974: 'geyser', 975: 'lakeside, lakeshore', 976: 'promontory, headland, head, foreland', 977: 'sandbar, sand bar', 978: 'seashore, coast, seacoast, sea-coast', 979: 'valley, vale', 980: 'volcano', 981: 'ballplayer, baseball player', 982: 'groom, bridegroom', 983: 'scuba diver', 984: 'rapeseed', 985: 'daisy', 986: "yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum", 987: 'corn', 988: 'acorn', 989: 'hip, rose hip, rosehip', 990: 'buckeye, horse chestnut, conker', 991: 'coral fungus', 992: 'agaric', 993: 'gyromitra', 994: 'stinkhorn, carrion fungus', 995: 'earthstar', 996: 'hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa', 997: 'bolete', 998: 'ear, spike, capitulum', 999: 'toilet tissue, toilet paper, bathroom tissue'}

### Load COCO 2017 and COCO-Stuff Datasets ###

# Input images were taken from the validation set of the COCO 2017 dataset: https://cocodataset.org
# Concept segmentations, used to generate ground truth explanations aand output class vectors, were taken from the validation set of the COCO-Stuff dataset: https://github.com/nightrome/cocostuff

import cv2
import numpy as np

from Dataset.Dataset import *
from DatasetCoco.DatasetCoco import DatasetCoco

data = DatasetCoco(folders=["val2017"], calc_concepts_in_images = settings_global['calc_concepts_in_images'])

concept_dataloader_test = Dataloader__daan__concepts_test(8, data)
concept_dataloader_test_lazy = Dataloader__daan__concepts_test_lazy(8, data)

### Determine which concepts are in both Imagenet and COCO. Each of these become an output class of the model. ###

concepts = []
concepts_i = []

for label_c in coco:
	for label_i in imagenet:
		if ", "+coco[label_c]+"," in ", "+imagenet[label_i]+", ":
			print(label_c,"&",coco[label_c],"&",label_i,"&",imagenet[label_i],"\\\\")
			concepts.append(label_c)
			concepts_i.append(label_i)

#print(concepts)

print("Our experiments involve ",len(concepts),"classes.")

# The weights of all incoming connections to all output nodes corresponding to classes are set to 0. Also, the bias values of these output nodes are set to 0. The weights of the incoming connections to all other output nodes are  multiplied by 2.

import tensorflow as tf

input_layer = tf.keras.layers.Input(shape=(224, 224, 3))
model_classifier = tf.keras.applications.ResNet50(include_top=True,weights="imagenet",input_tensor=input_layer)
model_classifier.compile(optimizer="adam",loss="categorical_crossentropy",metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

weights = model_classifier.get_layer("predictions").get_weights()
for i in range(1000):
  if not i in concepts_i:
    weights[0][:,i] = 0
    weights[1][i] = 0
  else:
    weights[0][:,i] = weights[0][:,i]*2

model_classifier.get_layer("predictions").set_weights(weights)

model_classifier.compile(optimizer="adam",loss="categorical_crossentropy",metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

### Determine Input Examples, Class Vectors, and 'Ground Truth' Explanations ###

if settings_global['determine_concept_exanmples']:

  examples_items = []
  examples = list(np.zeros(1000))
  index = 0

  for i1 in range(concept_dataloader_test_lazy.__len__()-1):
    items = concept_dataloader_test_lazy.__getitem__(i1)

    for i2 in range(8):
      item = np.nonzero(items[1][i2])[0]
      conceots_in_image_in_concepts = [x for x in item if x in concepts]
      if len(conceots_in_image_in_concepts)==1: # Check if one of the concepts is in the image
        items2 = concept_dataloader_test.__getitem__(i1)

        pred = model_classifier.predict(np.array([items2[0][i2]]),verbose=0)
        pred = np.argmax(pred[0])

        id_imagenet = concepts_i[concepts.index(conceots_in_image_in_concepts[0])]

        if pred == id_imagenet:

            # Determine class ID
            class_id = list(np.zeros(1000))
            class_id[id_imagenet] = 1

            # Generate GT explanation
            exp_org = cv2.imread(data.paths_concept["val2017"][index][0], cv2.IMREAD_GRAYSCALE)
            exp = np.zeros((exp_org.shape[0],exp_org.shape[1]))

            exp[exp_org == conceots_in_image_in_concepts[0]] = 1

            # Resize to (224,224), the size of input examples.
            scale = np.min([exp.shape[0], exp.shape[1]])
            new_shape = [int(np.ceil(scale * x)) for x in [exp.shape[0], exp.shape[1]]]
            exp = exp[0:scale,0:scale]
            exp = cv2.resize(exp, (224,224), interpolation=cv2.INTER_NEAREST)

            if np.amax(exp) >= 1:
              print("len:",len(examples_items))
              examples_items.append([items2[0][i2], class_id, exp])
              examples += class_id

            print((i1/concept_dataloader_test_lazy.__len__()))

      index += 1

  with open('./DatasetCoco/examples_items_RESNET50.npy', 'wb') as f:
    np.save(f, examples_items)

else: # Load examples computed earlier

  with open('./DatasetCoco/examples_items_RESNET50.npy', 'rb') as f:
    examples_items = np.load(f, allow_pickle=True)

data_x = [] # Input examples
data_y = [] # Output classes of input examples
data_exp_2d = []  # Ground truth explanations w.r.t. the output class of input examples. One value per (x,y) coordinate.
data_exp_3d = []  # Ground truth explanations w.r.t. the output class of input examples. One value per (x,y,c) coordinate.

num_examples_total = len(examples_items)

print("num_examples_total",num_examples_total)

for item in examples_items:
  image = item[0]
  class_id = item[1]
  exp = item[2]

  data_x.append(image)
  data_y.append(class_id)
  data_exp_2d.append(exp)
  exp_3d = np.zeros((224,224,3))
  exp_3d[:,:,0] = exp
  exp_3d[:,:,1] = exp
  exp_3d[:,:,2] = exp
  data_exp_3d.append(exp_3d)

data_x = np.array(data_x)
data_y = np.array(data_y)
data_exp_2d = np.array(data_exp_2d)
data_exp_3d = np.array(data_exp_3d)


# Print the shape, minimum, maximum and average value of data_x, data_y, data_exp_2d, and data_exp_3d

print("data_x", data_x.shape, np.min(data_x),np.max(data_x),np.average(data_x))
print("data_y", data_y.shape, np.min(data_y),np.max(data_y),np.average(data_y))
print("data_exp_2d", data_exp_2d.shape, np.min(data_exp_2d),np.max(data_exp_2d),np.average(data_exp_2d))
print("data_exp_3d", data_exp_3d.shape, np.min(data_exp_3d),np.max(data_exp_3d),np.average(data_exp_3d))

if True: # Evaluate model_classifier's accuracy.
  model_classifier.evaluate(data_x, data_y)

### Divide the input examples over all runs. ###

items_per_run = int(np.floor(num_examples_total/settings_global['num_runs']))

from_id = settings_global['run_id']*items_per_run
to_id = (settings_global['run_id']+1)*items_per_run

data_x = np.array(data_x[from_id:to_id])
data_y = np.array(data_y[from_id:to_id])
data_exp_2d = np.array(data_exp_2d[from_id:to_id])
data_exp_3d = np.array(data_exp_3d[from_id:to_id])

### Compute explanations using XAI methods on the data computed above, and compute XAI metrics of these explanations ###

# ---------------------- Import Libs ---------------------- #

from xai_metrics.Our_CompactnessMetric import Our_CompactnessMetric
from xai_metrics.Our_CompletenessMetric import Our_CompletenessMetric
from xai_metrics.Our_CorrectnessMetric import Our_CorrectnessMetric

import cv2

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *

import pandas as pd

import numpy as np

import time

from xplique.attributions import DeconvNet,GradCAM,GradCAMPP,GradientInput,GuidedBackprop,IntegratedGradients,Saliency,SmoothGrad,SquareGrad,VarGrad,KernelShap,Lime,Occlusion,Rise
from xplique.metrics import MuFidelity,Deletion,Insertion

from skimage.metrics import structural_similarity

import sklearn.metrics

import os


# ---- coco ---#

import sys

from Dataset.Dataset import *

from DatasetCoco.DatasetCoco import DatasetCoco


# ---------------------- Global Settings ---------------------- #

settings = {}

settings['num_examples_per_class'] = 16
settings['normalize_explanations'] = True

settings['saveSelectedImages'] = True
settings['saveAllImages'] = False
settings['resize_Images'] = False
settings['batchsize'] = 16
settings['dataset'] = ["coco","synth"][0]

all_xai_methods = [GradCAM,GradCAMPP,Saliency,DeconvNet, GradientInput, GuidedBackprop, IntegratedGradients, SmoothGrad, SquareGrad,VarGrad,KernelShap,Occlusion,Rise] #,Lime] Excluded due to "Weights sum to zero, can't be normalized" errors
all_xai_methods_names = ["GradCAM", "GradCAMPP", "Saliency","DeconvNet","GradientInput", "GuidedBackprop", "IntegratedGradients", "SmoothGrad", "SquareGrad","VarGrad","KernelShap","Occlusion","Rise"]# ,"Lime"]

all_xai_metrics = [Deletion,Insertion] #,MuFidelity] # Excluded due to "InternalError: stream did not block host until done; was already in an error state" errors
all_xai_metrics_names = ["Deletion","Insertion"] #,"MuFidelity"]


# ---------------------- Functions ---------------------- #

def checkNum(num):
    if (num == None) or (not np.isreal(num)) or (np.isnan(num)):
        print("ERROR checkNum: ",num)
        #exit()

def checkList(list_var):

    for num in list_var:
        checkNum(num)

def saveImages(path,images,conceptid, applyColormap,data_test_y,postfix=""):

    for img,id in zip(images,range(len(images))):

        if settings['resize_Images']:
            img2 = cv2.resize(img,(72,72), interpolation = cv2.INTER_NEAREST)
        else:
            img2 = img

        if applyColormap:
            saveImageColomap(path+str(conceptid)+"_"+str(np.argmax(data_test_y[id]))+"_"+str(id)+postfix+".png", img2)
        else:
            saveImage(path+str(conceptid)+"_"+str(np.argmax(data_test_y[id]))+"_"+str(id)+postfix+".png", img2)

def saveImage(path, image):
    cv2.imwrite(path, image*255)

def convert_to_2D(explanation):
    output = np.zeros((explanation.shape[0], explanation.shape[1]))
    for y in range(explanation.shape[0]):
        for x in range(explanation.shape[1]):
            mostExtremeValue = 0.0

            for c in range(3):
                if abs(explanation[y,x,c]) > abs(mostExtremeValue):
                    if False:
                      if mostExtremeValue != 0.0:
                          print("ERROR!!!!! mostExtremeValue != 0")
                          exit()
                    mostExtremeValue = explanation[y,x,c]

            output[y,x] = mostExtremeValue

    return output

def saveImageColomap(path, image):
    if image.shape[-1] == 3:
        image2 =  convert_to_2D(image)
    else:
        image2 = image

    scaled = (((image2 + 1)/2.0)*255).astype(np.uint8)
    im_color = cv2.applyColorMap(scaled, cv2.COLORMAP_CIVIDIS)
    cv2.imwrite(path, im_color)


def saveColormap():

    img = np.zeros((256,10))

    for i in range(0,256):
        img[i,:] = (i/255.0)*2-1

    saveImageColomap("./paper_images/selected/colormap.png", img)

def startTimer():
    return time.time()

def endTimer(start_time, num_items):
    return ((time.time() - start_time)*1000.0) / num_items


def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


def flatten_and_turn_binary(explanations_copy,gt_explanation_copy):
    explanations_copy_flat = np.zeros((explanations_copy.shape[0], np.prod(explanations_copy.shape[1:])), dtype='bool')
    gt_explanation_copy_flat = np.zeros((gt_explanation_copy.shape[0], np.prod(gt_explanation_copy.shape[1:])), dtype='bool')

    for idx_iter in range(gt_explanation_copy.shape[0]):

        gt_binary = np.zeros(gt_explanation_copy[idx_iter].shape, dtype='bool')
        gt_binary[gt_explanation_copy[idx_iter] != 0] = True

        gt_explanation_copy_flat[idx_iter] = gt_binary.flatten()


        explanation_binary = np.zeros(explanations_copy[idx_iter].shape, dtype='bool')
        explanation_binary[explanations_copy[idx_iter] != 0] = True

        explanations_copy_flat[idx_iter] = explanation_binary.flatten()

    return explanations_copy_flat, gt_explanation_copy_flat

def turn_binary(explanations_copy, invert):

    if invert:
        explanations_copy_binary = np.ones(explanations_copy.shape, dtype='bool')

        explanations_copy_binary[explanations_copy != 0] = False
    else:
        explanations_copy_binary = np.zeros(explanations_copy.shape, dtype='bool')

        explanations_copy_binary[explanations_copy != 0] = True

    return explanations_copy_binary


def flatten_only(explanations_copy,gt_explanation_copy):
    explanations_copy_flat = np.zeros((explanations_copy.shape[0], np.prod(explanations_copy.shape[1:])))
    gt_explanation_copy_flat = np.zeros((gt_explanation_copy.shape[0], np.prod(gt_explanation_copy.shape[1:])))

    for idx_iter in range(gt_explanation_copy.shape[0]):
        gt_explanation_copy_flat[idx_iter] = gt_explanation_copy[idx_iter].flatten()
        explanations_copy_flat[idx_iter] = explanations_copy[idx_iter].flatten()

    return explanations_copy_flat, gt_explanation_copy_flat

def normalize_explanation(exp):

    max = np.amax(exp)
    min = np.amin(exp)

    if max != 0 or min != 0:
        if abs(min) > abs(max):
            exp = exp * 1.0/abs(min)
        else:
            exp = exp * 1.0/abs(max)

    if (np.amin(exp) < -1 or np.amax(exp) > 1) or ((np.amin(exp) != -1 and np.amax(exp) != 1) and not (np.amin(exp) == 0 and np.amax(exp) == 0)):
        print("ERROR: np.amin(exp) < -1 or np.amax(exp) > 1 ")
        print(np.amin(exp),np.amax(exp))
        #exit()

    return exp

def average_dicts(dict_list):
  """Computes the average of two dicts element wise.

  Args:
    dict1: The first dict.
    dict2: The second dict.

  Returns:
    A dict with the average of each key-value pair in dict1 and dict2.
  """
  result = {}
  for dicts in dict_list:
      for key, value in dicts.items():
        if key in result:
          result[key] = (value + result[key])
        else:
          result[key] = value

  for key, value in result.items():
      result[key] = value / len(dict_list)

  return result



def calc_metrics_for_XAI_method(Xai_method, xai_name, model2, data, conceptid):
        print("*** ",xai_name," ***")

        result_data_xaimethod = []
        result_data_xaimethod_time = []




        result_data_xaimethod.append({})
        result_data_xaimethod_time.append({})

        data_test_x, data_test_y = data_x, data_y

        num_items = data_test_y.shape[0]


        # Run xai method and time

        start_time = startTimer()

        explainer = Xai_method(model2,batch_size=1)

        explanations = explainer(data_test_x, data_test_y)
        if not isinstance(explanations, np.ndarray):
            explanations = explanations.numpy()
        xai_method_time = endTimer(start_time, num_items)

        # normalize explanations
        if settings['normalize_explanations']:
            for i in range(len(explanations)):
                explanations[i] = normalize_explanation(explanations[i])
        else:
            print("WARNING! NOT NORMALIZING EXPLANATIONS! ")



        # Determine GT: 2D or 3D?

        if explanations.shape[-1] == 3:# 3D GT

            print("3D explanations")

            gt_explanation = data_exp_3d

            # mufidelity fix
            for i in range(len(explanations)):
                if abs(explanations[i]).sum() == 0:
                    explanations[i][0][0][:] = 0.000000001


        else: #2D GT

            print("2D explanations")

            gt_explanation = data_exp_2d


            # mufidelity fix
            for i in range(len(explanations)):
                if abs(explanations[i]).sum() == 0:
                    explanations[i][0][0] = 0.000000001

        print("stats data_test_x: ",np.min(data_test_x),np.max(data_test_x),np.sum(data_test_x))
        print("stats data_test_y: ",np.min(data_test_y),np.max(data_test_y),np.sum(data_test_y))
        print("stats explanations: ",np.min(explanations),np.max(explanations),np.sum(explanations))
        print("stats GT explanations: ",np.min(gt_explanation),np.max(gt_explanation),np.sum(gt_explanation))


        if settings_global['calc_metrics']: # metric calc switch

          # Calc prior metrics

          for Metric, metric_name in zip(all_xai_metrics,all_xai_metrics_names):
              print(metric_name)

              data_x_copy = data_test_x.astype('float32')
              data_y_copy = data_test_y.astype('float32')
              explanations_copy = explanations.astype('float32')

              start_time = startTimer()
              explainer = Metric(model2, data_x_copy, data_y_copy)
              result_data_xaimethod[-1][metric_name] =  explainer(explanations_copy)
              result_data_xaimethod_time[-1][metric_name] = endTimer(start_time, num_items)

              checkNum(result_data_xaimethod[-1][metric_name])




          # Cosine Similarity (GUIDOTTI2021103428)
          print("CosSim")

          explanations_copy = explanations.astype('float32')
          gt_explanation_copy = gt_explanation.astype('float32')


          explanations_copy_flat, gt_explanation_copy_flat = flatten_only(explanations_copy, gt_explanation_copy)

          start_time = startTimer()
          scores = []
          for gt, explanation in zip(gt_explanation_copy_flat, explanations_copy_flat):

              score = np.linalg.norm(np.dot(explanation,gt))/(np.linalg.norm(explanation)*np.linalg.norm(gt))

              scores.append(score)

          result_data_xaimethod_time[-1]['CosSim'] = endTimer(start_time, num_items)
          result_data_xaimethod[-1]['CosSim'] = sum(scores) / len(scores)

          checkList(scores)




          # Cosine Similarity (GUIDOTTI2021103428)

          explanations_copy = explanations.astype('float32')
          gt_explanation_copy = gt_explanation.astype('float32')


          explanations_copy_flat, gt_explanation_copy_flat = flatten_only(explanations_copy, gt_explanation_copy)

          start_time = startTimer()
          scores = []
          for gt, explanation in zip(gt_explanation_copy_flat, explanations_copy_flat):

              score = np.dot(explanation,gt)/(np.linalg.norm(explanation)*np.linalg.norm(gt))

              scores.append(score)

          result_data_xaimethod_time[-1]['_CosSim'] = endTimer(start_time, num_items)
          result_data_xaimethod[-1]['_CosSim'] = sum(scores) / len(scores)

          checkList(scores)






          # SSIM
          print("SSIM")

          explanations_copy = explanations.astype('float32')
          gt_explanation_copy = gt_explanation.astype('float32')


          start_time = startTimer()
          scores = []

          for gt, explanation in zip(gt_explanation_copy, explanations_copy):

              if explanation.shape[-1] == 3:
                  score = structural_similarity(gt,explanation, channel_axis=-1, data_range=2)
              else:
                  score = structural_similarity(gt, explanation, data_range=2)

              scores.append(score)

          result_data_xaimethod_time[-1]['SSIM'] = endTimer(start_time, num_items)
          result_data_xaimethod[-1]['SSIM'] = sum(scores) / len(scores)

          checkList(scores)




          # CONCISSENESS (Amparore_2021)
          print("CONCISSENESS")

          explanations_copy = explanations.astype('bool')

          start_time = startTimer()
          scores = []
          for explanation in explanations_copy:
              total = np.prod(explanation.shape)

              iszero = (total - np.count_nonzero(explanation))+0.0

              score = iszero / total

              scores.append(score)


          result_data_xaimethod_time[-1]['Conciseness'] = endTimer(start_time,num_items )
          result_data_xaimethod[-1]['Conciseness'] = sum(scores) / len(scores)

          checkList(scores)




          # F1
          print("F1")

          explanations_copy = explanations.astype('float32')
          gt_explanation_copy = gt_explanation.astype('float32')

          explanations_copy_flat, gt_explanation_copy_flat = flatten_and_turn_binary(explanations_copy,gt_explanation_copy)

          start_time = startTimer()
          scores = []
          for gt, explanation in zip(gt_explanation_copy_flat, explanations_copy_flat):

              score = sklearn.metrics.f1_score(gt, explanation, average='binary', pos_label=1)

              scores.append(score)


          result_data_xaimethod_time[-1]['F1'] = endTimer(start_time, num_items)
          result_data_xaimethod[-1]['F1'] = sum(scores) / len(scores)

          checkList(scores)





          # MAE
          print("MAE")

          explanations_copy = explanations.astype('float32')
          gt_explanation_copy = gt_explanation.astype('float32')

          explanations_copy_flat, gt_explanation_copy_flat = flatten_only(explanations_copy, gt_explanation_copy)

          start_time = startTimer()
          scores = []
          for gt, explanation in zip(gt_explanation_copy_flat, explanations_copy_flat):

              score = sklearn.metrics.mean_absolute_error(gt, explanation)

              scores.append(score)


          result_data_xaimethod_time[-1]['MAE'] = endTimer(start_time, num_items)
          result_data_xaimethod[-1]['MAE'] = sum(scores) / len(scores)

          checkList(scores)




          # Intersection over union IoU
          print("IoU")

          explanations_copy = explanations.astype('float32')
          gt_explanation_copy = gt_explanation.astype('float32')

          explanations_copy_flat, gt_explanation_copy_flat = flatten_and_turn_binary(explanations_copy,gt_explanation_copy)

          start_time = startTimer()
          scores = []
          for gt, explanation in zip(gt_explanation_copy_flat, explanations_copy_flat):

              score = sklearn.metrics.jaccard_score(gt, explanation, average='binary', pos_label=1)

              scores.append(score)

          result_data_xaimethod_time[-1]['IoU'] = endTimer(start_time, num_items)
          result_data_xaimethod[-1]['IoU'] = sum(scores) / len(scores)

          checkList(scores)




          # Precision
          print("Precision")

          explanations_copy = explanations.astype('float32')
          gt_explanation_copy = gt_explanation.astype('float32')

          explanations_copy_flat, gt_explanation_copy_flat = flatten_and_turn_binary(explanations_copy,gt_explanation_copy)

          start_time = startTimer()
          scores = []
          for gt, explanation in zip(gt_explanation_copy_flat, explanations_copy_flat):

              score = sklearn.metrics.precision_score(gt, explanation, average='binary', pos_label=1)

              scores.append(score)


          result_data_xaimethod_time[-1]['PR'] = endTimer(start_time, num_items)
          result_data_xaimethod[-1]['PR'] = sum(scores) / len(scores)

          checkList(scores)






          # Recall
          print("Recall")

          explanations_copy = explanations.astype('float32')
          gt_explanation_copy = gt_explanation.astype('float32')

          explanations_copy_flat, gt_explanation_copy_flat = flatten_and_turn_binary(explanations_copy,gt_explanation_copy)

          start_time = startTimer()
          scores = []
          for gt, explanation in zip(gt_explanation_copy_flat, explanations_copy_flat):

              score = sklearn.metrics.recall_score(gt, explanation, average='binary', pos_label=1)

              scores.append(score)


          result_data_xaimethod_time[-1]['RE'] = endTimer(start_time, num_items)
          result_data_xaimethod[-1]['RE'] = sum(scores) / len(scores)

          checkList(scores)






          # Energy-Based Pointing Game (DBLP:journals/corr/abs-1910-01279)
          print("Energy-Based Pointing Game")

          explanations_copy = explanations.astype('float32')
          gt_explanation_copy = gt_explanation.astype('float32')

          gt_explanation_copy_binary = turn_binary(gt_explanation_copy,True)

          start_time = startTimer()
          scores = []
          for gt, explanation in zip(gt_explanation_copy_binary, explanations_copy):

              top = np.ma.array(abs(explanation), mask = gt).sum()
              if not top:
                  top = 0.0

              bottom = abs(explanation).sum()

              if top == 0 and bottom == 0:
                  score = 1.0
              elif top == 0:
                  score = 0.0
              else:
                  score = top / bottom

              scores.append(score)



          result_data_xaimethod_time[-1]['EBPG'] = endTimer(start_time, num_items)
          result_data_xaimethod[-1]['EBPG'] = sum(scores) / len(scores)

          checkList(scores)




          # Relevance Rank Accuracy (arras2021ground)
          print("Relevance Rank Accuracy")

          explanations_copy = explanations.astype('float32')
          gt_explanation_copy = gt_explanation.astype('float32')

          gt_explanation_copy_binary = turn_binary(gt_explanation_copy,False)



          start_time = startTimer()
          scores = []
          for gt, explanation in zip(gt_explanation_copy_binary, explanations_copy):

              num_K = np.count_nonzero(gt)

              kth_highest_value = np.partition(abs(explanation).flatten(), -(num_K))[-(num_K)]

              exp_greater_k = np.zeros(explanation.shape, dtype='bool')
              exp_greater_k[abs(explanation) >= kth_highest_value] = True
              exp_greater_k[gt == 0] = False

              score =  exp_greater_k.sum() / num_K

              scores.append(score)

          result_data_xaimethod_time[-1]['RRA'] = endTimer(start_time, num_items)
          result_data_xaimethod[-1]['RRA'] = sum(scores) / len(scores)

          checkList(scores)






























          for version_soft in ["abs","pos","neg"]:


              # Soft Precision (AttributionLab)
              print("Soft Precision "+version_soft)

              explanations_copy = explanations.astype('float32')
              gt_explanation_copy = gt_explanation.astype('float32')


              if version_soft == "abs":
                  explanations_copy = np.abs(explanations_copy)
                  gt_explanation_copy = np.abs(gt_explanation_copy)
              elif version_soft == "pos":
                  explanations_copy[explanations_copy<0] = 0
                  gt_explanation_copy[gt_explanation_copy<0] = 0
              elif version_soft == "neg":
                  explanations_copy[explanations_copy>0] = 0
                  gt_explanation_copy[gt_explanation_copy>0] = 0

              gt_explanation_copy_binary = turn_binary(gt_explanation_copy,True)

              start_time = startTimer()
              scores = []
              for gt, explanation in zip(gt_explanation_copy_binary, explanations_copy):

                  top = np.ma.array(abs(explanation), mask = gt).sum()
                  if not top:
                      top = 0.0

                  bottom = abs(explanation).sum()

                  if top == 0 and bottom == 0:
                      score = 1.0
                  elif top == 0:
                      score = 0.0
                  else:
                      score = top / bottom

                  scores.append(score)

              result_data_xaimethod_time[-1]['soft_PR_'+version_soft] = endTimer(start_time, num_items)
              result_data_xaimethod[-1]['soft_PR_'+version_soft] = sum(scores) / len(scores)

              checkList(scores)




              # Soft Recall (AttributionLab)
              print("Soft Recall "+version_soft)

              explanations_copy = explanations.astype('float32')
              gt_explanation_copy = gt_explanation.astype('float32')


              if version_soft == "abs":
                  explanations_copy = np.abs(explanations_copy)
                  gt_explanation_copy = np.abs(gt_explanation_copy)
              elif version_soft == "pos":
                  explanations_copy[explanations_copy<0] = 0
                  gt_explanation_copy[gt_explanation_copy<0] = 0
              elif version_soft == "neg":
                  explanations_copy[explanations_copy>0] = 0
                  gt_explanation_copy[gt_explanation_copy>0] = 0


              gt_explanation_copy_binary = turn_binary(gt_explanation_copy,True)

              start_time = startTimer()
              scores = []
              for gt, explanation in zip(gt_explanation_copy_binary, explanations_copy):

                  top = np.ma.array(abs(explanation), mask = gt).sum()
                  if not top:
                      top = 0.0

                  bottom = (1-gt).sum()

                  if top == 0 and bottom == 0:
                      score = 1.0
                  elif top == 0:
                      score = 0.0
                  else:
                      score = top / bottom

                  scores.append(score)

              result_data_xaimethod_time[-1]['soft_RE_'+version_soft] = endTimer(start_time, num_items)
              result_data_xaimethod[-1]['soft_RE_'+version_soft] = sum(scores) / len(scores)

              checkList(scores)





              # Soft F1 (AttributionLab)
              print("Soft F1 "+version_soft)

              explanations_copy = explanations.astype('float32')
              gt_explanation_copy = gt_explanation.astype('float32')


              if version_soft == "abs":
                  explanations_copy = np.abs(explanations_copy)
                  gt_explanation_copy = np.abs(gt_explanation_copy)
              elif version_soft == "pos":
                  explanations_copy[explanations_copy<0] = 0
                  gt_explanation_copy[gt_explanation_copy<0] = 0
              elif version_soft == "neg":
                  explanations_copy[explanations_copy>0] = 0
                  gt_explanation_copy[gt_explanation_copy>0] = 0


              gt_explanation_copy_binary = turn_binary(gt_explanation_copy,True)

              start_time = startTimer()
              scores = []
              for gt, explanation in zip(gt_explanation_copy_binary, explanations_copy):


                  # soft precision

                  top = np.ma.array(abs(explanation), mask = gt).sum()
                  if not top:
                      top = 0.0

                  bottom = abs(explanation).sum()

                  if top == 0 and bottom == 0:
                      score_softPR = 1.0
                  elif top == 0:
                      score_softPR = 0.0
                  else:
                      score_softPR = top / bottom



                  # soft recall

                  bottom = (1-gt).sum()

                  if top == 0 and bottom == 0:
                      score_softRE = 1.0
                  elif top == 0:
                      score_softRE = 0.0
                  else:
                      score_softRE = top / bottom



                  #soft F1
                  top = (score_softPR * score_softRE)
                  bottom = (score_softPR + score_softRE)


                  if top == 0 and bottom == 0:
                      score = 0.0
                  elif top == 0:
                      score = 0.0
                  else:
                      score = 2 * top / bottom


                  scores.append(score)

              result_data_xaimethod_time[-1]['soft_F1_'+version_soft] = endTimer(start_time, num_items)
              result_data_xaimethod[-1]['soft_F1_'+version_soft] = sum(scores) / len(scores)

              checkList(scores)


          for version_soft in ["abs","pos","neg"]:


              # Soft Precision (AttributionLab)

              explanations_copy = explanations.astype('float32')
              gt_explanation_copy = gt_explanation.astype('float32')


              if version_soft == "abs":
                  explanations_copy = np.abs(explanations_copy)
                  gt_explanation_copy = np.abs(gt_explanation_copy)
              elif version_soft == "pos":
                  explanations_copy[explanations_copy<0] = 0
                  gt_explanation_copy[gt_explanation_copy<0] = 0
              elif version_soft == "neg":
                  explanations_copy[explanations_copy>0] = 0
                  gt_explanation_copy[gt_explanation_copy>0] = 0


              start_time = startTimer()
              scores = []
              for gt, explanation in zip(gt_explanation_copy, explanations_copy):

                  top = (abs(explanation * gt)).sum()
                  if not top:
                      top = 0.0

                  bottom = (abs(explanation)).sum()

                  if top == 0 and bottom == 0:
                      score = 1.0
                  elif top == 0:
                      score = 0.0
                  else:
                      score = top / bottom

                  scores.append(score)

              result_data_xaimethod_time[-1]['_soft_PR_'+version_soft] = endTimer(start_time, num_items)
              result_data_xaimethod[-1]['_soft_PR_'+version_soft] = sum(scores) / len(scores)

              checkList(scores)




              # Soft Recall (AttributionLab)

              explanations_copy = explanations.astype('float32')
              gt_explanation_copy = gt_explanation.astype('float32')


              if version_soft == "abs":
                  explanations_copy = np.abs(explanations_copy)
                  gt_explanation_copy = np.abs(gt_explanation_copy)
              elif version_soft == "pos":
                  explanations_copy[explanations_copy<0] = 0
                  gt_explanation_copy[gt_explanation_copy<0] = 0
              elif version_soft == "neg":
                  explanations_copy[explanations_copy>0] = 0
                  gt_explanation_copy[gt_explanation_copy>0] = 0

              start_time = startTimer()
              scores = []
              for gt, explanation in zip(gt_explanation_copy, explanations_copy):

                  top = (abs(explanation * gt)).sum()
                  if not top:
                      top = 0.0

                  bottom = (abs(gt)).sum()

                  if top == 0 and bottom == 0:
                      score = 1.0
                  elif top == 0:
                      score = 0.0
                  else:
                      score = top / bottom

                  scores.append(score)

              result_data_xaimethod_time[-1]['_soft_RE_'+version_soft] = endTimer(start_time, num_items)
              result_data_xaimethod[-1]['_soft_RE_'+version_soft] = sum(scores) / len(scores)

              checkList(scores)





              # Soft F1 (AttributionLab)

              explanations_copy = explanations.astype('float32')
              gt_explanation_copy = gt_explanation.astype('float32')


              if version_soft == "abs":
                  explanations_copy = np.abs(explanations_copy)
                  gt_explanation_copy = np.abs(gt_explanation_copy)
              elif version_soft == "pos":
                  explanations_copy[explanations_copy<0] = 0
                  gt_explanation_copy[gt_explanation_copy<0] = 0
              elif version_soft == "neg":
                  explanations_copy[explanations_copy>0] = 0
                  gt_explanation_copy[gt_explanation_copy>0] = 0


              start_time = startTimer()
              scores = []
              for gt, explanation in zip(gt_explanation_copy, explanations_copy):


                  # soft precision

                  top = (abs(explanation * gt)).sum()
                  if not top:
                      top = 0.0

                  bottom = (abs(explanation)).sum()

                  if top == 0 and bottom == 0:
                      score_softPR = 1.0
                  elif top == 0:
                      score_softPR = 0.0
                  else:
                      score_softPR = top / bottom



                  # soft recall

                  bottom = (abs(gt)).sum()

                  if top == 0 and bottom == 0:
                      score_softRE = 1.0
                  elif top == 0:
                      score_softRE = 0.0
                  else:
                      score_softRE = top / bottom



                  #soft F1
                  top = (score_softPR * score_softRE)
                  bottom = (score_softPR + score_softRE)


                  if top == 0 and bottom == 0:
                      score = 0.0
                  elif top == 0:
                      score = 0.0
                  else:
                      score = 2 * top / bottom


                  scores.append(score)

              result_data_xaimethod_time[-1]['_soft_F1_'+version_soft] = endTimer(start_time, num_items)
              result_data_xaimethod[-1]['_soft_F1_'+version_soft] = sum(scores) / len(scores)

              checkList(scores)






          # Attention Percentage Contributing (Do feature attribution methods correctly attribute features)
          print("Attention Percentage Contributing")

          explanations_copy = explanations.astype('float32')
          gt_explanation_copy = gt_explanation.astype('float32')

          gt_explanation_copy_binary = turn_binary(gt_explanation_copy,True) # False = contributing

          start_time = startTimer()
          scores = []
          for gt, explanation in zip(gt_explanation_copy_binary, explanations_copy):

              top = np.ma.array(abs(explanation), mask = gt).sum()
              if not top:
                  top = 0.0

              bottom = abs(explanation).sum()

              if top == 0 and bottom == 0:
                  score = 1.0
              elif top == 0:
                  score = 0.0
              else:
                  score = top / bottom

              scores.append(score)

          result_data_xaimethod_time[-1]['Atten_C'] = endTimer(start_time, num_items)
          result_data_xaimethod[-1]['Atten_C'] = sum(scores) / len(scores)

          checkList(scores)





          # Attention Percentage Contributing (Do feature attribution methods correctly attribute features)
          print("Attention Percentage Non-Contributing")

          explanations_copy = explanations.astype('float32')
          gt_explanation_copy = gt_explanation.astype('float32')

          gt_explanation_copy_binary = turn_binary(gt_explanation_copy, False) # False = Non-contributing

          start_time = startTimer()
          scores = []
          for gt, explanation in zip(gt_explanation_copy_binary, explanations_copy):

              top = np.ma.array(abs(explanation), mask = gt).sum()
              if not top:
                  top = 0.0

              bottom = abs(explanation).sum()

              if top == 0 and bottom == 0:
                  score = 1.0
              elif top == 0:
                  score = 0.0
              else:
                  score = top / bottom

              scores.append(score)

          result_data_xaimethod_time[-1]['Atten_NC'] = endTimer(start_time, num_items)
          result_data_xaimethod[-1]['Atten_NC'] = sum(scores) / len(scores)

          checkList(scores)




































          # Calc our metrics

          print("our metrics")

          for operator in ["!=",">","<"]:

              explanations_copy = explanations.astype('float32')
              gt_explanation_copy = gt_explanation.astype('float32')


              start_time = startTimer()
              compactnessMetric = Our_CompactnessMetric(gt_explanation_copy, explanations_copy,operator)
              result_data_xaimethod[-1]['cpa'+operator] = compactnessMetric.get_score()
              result_data_xaimethod_time[-1]['cpa'+operator] = endTimer(start_time, num_items)

              checkList(compactnessMetric._scores)


              explanations_copy = explanations.astype('float32')
              gt_explanation_copy = gt_explanation.astype('float32')

              start_time = startTimer()
              completenessMetric = Our_CompletenessMetric( gt_explanation_copy,  explanations_copy,operator)
              result_data_xaimethod[-1]['cpl'+operator] = completenessMetric.get_score()
              result_data_xaimethod_time[-1]['cpl'+operator] = endTimer(start_time, num_items)

              checkList(completenessMetric._scores)


              explanations_copy = explanations.astype('float32')
              gt_explanation_copy = gt_explanation.astype('float32')

              start_time = startTimer()
              compactnessMetric = Our_CompactnessMetric(gt_explanation_copy, explanations_copy,operator)
              completenessMetric = Our_CompletenessMetric( gt_explanation_copy,  explanations_copy,operator)
              correctnessMetric = Our_CorrectnessMetric(completenessMetric,compactnessMetric)
              result_data_xaimethod[-1]['cor'+operator] = correctnessMetric.get_score()
              result_data_xaimethod_time[-1]['cor'+operator] = endTimer(start_time, num_items)

              checkList(compactnessMetric._scores)
              checkList(completenessMetric._scores)


          #cor_nosign

          explanations_copy = explanations.astype('float32')
          gt_explanation_copy = gt_explanation.astype('float32')

          start_time = startTimer()
          compactnessMetric = Our_CompactnessMetric( gt_explanation_copy,  explanations_copy,"!=")
          completenessMetric = Our_CompletenessMetric( gt_explanation_copy,  explanations_copy,"!=")
          correctnessMetric = Our_CorrectnessMetric(completenessMetric,compactnessMetric)
          result_data_xaimethod[-1]['cor_nosign'] = correctnessMetric.get_score()
          result_data_xaimethod_time[-1]['cor_nosign'] = endTimer(start_time, num_items)

          checkList(compactnessMetric._scores)
          checkList(completenessMetric._scores)


          #cor_sign

          explanations_copy = explanations.astype('float32')
          gt_explanation_copy = gt_explanation.astype('float32')

          start_time = startTimer()
          compactnessMetric = Our_CompactnessMetric(gt_explanation_copy,  explanations_copy,">")
          completenessMetric = Our_CompletenessMetric(gt_explanation_copy,  explanations_copy,">")
          correctnessMetric = Our_CorrectnessMetric(completenessMetric,compactnessMetric)
          completenessscore_greater = correctnessMetric.get_score()

          compactnessMetric2 = Our_CompactnessMetric(gt_explanation_copy,  explanations_copy,"<")
          completenessMetric2 = Our_CompletenessMetric(gt_explanation_copy,  explanations_copy,"<")
          correctnessMetric2 = Our_CorrectnessMetric(completenessMetric2,compactnessMetric2)
          completenessscore_less = correctnessMetric2.get_score()

          result_data_xaimethod[-1]['cor_sign'] = (completenessscore_greater+completenessscore_less)/2.0
          result_data_xaimethod_time[-1]['cor_sign'] = endTimer(start_time, num_items)

          checkList(compactnessMetric._scores)
          checkList(completenessMetric._scores)

          checkList(compactnessMetric2._scores)
          checkList(completenessMetric2._scores)

          # Time metric

          result_data_xaimethod[-1]['Time'] = xai_method_time



        # Save images
        if settings_global['saveSelectedImages']:
            print("saving images...")

            for class_idx in range(8):

              #saveImage("./"+"Natural_Normalized-yes_Cuda-yes/Run "+str(settings_global['run_id'])+"/selected/nocolormap_"+xai_name+"_"+str(class_idx)+".png", explanations[class_idx])
              saveImageColomap("./paper_images/selected/True/"+str(settings_global['run_id'])+"/"+xai_name+"_"+str(class_idx)+".png", explanations[class_idx])
              if xai_name == "GradCAM":
                saveImageColomap("/./paper_images/selected/True/"+str(settings_global['run_id'])+"/gt2d_"+str(class_idx)+".png", gt_explanation[class_idx])
                saveImage("./paper_images/selected/True/"+str(settings_global['run_id'])+"/input_"+str(class_idx)+".png", data_test_x[class_idx]/255.+.5)

            else:
                print("!!! NOT saving all images... !!!")
        else:
            print("!!! NOT saving images... !!!")





        # ---------------------- test ---------------------- #
        if settings['normalize_explanations']:
            for exp in range(len(data_test_y)):
                if np.amax(gt_explanation[exp]) > 1 or np.amin(gt_explanation[exp]) < -1:
                    print("ERROR! np.amax(gt_explanation[exp]) > 1 or np.amin(gt_explanation[exp]) < -1")
                    print(np.amax(gt_explanation[exp]), np.amin(gt_explanation[exp]))
                    #exit()
                if np.amax(gt_explanation[exp]) > 1 or np.amin(gt_explanation[exp]) < -1:
                    print("ERROR! np.amax(gt_explanation[exp]) > 1 or np.amin(gt_explanation[exp]) < -1")
                    print(np.amax(gt_explanation[exp]), np.amin(gt_explanation[exp]))
                    #exit()


        result_data_xaimethod = result_data_xaimethod[0]
        result_data_xaimethod_time = result_data_xaimethod_time[0]


        # Store results

        if not xai_name in result_data:
            result_data[xai_name] = [result_data_xaimethod]
            result_data_time[xai_name] = [result_data_xaimethod_time]

        else:
            result_data[xai_name].append(result_data_xaimethod)
            result_data_time[xai_name].append(result_data_xaimethod_time)

        print("xai_name",xai_name)
        print(result_data_xaimethod)


result_data = {}
result_data_time = {}

all_xai_methods_names_succesful = []

def main():

    # ---------------------- calc_metrics_for_XAI_method ---------------------- #

    for xai_method, xai_name in zip(all_xai_methods, all_xai_methods_names):

        calc_metrics_for_XAI_method(xai_method, xai_name, model_classifier, data, 0)

    # ---------------------- Print and save results ---------------------- #

    for xai_name in result_data.keys():
        result_data[xai_name] = dict_mean(result_data[xai_name])
        result_data_time[xai_name] = dict_mean(result_data_time[xai_name])

    df = pd.DataFrame(data=result_data)
    df.to_csv(r'./table_results_'+str(settings_global['run_id'])+'.csv', index = True)
    print(df.to_latex(index=True))

    print("")
    print("")
    print("")

    df_time = pd.DataFrame(data=result_data_time)
    #df_time = df_time.mean(axis=1)
    df_time.to_csv(r'./table_times_'+str(settings_global['run_id'])+'.csv', index = True)
    print(df_time.to_latex(index=True))

if __name__ == "__main__":
  main()

  !pip list
  !python --version
  !nvidia-smi
  !lscpu
