import json
from pathlib import Path
from tqdm import tqdm


'''

COCO_CLASSES = [
    ["man","woman","person"], ["bicycle"], ["car"], ["motorcycle"], ["plane","jet"],
    ["buses","the bus","city bus"], ["train"], ["truck"], ["boat"], ["traffic light"],
    ["fire hydrant"], ["stop sign"], ["parking meter"], ["bench"],
    ["bird"], ["kitten","kitty"], ["brown dog","the dog"], ["horse"], ["sheep"], ["bull"],
    ["elephants","elephant"], ["bear"], ["zebra"], ["giraffe"], ["backpack"],
    ["the umbrella"], ["handbag"], ["necktie","neck tie","tie"], ["suitcase"], ["frisbee"],
    ["skis"], ["board","snow board","snowboard"], ["sports ball"], ["kite"], ["baseball bat"],
    ["baseball glove"], ["skate board"], ["surfboard"], ["tennis racket"],
    ["bottle"], ["wine glass"], ["cup","mug","glass"], ["fork"], ["knife","knives","butter knife"],
    ["spoon"], ["bowl"], ["banana"], ["apple"], ["sandwich"],
    ["orange"], ["broccoli"], ["carrot"], ["hot dog"], ["pizza"],
    ["donut"], ["birthday cake"], ["chair"], ["sofa"], ["potted plant"],
    ["bed"], ["dining table"], ["toilet"], ["tv"], ["laptop"],
    ["mouse"], ["remote"], ["keyboard","a black keyboard"], ["cell phone"], ["microwave"],
    ["oven"], ["toaster"], ["bathroom sink","the sink","sink"], ["refrigerator"], ["book"],
    ["clock"], ["vase"], ["scissor","shears","pair of scissors"], ["teddy bear"], ["hair drier"],
    ["toothbrush"],
    ["window"],["tree"],["building"],["sky"],["shirt"],["wall"],["ground"],["sign"],["grass"],
    ["water"],["pole"],["head"],["car"],["light"],["hand"],["plate"],["hair"],["leg"],["clouds"],
    ["road"],["fence"],["ear"],["floor"],["door"],["pants"],["jacket"],["shadow"],["shoe"],["nose"],
    ["sidewalk"],["leaf"],["arm"],["bag"],["windows"],["rock"],["tile"],["post"],
    ["logo"],["mirror"],["stripe"],["roof"],["picture"],["box"],["pillow"],["laptop"],
    ["sand"],["brick"],["ocean"],["hill"],["fork"]
]

# 用户指定的COCO ID列表
COCO_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
    91, 92, 93, 94, 95, 96, 97, 98, 99,
    100, 101, 102, 103, 104, 105, 106, 107,108,109,
    110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
    120, 121, 122, 123, 124, 125, 126, 127,
    128, 129, 130, 131, 132, 133, 134, 135,
    136, 137, 138, 139, 140
]
'''
COCO_CLASSES = [
    ["man","woman","person"], ["bicycle"], ["car"], ["motorcycle"], ["plane","jet"],
    ["buses","the bus","city bus"], ["train"], ["truck"], ["boat"], ["traffic light"],
    ["fire hydrant"], ["stop sign"], ["parking meter"], ["bench"],
    ["bird"], ["kitten","kitty"], ["brown dog","the dog"], ["horse"], ["sheep"], ["bull"],
    ["elephants","elephant"], ["bear"], ["zebra"], ["giraffe"], ["backpack"],
    ["the umbrella"], ["handbag"], ["necktie","neck tie","tie"], ["suitcase"], ["frisbee"],
    ["skis"], ["board","snow board","snowboard"], ["sports ball"], ["kite"], ["baseball bat"],
    ["baseball glove"], ["skate board"], ["surfboard"], ["tennis racket"],
    ["bottle"], ["wine glass"], ["cup","mug","glass"], ["fork"], ["knife","knives","butter knife"],
    ["spoon"], ["bowl"], ["banana"], ["apple"], ["sandwich"],
    ["orange"], ["broccoli"], ["carrot"], ["hot dog"], ["pizza"],
    ["donut"], ["birthday cake"], ["chair"], ["sofa"], ["potted plant"],
    ["bed"], ["dining table"], ["toilet"], ["tv"], ["laptop"],
    ["mouse"], ["remote"], ["keyboard","a black keyboard"], ["cell phone"], ["microwave"],
    ["oven"], ["toaster"], ["bathroom sink","the sink","sink"], ["refrigerator"], ["book"],
    ["clock"], ["vase"], ["scissor","shears","pair of scissors"], ["teddy bear"], ["hair drier"],["toothbrush"],
    ["aerosol_can"], ["air_conditioner"], ["alarm_clock"],
    ["alcohol"], ["alligator"], ["almond"], ["ambulance"], ["amplifier"], ["anklet"],
    ["antenna"], ["apple"], ["applesauce"], ["apricot"], ["apron"], ["aquarium"],
    ["arctic"], ["armband"], ["armoire"], ["armor"],
    ["artichoke"], ["ashtray"], ["asparagus"], ["atomizer"],
    ["avocado"], ["award"], ["awning"], ["ax"], ["baby_buggy"],
    ["basketball_backboard"], ["backpack"], ["handbag"], ["suitcase"], ["bagel"],
    ["bagpipe"], ["baguet"], ["bait"], ["ball"], ["ballet_skirt"], ["balloon"],
    ["bamboo"], ["banana"], ["Band_Aid"], ["bandage"], ["bandanna"], ["banjo"],
    ["banner"], ["barbell"], ["barge"], ["barrette"], ["barrow"],
    ["baseball_base"], ["baseball"], ["baseball_bat"], ["baseball_cap"],
    ["baseball_glove"], ["basket"], ["basketball"], ["bass_horn"], ["bat"],
    ["bath_mat"], ["bath_towel"], ["bathrobe"], ["bathtub"], ["batter"],
    ["battery"], ["beachball"], ["bead"], ["bean_curd"], ["beanbag"], ["beanie"],
    ["bear"], ["bed"], ["bedpan"], ["bedspread"], ["cow"], ["beef"], ["beeper"],
    ["beer_bottle"], ["beer_can"], ["beetle"], ["bell"], ["bell_pepper"], ["belt"],
    ["belt_buckle"], ["bench"], ["beret"], ["bib"], ["Bible"], ["bicycle"], ["visor"],
    ["billboard"], ["binder"], ["binoculars"], ["bird"], ["birdfeeder"], ["birdbath"],
    ["birdcage"], ["birdhouse"], ["birthday_cake"], ["birthday_card"],
    ["pirate_flag"], ["black_sheep"], ["blackberry"], ["blackboard"], ["blanket"],
    ["blazer"], ["blender"], ["blimp"], ["blinker"], ["blouse"], ["blueberry"],
    ["gameboard"], ["boat"], ["bob"], ["bobbin"], ["bobby_pin"], ["boiled_egg"],
    ["bolo_tie"], ["deadbolt"], ["bolt"], ["bonnet"], ["book"], ["bookcase"],
    ["booklet"], ["bookmark"], ["boom_microphone"], ["boot"], ["bottle"],
    ["bottle_opener"], ["bouquet"], ["bow"],
    ["bow"], ["bow-tie"], ["bowl"], ["pipe_bowl"],
    ["bowler_hat"], ["bowling_ball"], ["box"], ["boxing_glove"], ["suspenders"],
    ["bracelet"], ["brass_plaque"], ["brassiere"], ["bread-bin"], ["bread"],
    ["breechcloth"], ["bridal_gown"], ["briefcase"], ["broccoli"], ["broach"],
    ["broom"], ["brownie"], ["brussels_sprouts"], ["bubble_gum"], ["bucket"],
    ["horse_buggy"], ["bull"], ["bulldog"], ["bulldozer"], ["bullet_train"],
    ["bulletin_board"], ["bulletproof_vest"], ["bullhorn"], ["bun"], ["bunk_bed"],
    ["buoy"], ["burrito"], ["bus"], ["business_card"], ["butter"],
    ["butterfly"], ["button"], ["cab"], ["cabana"], ["cabin_car"], ["cabinet"],
    ["locker"], ["cake"], ["calculator"], ["calendar"], ["calf"], ["camcorder"],
    ["camel"], ["camera"], ["camera_lens"], ["camper"], ["can"],
    ["can_opener"], ["candle"], ["candle_holder"], ["candy_bar"], ["candy_cane"],
    ["walking_cane"], ["canister"], ["canoe"], ["cantaloup"], ["canteen"],
    ["cap"], ["bottle_cap"], ["cape"], ["cappuccino"],
    ["car"], ["railcar"], ["elevator_car"],
    ["car_battery"], ["identity_card"], ["card"], ["cardigan"], ["cargo_ship"],
    ["carnation"], ["horse_carriage"], ["carrot"], ["tote_bag"], ["cart"], ["carton"],
    ["cash_register"], ["casserole"], ["cassette"], ["cast"], ["cat"],
    ["cauliflower"], ["cayenne"], ["CD_player"], ["celery"],
    ["cellular_telephone"], ["chain_mail"], ["chair"], ["chaise_longue"],
    ["chalice"], ["chandelier"], ["chap"], ["checkbook"], ["checkerboard"],
    ["cherry"], ["chessboard"], ["chicken"], ["chickpea"],
    ["chili"], ["chime"], ["chinaware"], ["crisp"],
    ["poker_chip"], ["chocolate_bar"], ["chocolate_cake"], ["chocolate_milk"],
    ["chocolate_mousse"], ["choker"], ["chopping_board"], ["chopstick"],
    ["Christmas_tree"], ["slide"], ["cider"], ["cigar_box"], ["cigarette"],
    ["cigarette_case"], ["cistern"], ["clarinet"], ["clasp"], ["cleansing_agent"],
    ["cleat"], ["clementine"], ["clip"], ["clipboard"],
    ["clippers"], ["cloak"], ["clock"], ["clock_tower"],
    ["clothes_hamper"], ["clothespin"], ["clutch_bag"], ["coaster"], ["coat"],
    ["coat_hanger"], ["coatrack"], ["cock"], ["cockroach"], ["cocoa"],
    ["coconut"], ["coffee_maker"], ["coffee_table"], ["coffeepot"], ["coil"],
    ["coin"], ["colander"], ["coleslaw"], ["coloring_material"],
    ["combination_lock"], ["pacifier"], ["comic_book"], ["compass"],
    ["computer_keyboard"], ["condiment"], ["cone"], ["control"],
    ["convertible"], ["sofa_bed"], ["cooker"], ["cookie"],
    ["cooking_utensil"], ["cooler"], ["cork"],
    ["corkboard"], ["corkscrew"], ["edible_corn"], ["cornbread"], ["cornet"],
    ["cornice"], ["cornmeal"], ["corset"], ["costume"], ["cougar"], ["coverall"],
    ["cowbell"], ["cowboy_hat"], ["crab"], ["crabmeat"], ["cracker"],
    ["crape"], ["crate"], ["crayon"], ["cream_pitcher"], ["crescent_roll"], ["crib"],
    ["crock_pot"], ["crossbar"], ["crouton"], ["crow"], ["crowbar"], ["crown"],
    ["crucifix"], ["cruise_ship"], ["police_cruiser"], ["crumb"], ["crutch"],
    ["cub"], ["cube"], ["cucumber"], ["cufflink"], ["cup"], ["trophy_cup"],
    ["cupboard"], ["cupcake"], ["hair_curler"], ["curling_iron"], ["curtain"],
    ["cushion"], ["cylinder"], ["cymbal"], ["dagger"], ["dalmatian"], ["dartboard"],
    ["date"], ["deck_chair"], ["deer"], ["dental_floss"], ["desk"],
    ["detergent"], ["diaper"], ["diary"], ["die"], ["dinghy"], ["dining_table"],
    ["tux"], ["dish"], ["dish_antenna"], ["dishrag"], ["dishtowel"], ["dishwasher"],
    ["dishwasher_detergent"], ["dispenser"], ["diving_board"], ["Dixie_cup"],
    ["dog"], ["dog_collar"], ["doll"], ["dollar"], ["dollhouse"], ["dolphin"],
    ["domestic_ass"], ["doorknob"], ["doormat"], ["doughnut"], ["dove"],
    ["dragonfly"], ["drawer"], ["underdrawers"], ["dress"], ["dress_hat"],
    ["dress_suit"], ["dresser"], ["drill"], ["drone"], ["dropper"],
    ["drum"], ["drumstick"], ["duck"], ["duckling"],
    ["duct_tape"], ["duffel_bag"], ["dumbbell"], ["dumpster"], ["dustpan"], ["eagle"],
    ["earphone"], ["earplug"], ["earring"], ["easel"], ["eclair"], ["eel"], ["egg"],
    ["egg_roll"], ["egg_yolk"], ["eggbeater"], ["eggplant"], ["electric_chair"],
    ["refrigerator"], ["elephant"], ["elk"], ["envelope"], ["eraser"], ["escargot"],
    ["eyepatch"], ["falcon"], ["fan"], ["faucet"], ["fedora"], ["ferret"],
    ["Ferris_wheel"], ["ferry"], ["fig"], ["fighter_jet"], ["figurine"],
    ["file_cabinet"], ["file"], ["fire_alarm"], ["fire_engine"],
    ["fire_extinguisher"], ["fire_hose"], ["fireplace"], ["fireplug"],
    ["first-aid_kit"], ["fish"], ["fish"], ["fishbowl"], ["fishing_rod"],
    ["flag"], ["flagpole"], ["flamingo"], ["flannel"], ["flap"], ["flash"],
    ["flashlight"], ["fleece"], ["flip-flop"], ["flipper"],
    ["flower_arrangement"], ["flute_glass"], ["foal"], ["folding_chair"],
    ["food_processor"], ["football"], ["football_helmet"],
    ["footstool"], ["fork"], ["forklift"], ["freight_car"], ["French_toast"],
    ["freshener"], ["frisbee"], ["frog"], ["fruit_juice"], ["frying_pan"], ["fudge"],
    ["funnel"], ["futon"], ["gag"], ["garbage"], ["garbage_truck"], ["garden_hose"],
    ["gargle"], ["gargoyle"], ["garlic"], ["gasmask"], ["gazelle"], ["gelatin"],
    ["gemstone"], ["generator"], ["giant_panda"], ["gift_wrap"], ["ginger"],
    ["giraffe"], ["cincture"], ["glass"], ["globe"], ["glove"],
    ["goat"], ["goggles"], ["goldfish"], ["golf_club"], ["golfcart"],
    ["gondola"], ["goose"], ["gorilla"], ["gourd"], ["grape"], ["grater"],
    ["gravestone"], ["gravy_boat"], ["green_bean"], ["green_onion"], ["griddle"],
    ["grill"], ["grits"], ["grizzly"], ["grocery_bag"], ["guitar"], ["gull"], ["gun"],
    ["hairbrush"], ["hairnet"], ["hairpin"], ["halter_top"], ["ham"], ["hamburger"],
    ["hammer"], ["hammock"], ["hamper"], ["hamster"], ["hair_dryer"], ["hand_glass"],
    ["hand_towel"], ["handcart"], ["handcuff"], ["handkerchief"], ["handle"],
    ["handsaw"], ["hardback_book"], ["harmonium"], ["hat"], ["hatbox"], ["veil"],
    ["headband"], ["headboard"], ["headlight"], ["headscarf"], ["headset"],
    ["headstall"], ["heart"], ["heater"], ["helicopter"], ["helmet"],
    ["heron"], ["highchair"], ["hinge"], ["hippopotamus"], ["hockey_stick"], ["hog"],
    ["home_plate"], ["honey"], ["fume_hood"], ["hook"], ["hookah"],
    ["hornet"], ["horse"], ["hose"], ["hot-air_balloon"], ["hotplate"], ["hot_sauce"],
    ["hourglass"], ["houseboat"], ["hummingbird"], ["hummus"], ["polar_bear"],
    ["icecream"], ["popsicle"], ["ice_maker"], ["ice_pack"], ["ice_skate"],
    ["igniter"], ["inhaler"], ["iPod"], ["iron"], ["ironing_board"],
    ["jacket"], ["jam"], ["jar"], ["jean"], ["jeep"], ["jelly_bean"], ["jersey"],
    ["jet_plane"], ["jewel"], ["jewelry"], ["joystick"], ["jumpsuit"], ["kayak"],
    ["keg"], ["kennel"], ["kettle"], ["key"], ["keycard"], ["kilt"], ["kimono"],
    ["kitchen_sink"], ["kitchen_table"], ["kite"], ["kitten"], ["kiwi_fruit"],
    ["knee_pad"], ["knife"], ["knitting_needle"], ["knob"], ["knocker"],
    ["koala"], ["lab_coat"], ["ladder"], ["ladle"], ["ladybug"], ["lamb"],
    ["lamb-chop"], ["lamp"], ["lamppost"], ["lampshade"], ["lantern"], ["lanyard"],
    ["laptop_computer"], ["lasagna"], ["latch"], ["lawn_mower"], ["leather"],
    ["legging"], ["Lego"], ["legume"], ["lemon"], ["lemonade"],
    ["lettuce"], ["license_plate"], ["life_buoy"], ["life_jacket"], ["lightbulb"],
    ["lightning_rod"], ["lime"], ["limousine"], ["lion"], ["lip_balm"], ["liquor"],
    ["lizard"], ["log"], ["lollipop"], ["speaker"], ["loveseat"],
    ["machine_gun"], ["magazine"], ["magnet"], ["mail_slot"], ["mailbox"],
    ["mallard"], ["mallet"], ["mammoth"], ["manatee"], ["mandarin_orange"],
    ["manger"], ["manhole"], ["map"], ["marker"], ["martini"], ["mascot"],
    ["mashed_potato"], ["masher"], ["mask"], ["mast"], ["mat"],
    ["matchbox"], ["mattress"], ["measuring_cup"], ["measuring_stick"],
    ["meatball"], ["medicine"], ["melon"], ["microphone"], ["microscope"],
    ["microwave_oven"], ["milestone"], ["milk"], ["milk_can"], ["milkshake"],
    ["minivan"], ["mint_candy"], ["mirror"], ["mitten"], ["mixer"],
    ["money"], ["monitor"], ["monkey"],
    ["motor"], ["motor_scooter"], ["motor_vehicle"], ["motorcycle"],
    ["mound"], ["mouse"], ["mousepad"],
    ["muffin"], ["mug"], ["mushroom"], ["music_stool"], ["musical_instrument"],
    ["nailfile"], ["napkin"], ["neckerchief"], ["necklace"], ["necktie"], ["needle"],
    ["nest"], ["newspaper"], ["newsstand"], ["nightshirt"],
    ["nosebag"], ["noseband"], ["notebook"],
    ["notepad"], ["nut"], ["nutcracker"], ["oar"], ["octopus"],
    ["octopus"], ["oil_lamp"], ["olive_oil"], ["omelet"], ["onion"],
    ["orange"], ["orange_juice"], ["ostrich"], ["ottoman"], ["oven"],
    ["overalls"], ["owl"], ["packet"], ["inkpad"], ["pad"], ["paddle"],
    ["padlock"], ["paintbrush"], ["painting"], ["pajamas"], ["palette"],
    ["pan"], ["pan"], ["pancake"], ["pantyhose"],
    ["papaya"], ["paper_plate"], ["paper_towel"], ["paperback_book"],
    ["paperweight"], ["parachute"], ["parakeet"], ["parasail"],
    ["parasol"], ["parchment"], ["parka"], ["parking_meter"], ["parrot"],
    ["passenger_car"], ["passenger_ship"], ["passport"],
    ["pastry"], ["patty"], ["pea"], ["peach"], ["peanut_butter"],
    ["pear"], ["peeler"], ["wooden_leg"],
    ["pegboard"], ["pelican"], ["pen"], ["pencil"], ["pencil_box"],
    ["pencil_sharpener"], ["pendulum"], ["penguin"], ["pennant"], ["penny"],
    ["pepper"], ["pepper_mill"], ["perfume"], ["persimmon"], ["person"], ["pet"],
    ["pew"], ["phonebook"], ["phonograph_record"], ["piano"],
    ["pickle"], ["pickup_truck"], ["pie"], ["pigeon"], ["piggy_bank"], ["pillow"],
    ["pin"], ["pineapple"], ["pinecone"], ["ping-pong_ball"],
    ["pinwheel"], ["tobacco_pipe"], ["pipe"], ["pistol"], ["pita"],
    ["pitcher"], ["pitchfork"], ["pizza"], ["place_mat"],
    ["plate"], ["platter"], ["playpen"], ["pliers"], ["plow"],
    ["plume"], ["pocket_watch"], ["pocketknife"], ["poker"],
    ["pole"], ["polo_shirt"], ["poncho"], ["pony"], ["pool_table"], ["pop"],
    ["postbox"], ["postcard"], ["poster"], ["pot"], ["flowerpot"],
    ["potato"], ["pothholder"], ["pottery"], ["pouch"], ["power_shovel"], ["prawn"],
    ["pretzel"], ["printer"], ["projectile"], ["projector"], ["propeller"],
    ["prune"], ["pudding"], ["puffer"], ["puffin"], ["pug-dog"], ["pumpkin"],
    ["puncher"], ["puppet"], ["p puppy"], ["quesadilla"], ["quiche"], ["quilt"],
    ["rabbit"], ["race_car"], ["racket"], ["radar"], ["radiator"], ["radio_receiver"],
    ["radish"], ["raft"], ["rag_doll"], ["raincoat"], ["ram"], ["raspberry"],
    ["rat"], ["razorblade"], ["reamer"], ["rearview_mirror"], ["receipt"],
    ["recliner"], ["record_player"], ["reflector"], ["remote_control"],
    ["rhinoceros"], ["rib"], ["rifle"], ["ring"], ["river_boat"], ["road_map"],
    ["robe"], ["rocking_chair"], ["rodent"], ["roller_skate"], ["Rollerblade"],
    ["rolling_pin"], ["root_beer"], ["router"],
    ["rubber_band"], ["runner"], ["plastic_bag"],
    ["saddle"], ["saddle_blanket"], ["saddlebag"], ["safety_pin"],
    ["sail"], ["salad"], ["salad_plate"], ["salami"], ["salmon"],
    ["salmon"], ["salsa"], ["saltshaker"], ["sandal"],
    ["sandwich"], ["satchel"], ["saucepan"], ["saucer"], ["sausage"], ["sawhorse"],
    ["saxophone"], ["scale"], ["scarecrow"], ["scarf"],
    ["school_bus"], ["scissors"], ["scoreboard"], ["scraper"], ["s screwdriver"],
    ["scrubbing_brush"], ["sculpture"], ["seabird"], ["seahorse"], ["seaplane"],
    ["seashell"], ["sewing_machine"], ["shaker"], ["shampoo"], ["shark"],
    ["sharpener"], ["Sharpie"], ["shaver"], ["shaving_cream"], ["shawl"],
    ["shears"], ["sheep"], ["shepherd_dog"], ["sherbert"], ["shield"], ["shirt"],
    ["shoe"], ["shopping_bag"], ["shopping_cart"], ["short_pants"], ["shot_glass"],
    ["shoulder_bag"], ["shovel"], ["shower_head"], ["shower_cap"],
    ["shower_curtain"], ["shredder"], ["signboard"], ["silo"], ["sink"],
    ["skateboard"], ["skewer"], ["ski"], ["ski_boot"], ["ski_parka"], ["ski_pole"],
    ["skirt"], ["skullcap"], ["sled"], ["sleeping_bag"], ["sling"],
    ["slipper"], ["smoothie"], ["snake"], ["snowboard"], ["s snowman"],
    ["snowmobile"], ["soap"], ["soccer_ball"], ["sock"], ["sofa"], ["softball"],
    ["solar_array"], ["sombrero"], ["soup"], ["soup_bowl"], ["soupspoon"],
    ["sour_cream"], ["soya_milk"], ["space_shuttle"], ["sparkler"],
    ["spatula"], ["spear"], ["spectacles"], ["spice_rack"], ["spider"], ["crawfish"],
    ["sponge"], ["spoon"], ["sportswear"], ["spotlight"], ["squid"],
    ["squirrel"], ["stagecoach"], ["stapler"], ["starfish"],
    ["statue"], ["steak"], ["steak_knife"], ["steering_wheel"],
    ["stepladder"], ["step_stool"], ["stereo"], ["stew"],
    ["stirrer"], ["stirrup"], ["stool"], ["stop_sign"], ["brake_light"], ["stove"],
    ["st strainer"], ["strap"], ["straw"], ["strawberry"],
    ["street_sign"], ["streetlight"], ["string_cheese"], ["stylus"], ["subwoofer"],
    ["sugar_bowl"], ["sugarcane"], ["suit"], ["sunflower"],
    ["sunglasses"], ["sunhat"], ["surfboard"], ["sushi"], ["mop"], ["sweat_pants"],
    ["sweatband"], ["sweater"], ["sweatshirt"], ["sweet_potato"], ["swimsuit"],
    ["sword"], ["syringe"], ["Tabasco_sauce"], ["table-tennis_table"], ["table"],
    ["table_lamp"], ["tablecloth"], ["tachometer"], ["taco"], ["tag"], ["taillight"],
    ["tambourine"], ["army_tank"], ["tank"],
    ["tank_top"], ["tape"], ["tape_measure"],
    ["tapestry"], ["tarp"], ["tartan"], ["tassel"], ["tea_bag"], ["teacup"],
    ["teakettle"], ["teapot"], ["teddy_bear"], ["telephone"], ["telephone_booth"],
    ["telephone_pole"], ["telephoto_lens"], ["television_camera"],
    ["television_set"], ["tennis_ball"], ["tennis_racket"], ["tequila"],
    ["thermometer"], ["thermos_bottle"], ["thermostat"], ["thimble"], ["thread"],
    ["thumbtack"], ["tiara"], ["tiger"], ["tights"], ["timer"],
    ["tinfoil"], ["tinsel"], ["tissue_paper"], ["toast"], ["toaster"],
    ["toaster_oven"], ["toilet"], ["toilet_tissue"], ["tomato"], ["tongs"],
    ["toolbox"], ["toothbrush"], ["toothpaste"], ["toothpick"], ["cover"],
    ["tortilla"], ["tow_truck"], ["towel"], ["towel_rack"], ["toy"],
    ["tractor"], ["traffic_light"], ["dirt_bike"],
    ["trailer_truck"], ["train"], ["trampoline"], ["tray"],
    ["trench_coat"], ["triangle"], ["tricycle"], ["tripod"],
    ["trousers"], ["truck"], ["truffle"], ["trunk"], ["vat"], ["turban"],
    ["turkey"], ["turnip"], ["turtle"], ["turtleneck"],
    ["typewriter"], ["umbrella"], ["underwear"], ["unicycle"], ["urinal"], ["urn"],
    ["vacuum_cleaner"], ["vase"], ["vending_machine"], ["vent"], ["vest"],
    ["videotape"], ["vinegar"], ["violin"], ["vodka"], ["volleyball"], ["vulture"],
    ["waffle"], ["waffle_iron"], ["wagon"], ["wagon_wheel"], ["walking_stick"],
    ["wall_clock"], ["wall_socket"], ["wallet"], ["walrus"], ["wardrobe"],
    ["washbasin"], ["automatic_washer"], ["watch"], ["water_bottle"],
    ["water_cooler"], ["water_faucet"], ["water_heater"], ["water_jug"],
    ["water_gun"], ["water_scooter"], ["water_ski"], ["water_tower"],
    ["watering_can"], ["watermelon"], ["weathervane"], ["webcam"], ["wedding_cake"],
    ["wedding_ring"], ["wet_suit"], ["wheel"], ["wheelchair"], ["whipped_cream"],
    ["whistle"], ["wig"], ["wind_chime"], ["windmill"], ["window_box"],
    ["windshield_wiper"], ["windsock"], ["wine_bottle"], ["wine_bucket"],
    ["wineglass"], ["blinder"], ["wok"], ["wolf"], ["wooden_spoon"],
    ["wreath"], ["wrench"], ["wristband"], ["wristlet"], ["yacht"], ["yogurt"],
    ["yoke"], ["zebra"], ["zucchini"]

]


COCO_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
91,92,93,94,95,96,97,98,99,100  ,
101,102,103,104,105,106,107,108,109,110  ,
111,112,113,114,115,116,117,118,119,120  ,
121,122,123,124,125,126,127,128,129,130  ,
131,132,133,134,135,136,137,138,139,140  ,
141,142,143,144,145,146,147,148,149,150  ,
151,152,153,154,155,156,157,158,159,160  ,
161,162,163,164,165,166,167,168,169,170  ,
171,172,173,174,175,176,177,178,179,180  ,
181,182,183,184,185,186,187,188,189,190  ,
191,192,193,194,195,196,197,198,199,200  ,
201,202,203,204,205,206,207,208,209,210  ,
211,212,213,214,215,216,217,218,219,220  ,
221,222,223,224,225,226,227,228,229,230  ,
231,232,233,234,235,236,237,238,239,240  ,
241,242,243,244,245,246,247,248,249,250  ,
251,252,253,254,255,256,257,258,259,260  ,
261,262,263,264,265,266,267,268,269,270  ,
271,272,273,274,275,276,277,278,279,280  ,
281,282,283,284,285,286,287,288,289,290  ,
291,292,293,294,295,296,297,298,299,300  ,
301,302,303,304,305,306,307,308,309,310  ,
311,312,313,314,315,316,317,318,319,320  ,
321,322,323,324,325,326,327,328,329,330  ,
331,332,333,334,335,336,337,338,339,340  ,
341,342,343,344,345,346,347,348,349,350  ,
351,352,353,354,355,356,357,358,359,360  ,
361,362,363,364,365,366,367,368,369,370  ,
371,372,373,374,375,376,377,378,379,380  ,
381,382,383,384,385,386,387,388,389,390  ,
391,392,393,394,395,396,397,398,399,400  ,
401,402,403,404,405,406,407,408,409,410  ,
411,412,413,414,415,416,417,418,419,420  ,
421,422,423,424,425,426,427,428,429,430  ,
431,432,433,434,435,436,437,438,439,440  ,
441,442,443,444,445,446,447,448,449,450  ,
451,452,453,454,455,456,457,458,459,460  ,
461,462,463,464,465,466,467,468,469,470  ,
471,472,473,474,475,476,477,478,479,480  ,
481,482,483,484,485,486,487,488,489,490  ,
491,492,493,494,495,496,497,498,499,500  ,
501,502,503,504,505,506,507,508,509,510  ,
511,512,513,514,515,516,517,518,519,520  ,
521,522,523,524,525,526,527,528,529,530 ,
531,532,533,534,535,536,537,538,539,540 ,
541,542,543,544,545,546,547,548,549,550,
551,552,553,554,555,556,557,558,559,560  ,
561,562,563,564,565,566,567,568,569,570 ,
571,572,573,574,575,576,577,578,579,580 ,
581,582,583,584,585,586,587,588,589,590 ,
591,592,593,594,595,596,597,598,599,600,
601,602,603,604,605,606,607,608,609,610,
611,612,613,614,615,616,617,618,619,620,
621,622,623,624,625,626,627,628,629,630,
631,632,633,634,635,636,637,638,639,640,
641,642,643,644,645,646,647,648,649,650 ,
651,652,653,654,655,656,657,658,659,660,
661,662,663,664,665,666,667,668,669,670 ,
671,672,673,674,675,676,677,678,679,680 ,
681,682,683,684,685,686,687,688,689,690 ,
691,692,693,694,695,696,697,698,699,700 ,
701,702,703,704,705,706,707,708,709,710 ,
711,712,713,714,715,716,717,718,719,720,
721,722,723,724,725,726,727,728,729,730 ,
731,732,733,734,735,736,737,738,739,740 ,
741,742,743,744,745,746,747,748,749,750 ,
751,752,753,754,755,756,757,758,759,760 ,
761,762,763,764,765,766,767,768,769,770,
771,772,773,774,775,776,777,778,779,780,
781,782,783,784,785,786,787,788,789,790,
791,792,793,794,795,796,797,798,799,800,
801,802,803,804,805,806,807,808,809,810,
811,812,813,814,815,816,817,818,819,820,
821,822,823,824,825,826,827,828,829,830,
831,832,833,834,835,836,837,838,839,840,
841,842,843,844,845,846,847,848,849,850,
851,852,853,854,855,856,857,858,859,860,
861,862,863,864,865,866,867,868,869,870,
871,872,873,874,875,876,877,878,879,880,
881,882,883,884,885,886,887,888,889,890,
891,892,893,894,895,896,897,898,899,900,
901,902,903,904,905,906,907,908,909,910,
911,912,913,914,915,916,917,918,919,920,
921,922,923,924,925,926,927,928,929,930,
931,932,933,934,935,936,937,938,939,940,
941,942,943,944,945,946,947,948,949,950  ,
951,952,953,954,955,956,957,958,959,960 ,
961,962,963,964,965,966,967,968,969,970,
971,972,973,974,975,976,977,978,979,980,
981,982,983,984,985,986,987,988,989,990,
991,992,993,994,995,996,997,998,999,1000,
1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,
1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,
1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,
1031,1032,1033,1034,1035,1036,1037,1038,1039,1040,
1041,1042,1043,1044,1045,1046,1047,1048,1049,1050,
1051,1052,1053,1054,1055,1056,1057,1058,1059,1060,
1061,1062,1063,1064,1065,1066,1067,1068,1069,1070,
1071,1072,1073,1074,1075,1076,1077,1078,1079,1080,
1081,1082,1083,1084,1085,1086,1087,1088,1089,1090,

]



# 构建多对一映射字典
COCO_CATEGORIES = {}
for class_group, class_id in zip(COCO_CLASSES, COCO_IDS):
    for name in class_group:
        COCO_CATEGORIES[name] = class_id
        COCO_CATEGORIES[name.lower()] = class_id  # 大小写不敏感匹配


def convert_annotation(input_path, image_data_path, output_path):
    """转换标注格式，使用独立图像尺寸文件"""
    # 加载原始标注数据
    with open(input_path) as f:
        original_data = json.load(f)

    # 加载图像尺寸数据
    with open(image_data_path) as f:
        image_data = json.load(f)

    # 构建图像尺寸映射表
    size_map = {img["id"]: (img["width"], img["height"]) for img in image_data}

    # 初始化COCO数据结构
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": class_id,
                "name": class_group[0],
                "synonyms": class_group  # 存储同义词信息
            }
            for class_group, class_id in zip(COCO_CLASSES, COCO_IDS)
        ]
    }

    stats = {
        "total_images": len(original_data),
        "total_objects": 0,
        "valid_objects": 0,
        "skipped_objects": 0,
        "skipped_images": 0
    }

    annotation_id = 1

    # 处理每张图像
    for img in tqdm(original_data, desc="Converting annotations"):
        img_id = img["id"]

        # 获取实际图像尺寸
        width, height = size_map.get(img_id, (0, 0))
        if width == 0 or height == 0:
            print(f"警告: 图像 {img_id} 缺少尺寸信息，使用默认尺寸 (0, 0)")

        # 添加图像元信息
        coco_format["images"].append({
            "id": img_id,
            "width": width,
            "height": height,
            "file_name": f"{img_id}.jpg"
        })

        # 处理标注对象
        has_valid = False
        for obj in img["objects"]:
            stats["total_objects"] += 1

            # 查找匹配的类别ID
            matched_id = None
            for name in obj["names"]:
                if (matched_id := COCO_CATEGORIES.get(name)) or \
                        (matched_id := COCO_CATEGORIES.get(name.lower())):
                    break

            if matched_id:
                # 添加标注信息
                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": matched_id,
                    "bbox": obj.get("bbox", [obj["x"], obj["y"], obj["w"], obj["h"]]),
                    "area": obj.get("area", obj["w"] * obj["h"]),
                    "iscrowd": obj.get("iscrowd", 0),
                    "segmentation": []
                })
                annotation_id += 1
                stats["valid_objects"] += 1
                has_valid = True
            else:
                stats["skipped_objects"] += 1

        if not has_valid:
            stats["skipped_images"] += 1

    # 保存结果文件
    with open(output_path, 'w') as f:
        json.dump(coco_format, f, indent=2)

    # 打印统计信息
    print("\n转换完成，统计结果:")
    print(f"▪ 总图像数: {stats['total_images']}")
    print(f"▪ 有效标注图像: {stats['total_images'] - stats['skipped_images']}")
    print(f"▪ 总对象数: {stats['total_objects']}")
    print(f"▪ 有效对象: {stats['valid_objects']} (保留率: {stats['valid_objects'] / stats['total_objects']:.1%})")
    print(f"▪ 无效对象: {stats['skipped_objects']} (原因: 未匹配到任何类别)")


if __name__ == "__main__":
    # 文件路径配置
    input_json = "/root/autodl-tmp/objects.json"  # 原始标注文件
    image_data_json = "/root/autodl-tmp/image_data.json"  # 图像尺寸文件
    output_json = "/root/autodl-tmp/VGcoco_strict91.json"  # 输出文件

    # 确保输出目录存在
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)

    # 执行转换
    convert_annotation(input_json, image_data_json, output_json)