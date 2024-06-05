from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)

# Load the trained YOLOv8 model
model = YOLO("best.pt")

# Define a dictionary to map class IDs to class names
class_names = {
    0: 'Barbecued red pork in sauce with rice', 1: 'Chicken Rice Curry With Coconut',2: 'Coconut milk-flavored crepes with shrimp and beef',3: 'Crispy Noodles',4: 'Deep Fried Chicken Wing',5: 'Egg Noodle In Chicken Yellow Curry',6: 'Fried spring rolls',7: 'Hue beef rice vermicelli soup',8: 'Pork Sticky Noodles',9: 'Pork with lemon',10: 'Rice crispy pork',11: 'Rice with roast duck',12: 'Small steamed savory rice pancake',13: 'Sour prawn soup',14: 'Steamed rice roll',15: 'Thai papaya salad',16: 'Vermicelli noodles with snails',17: 'Wonton soup',18: 'adobo',19: 'ayam bakar',20: 'babi guling',21: 'bak kut teh',22: 'baked salmon',23: 'ball shaped bun with pork',24: 'bean curd family style',25: 'beef curry',26: 'beef in oyster sauce',27: 'bibimbap',28: 'boned andsliced Hainan-style chicken with marinated rice',29: 'braised pork meat ball with napa cabbage',30: 'brownie',31: 'charcoal-boiled pork neck',32: 'chicken-n-egg on rice',33: 'chinese pumpkin pie',34: 'chow mein',35: 'coconut milk soup',36: 'croissant',37: 'crullers',38: 'curry puff',39: 'eels on rice',40: 'eggplant with garlic sauce',41: 'eight treasure rice',42: 'fried mussel pancakes',43: 'hamburger',44: 'haupia',45: 'hot - sour soup',46: 'hot and sour and fish and vegetable ragout',47: 'jambalaya',48: 'khao soi',49: 'kung pao chicken',50: 'laulau',51: 'lemon fig jelly',52: 'loco moco',53: 'nasi goreng',54: 'nasi uduk',55: 'noodles with fish curry',56: 'oxtail soup',57: 'pho',58: 'pilaf',59: 'pizza',60: 'pork cutlet on rice',61: 'pork satay',62: 'raisin bread',63: 'ramen noodle',64: 'rice',65: 'salt - pepper fried shrimp with shell',66: 'sandwiches',67: 'soba noodle',68: 'spam musubi',69: 'spicy chicken salad',70: 'steamed spareribs',71: 'stewed pork leg',72: 'stinky tofu',73: 'stir-fried mixed vegetables',74: 'sushi',75: 'tempura bowl',76: 'tempura udon',77: 'three cup chicken',78: 'toast',79: 'udon noodle',80: 'winter melon soup',81: 'zha jiang mian',82: 'Caesar salad',83: 'apple pie',84: 'cream puff',85: 'doughnut',86: 'fried pork dumplings served in soup',87: 'lasagna',88: 'muffin',89: 'oatmeal',90: 'oshiruko',91: 'parfait',92: 'popcorn',93: 'rice gratin',94: 'Chinese soup',95: 'French fries',96: 'Japanese tofu and vegetable chowder',97: 'Okinawa soba',98: 'almond jelly',99: 'bagel',100: 'beef bowl',101: 'beef noodle soup',102: 'broiled eel bowl',103: 'champon',104: 'chicken cutlet',105: 'chicken nugget',106: 'chop suey',107: 'clear soup',108: 'crape',109: 'custard tart',110: 'cutlet curry',111: 'dak galbi',112: 'dipping noodles',113: 'dish consisting of stir-fried potato andeggplant and green pepper',114: 'dry curry',115: 'fine white noodles',116: 'fish ball soup',117: 'fish-shaped pancake with bean jam',118: 'french bread',119: 'french toast',120: 'fried pork in scoop',121: 'fried shrimp',122: 'goya chanpuru',123: 'green curry',124: 'green salad',125: 'ham cutlet',126: 'hot dog',127: 'hot pot',128: 'inarizushi',129: 'jjigae',130: 'kamameshi',131: 'kinpira-style sauteed burdock',132: 'kushikatu',133: 'lamb kebabs',134: 'macaroni salad',135: 'mango pudding',136: 'meat loaf',137: 'minced meat cutlet',138: 'minced pork rice',139: 'minestrone',140: 'moon cake',141: 'mozuku',142: 'mushroom risotto',143: 'nachos',144: 'namero',145: 'omelet with fried rice',146: 'oyster omelette',147: 'paella',148: 'pancake',149: 'pizza toast',150: 'pork belly',151: 'pork cutlet',152: 'pork fillet cutlet',153: 'pork loin cutlet',154: 'pork miso soup',155: 'pot au feu',156: 'potato salad',157: 'rare cheese cake',158: 'rice ball',159: 'rice gruel',160: 'rice vermicelli',161: 'roast chicken',162: 'roast duck',163: 'samul',164: 'scone',165: 'scrambled egg',166: 'shortcake',167: 'shrimp with chill source',168: 'spaghetti meat sauce',169: 'steamed meat dumpling',170: 'tacos',171: 'tanmen',172: 'thinly sliced raw horsemeat',173: 'tiramisu',174: 'tortilla',175: 'turnip pudding',176: 'twice cooked pork',177: 'waffle',178: 'xiao long bao',179: 'yellow cury',180: 'yudofu',181: 'zoni'
}

@app.route('/')
def home():
    return render_template('index.html')

import logging

logging.basicConfig(level=logging.INFO)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image = Image.open(file.stream)
    except Exception as e:
        return jsonify({'error': f'Error opening image: {str(e)}'}), 400

    try:
        # Perform inference
        results = model(image)

        # Log the structure of the results object
        logging.info("Results: %s", results)

        predictions = []

        if isinstance(results, list) and len(results) > 0:
            result = results[0]
            logging.info("Result: %s", result)
            logging.info("Type of result: %s", type(result))

            if hasattr(result, 'boxes'):
                boxes = result.boxes
                logging.info("Boxes: %s", boxes)

                xywh = boxes.xywh
                conf = boxes.conf
                cls = boxes.cls

                for i in range(len(xywh)):
                    box = xywh[i]
                    class_id = cls[i].item()
                    class_name = class_names.get(class_id, f'Unknown class {class_id}')
                    predictions.append({
                        'x': box[0].item(),
                        'y': box[1].item(),
                        'w': box[2].item(),
                        'h': box[3].item(),
                        'confidence': conf[i].item(),
                        'class': class_name
                    })
            else:
                logging.info("Result does not have 'boxes' attribute")
                return jsonify({'error': 'Unexpected result format: no boxes attribute'}), 500

        else:
            logging.info("Results list is empty or not as expected")
            return jsonify({'error': 'Unexpected result format'}), 500
    except Exception as e:
        logging.exception("Error during model inference")
        return jsonify({'error': f'Error during model inference: {str(e)}'}), 500

    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(debug=True)
