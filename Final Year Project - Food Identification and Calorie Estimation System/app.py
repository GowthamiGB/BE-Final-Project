import tensorflow
from flask import Flask, request, render_template
import csv
import math
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.models import load_model
from werkzeug.utils import secure_filename

tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=tmpl_dir)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

north_food = [
    'Anarsa', 'Butter_Naan', 'Chana_Masala', 'Chola_Bhatura', 'Dal_Makhani', 'Dhokla', 'Gajar_Halwa',
    'Gulab_Jamun', 'Jalebi', 'Kaathi_Rolls', 'Kadhai_Paneer', 'Kajjikaya', 'lassi', 'paani_puri', 'Paratha',
    'Pav_Bhaji', 'Poha', 'Rasgulla', 'Roti', 'Vada_Pav'
]

south_food = [
    'Appam', 'Biryani', 'bombil_fry', 'Bonda', 'Chikki', 'Dosa', 'Idli', 'Kerala_Beef_Fry', 'medhuvada', 
    'Mirchi_Bajji', 'Murukku', 'Mysore_Pak', 'Neyyapam', 'Paniyaram', 'Pongal', 'Seera', 'unni_appam',
    'Upma', 'Uttapam'
]

label_11 = [
    'Anarsa',
    'Appam',
    'Biryani',
    'Bissibellebaath',
    'Bonda',
    'Butter_Naan',
    'Chana_Masala',
    'Chikki',
    'Chola_Bhatura',
    'Dal_Makhani',
    'Dhokla',
    'Dosa',
    'Gajar_Halwa',
    'Gulab_Jamun',
    'Idli',
    'Jalebi',
    'Kaathi_Rolls',
    'Kadhai_Paneer',
    'Kajjikaya',
    'Kerala_Beef_Fry',
    'Mirchi_Bajji',
    'Murukku',
    'Mysore_Pak',
    'Neyyapam',
    'Paniyaram',
    'Paratha',
    'Pav_Bhaji',
    'Poha',
    'Pongal',
    'Rasgulla',
    'Roti',
    'Seera',
    'Upma',
    'Uttapam',
    'Vada_Pav',
    'bombil_fry',
    'lassi',
    'medhuvada',
    'paani_puri',
    'unni_appam'
]

# label_11 = [
#     'apple pie', 'cheesecake', 'chicken wings', 'churros', 'cup_cakes', 'donuts', 'dumplings',
#     'samosa', 'pizza', 'omelette', 'ramen'
# ]

label_3 = [
    'samosa', 'pizza', 'omelette'
]

# define label meaning
label_all = ['apple pie',
         'baby back ribs',
         'baklava',
         'beef carpaccio',
         'beef tartare',
         'beet salad',
         'beignets',
         'bibimbap',
         'bread pudding',
         'breakfast burrito',
         'bruschetta',
         'caesar salad',
         'cannoli',
         'caprese salad',
         'carrot cake',
         'ceviche',
         'cheese plate',
         'cheesecake',
         'chicken curry',
         'chicken quesadilla',
         'chicken wings',
         'chocolate cake',
         'chocolate mousse',
         'churros',
         'clam chowder',
         'club sandwich',
         'crab cakes',
         'creme brulee',
         'croque madame',
         'cup cakes',
         'deviled eggs',
         'donuts',
         'dumplings',
         'edamame',
         'eggs benedict',
         'escargots',
         'falafel',
         'filet mignon',
         'fish and_chips',
         'foie gras',
         'french fries',
         'french onion soup',
         'french toast',
         'fried calamari',
         'fried rice',
         'frozen yogurt',
         'garlic bread',
         'gnocchi',
         'greek salad',
         'grilled cheese sandwich',
         'grilled salmon',
         'guacamole',
         'gyoza',
         'hamburger',
         'hot and sour soup',
         'hot dog',
         'huevos rancheros',
         'hummus',
         'ice cream',
         'lasagna',
         'lobster bisque',
         'lobster roll sandwich',
         'macaroni and cheese',
         'macarons',
         'miso soup',
         'mussels',
         'nachos',
         'omelette',
         'onion rings',
         'oysters',
         'pad thai',
         'paella',
         'pancakes',
         'panna cotta',
         'peking duck',
         'pho',
         'pizza',
         'pork chop',
         'poutine',
         'prime rib',
         'pulled pork sandwich',
         'ramen',
         'ravioli',
         'red velvet cake',
         'risotto',
         'samosa',
         'sashimi',
         'scallops',
         'seaweed salad',
         'shrimp and grits',
         'spaghetti bolognese',
         'spaghetti carbonara',
         'spring rolls',
         'steak',
         'strawberry shortcake',
         'sushi',
         'tacos',
         'octopus balls',
         'tiramisu',
         'tuna tartare',
         'waffles']
label = label_11.copy()

nu_link = 'https://www.nutritionix.com/food/'

# Loading the best saved model to make predictions.
tensorflow.keras.backend.clear_session()
model_best = tensorflow.keras.models.load_model(
    'model.h5',
    compile=False,
    custom_objects={'Functional':tensorflow.keras.models.Model})
#model_best = load_model('best_model_11class.hdf5', compile=False)
print('model successfully loaded!')

start = [0]
passed = [0]
pack = [[]]
num = [0]

nutrients = [
    {'name': 'protein', 'value': 0.0},
    {'name': 'calcium', 'value': 0.0},
    {'name': 'fat', 'value': 0.0},
    {'name': 'carbohydrates', 'value': 0.0},
    {'name': 'vitamins', 'value': 0.0}
]

with open('nutrition101.csv', 'r') as file:
    reader = csv.reader(file)
    nutrition_table = dict()
    for i, row in enumerate(reader):
        if i == 0:
            name = ''
            continue
        else:
            name = row[1].strip()
        nutrition_table[name] = [
            {'name': 'protein', 'value': float(row[2])},
            {'name': 'calcium', 'value': float(row[3])},
            {'name': 'fat', 'value': float(row[4])},
            {'name': 'carbohydrates', 'value': float(row[5])},
            {'name': 'vitamins', 'value': float(row[6])}
        ]


@app.route('/')
def index():
    img = 'static/profile.jpg'
    return render_template('index.html', img=img)


@app.route('/recognize')
def magic():
    return render_template('recognize.html', img=file)


@app.route('/upload', methods=['POST'])
def upload():
    
    #Camera End
    file = request.files.getlist("img")
    for f in file:
        filename = secure_filename(str(num[0] + 500) + '.jpg')
        num[0] += 1
        name = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print('save name', name)
        f.save(name)

    pack[0] = []
    return render_template('recognize.html', img=file)


@app.route('/predict1')
def predict1():
    cam = cv2.VideoCapture(0)

    # reading the input using the camera
    result, image = cam.read()

    # If image will detected without any error,
    # show result
    if result:

        # showing result, it take frame name and image
        # output
        cv2.imshow("image", image)

        # saving image in local storage
        cv2.imwrite("image12.png", image)

        # If keyboard interrupt occurs, destroy image
        # window
        cv2.waitKey(0)
        #cv2.destroyWindow("image")

    # If captured image is corrupted, moving to else part
    else:
        print("No image detected. Please! try again")

    
    return render_template('recognize.html', img=file)

@app.route('/predict')
def predict():
    result = []
    # pack = []
    print('total image', num[0])
    for i in range(start[0], num[0]):
        pa = dict()

        filename = f'{UPLOAD_FOLDER}/{i + 500}.jpg'
        print('image filepath', filename)
        pred_img = filename
        pred_img = image.load_img(pred_img, target_size=(200, 200))
        pred_img = image.img_to_array(pred_img)
        pred_img = np.expand_dims(pred_img, axis=0)
        pred_img = pred_img / 255.

        pred = model_best.predict(pred_img)
        print("Pred")
        print(pred)

        if math.isnan(pred[0][0]) and math.isnan(pred[0][1]) and \
                math.isnan(pred[0][2]) and math.isnan(pred[0][3]):
            pred = np.array([0.05, 0.05, 0.05, 0.07, 0.09, 0.19, 0.55, 0.0, 0.0, 0.0, 0.0])

        top = pred.argsort()[0][-3:]
        print("top: ", top)
        label.sort()
        _true = label[top[2]]
        top_01 = label[top[1]]
        top_02 = label[top[0]]

        if(_true in south_food):
            _true_ = _true + " (South India)"
        elif(_true in north_food):
            _true_ = _true + " (North India)"

        if(top_01 in south_food):
            top_01_ = top_01 + " (South India)"
        elif(top_01 in north_food):
            top_01_ = top_01 + " (North India)"

        if(top_02 in south_food):
            top_02_ = top_02 + " (South India)"
        elif(top_02 in north_food):
            top_02_ = top_02 + " (North India)"

        pa['image'] = f'{UPLOAD_FOLDER}/{i + 500}.jpg'
        x = dict()
        x[_true_] = float("{:.2f}".format(pred[0][top[2]] * 100))
        x[top_01_] = float("{:.2f}".format(pred[0][top[1]] * 100))
        x[top_02_] = float("{:.2f}".format(pred[0][top[0]] * 100))
        pa['result'] = x
        pa['nutrition'] = nutrition_table[_true]
        pa['food'] = f'{nu_link}{_true}'
        pa['idx'] = i - start[0]
        pa['quantity'] = 100

        pack[0].append(pa)
        passed[0] += 1

    start[0] = passed[0]
    print('successfully packed')
    # compute the average source of calories
    for p in pack[0]:
        nutrients[0]['value'] = (nutrients[0]['value'] + p['nutrition'][0]['value'])
        nutrients[1]['value'] = (nutrients[1]['value'] + p['nutrition'][1]['value'])
        nutrients[2]['value'] = (nutrients[2]['value'] + p['nutrition'][2]['value'])
        nutrients[3]['value'] = (nutrients[3]['value'] + p['nutrition'][3]['value'])
        nutrients[4]['value'] = (nutrients[4]['value'] + p['nutrition'][4]['value'])

    nutrients[0]['value'] = nutrients[0]['value'] / num[0] + 0.0000000000000000000000000000000000001
    nutrients[1]['value'] = nutrients[1]['value'] / num[0] + 0.0000000000000000000000000000000000001
    nutrients[2]['value'] = nutrients[2]['value'] / num[0] + 0.0000000000000000000000000000000000001
    nutrients[3]['value'] = nutrients[3]['value'] / num[0] + 0.0000000000000000000000000000000000001
    nutrients[4]['value'] = nutrients[4]['value'] / num[0] + 0.0000000000000000000000000000000000001

    return render_template('results.html', pack=pack[0], whole_nutrition=nutrients)


@app.route('/update', methods=['POST'])
def update():
    return render_template('index.html', img='static/P2.jpg')


if __name__ == "__main__":
    import click

    @click.command()
    @click.option('--debug', is_flag=True)
    @click.option('--threaded', is_flag=True)
    @click.argument('HOST', default='127.0.0.1')
    @click.argument('PORT', default=5000, type=int)
    def run(debug, threaded, host, port):
        """
        This function handles command line parameters.
        Run the server using
            python server.py
        Show the help text using
            python server.py --help
        """
        HOST, PORT = host, port
        app.run(host=HOST, port=PORT, debug=debug, threaded=threaded)
    run()
    
# import tensorflow
# from flask import Flask, request, render_template
# import csv
# import math
# import os
# import numpy as np
# from tensorflow.keras.preprocessing import image
# from tensorflow.python.keras.models import load_model
# from werkzeug.utils import secure_filename

# tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
# app = Flask(__name__, template_folder=tmpl_dir)

# UPLOAD_FOLDER = 'static/uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # define label meaning
# label_11 = [
#     'apple pie', 'cheesecake', 'chicken wings', 'churros', 'cup_cakes', 'donuts', 'dumplings',
#     'samosa', 'pizza', 'omelette', 'ramen'
# ]

# label_3 = [
#     'samosa', 'pizza', 'omelette'
# ]

# label_101 = ['apple pie',
#          'baby back ribs',
#          'baklava',
#          'beef carpaccio',
#          'beef tartare',
#          'beet salad',
#          'beignets',
#          'bibimbap',
#          'bread pudding',
#          'breakfast burrito',
#          'bruschetta',
#          'caesar salad',
#          'cannoli',
#          'caprese salad',
#          'carrot cake',
#          'ceviche',
#          'cheese plate',
#          'cheesecake',
#          'chicken curry',
#          'chicken quesadilla',
#          'chicken wings',
#          'chocolate cake',
#          'chocolate mousse',
#          'churros',
#          'clam chowder',
#          'club sandwich',
#          'crab cakes',
#          'creme brulee',
#          'croque madame',
#          'cup cakes',
#          'deviled eggs',
#          'donuts',
#          'dumplings',
#          'edamame',
#          'eggs benedict',
#          'escargots',
#          'falafel',
#          'filet mignon',
#          'fish and_chips',
#          'foie gras',
#          'french fries',
#          'french onion soup',
#          'french toast',
#          'fried calamari',
#          'fried rice',
#          'frozen yogurt',
#          'garlic bread',
#          'gnocchi',
#          'greek salad',
#          'grilled cheese sandwich',
#          'grilled salmon',
#          'guacamole',
#          'gyoza',
#          'hamburger',
#          'hot and sour soup',
#          'hot dog',
#          'huevos rancheros',
#          'hummus',
#          'ice cream',
#          'lasagna',
#          'lobster bisque',
#          'lobster roll sandwich',
#          'macaroni and cheese',
#          'macarons',
#          'miso soup',
#          'mussels',
#          'nachos',
#          'omelette',
#          'onion rings',
#          'oysters',
#          'pad thai',
#          'paella',
#          'pancakes',
#          'panna cotta',
#          'peking duck',
#          'pho',
#          'pizza',
#          'pork chop',
#          'poutine',
#          'prime rib',
#          'pulled pork sandwich',
#          'ramen',
#          'ravioli',
#          'red velvet cake',
#          'risotto',
#          'samosa',
#          'sashimi',
#          'scallops',
#          'seaweed salad',
#          'shrimp and grits',
#          'spaghetti bolognese',
#          'spaghetti carbonara',
#          'spring rolls',
#          'steak',
#          'strawberry shortcake',
#          'sushi',
#          'tacos',
#          'octopus balls',
#          'tiramisu',
#          'tuna tartare',
#          'waffles']

# label = label_11.copy()

# nu_link = 'https://www.nutritionix.com/food/'

# # Loading the best saved model to make predictions.
# tensorflow.keras.backend.clear_session()
# model_best = tensorflow.keras.models.load_model(
#     'best_model_11class_new.hdf5',
#     custom_objects={'Functional':tensorflow.keras.models.Model})
# #model_best = load_model('best_model_11class.hdf5', compile=False)
# print('model successfully loaded!')

# start = [0]
# passed = [0]
# pack = [[]]
# num = [0]

# nutrients = [
#     {'name': 'protein', 'value': 0.0},
#     {'name': 'calcium', 'value': 0.0},
#     {'name': 'fat', 'value': 0.0},
#     {'name': 'carbohydrates', 'value': 0.0},
#     {'name': 'vitamins', 'value': 0.0}
# ]

# with open('nutrition101.csv', 'r') as file:
#     reader = csv.reader(file)
#     nutrition_table = dict()
#     for i, row in enumerate(reader):
#         if i == 0:
#             name = ''
#             continue
#         else:
#             name = row[1].strip()
#         nutrition_table[name] = [
#             {'name': 'protein', 'value': float(row[2])},
#             {'name': 'calcium', 'value': float(row[3])},
#             {'name': 'fat', 'value': float(row[4])},
#             {'name': 'carbohydrates', 'value': float(row[5])},
#             {'name': 'vitamins', 'value': float(row[6])}
#         ]


# @app.route('/')
# def index():
#     img = 'static/profile.jpg'
#     return render_template('index.html', img=img)


# @app.route('/recognize')
# def magic():
#     return render_template('recognize.html', img=file)


# @app.route('/upload', methods=['POST'])
# def upload():
#     file = request.files.getlist("img")
#     for f in file:
#         filename = secure_filename(str(num[0] + 500) + '.jpg')
#         num[0] += 1
#         name = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         print('save name', name)
#         f.save(name)

#     pack[0] = []
#     return render_template('recognize.html', img=file)


# @app.route('/predict')
# def predict():
#     result = []
#     # pack = []
#     print('total image', num[0])
#     for i in range(start[0], num[0]):
#         pa = dict()

#         filename = f'{UPLOAD_FOLDER}/{i + 500}.jpg'
#         print(filename)
#         print('image filepath', filename)
#         pred_img = filename
#         pred_img = image.load_img(pred_img, target_size=(200, 200))
#         pred_img = image.img_to_array(pred_img)
#         pred_img = np.expand_dims(pred_img, axis=0)
#         pred_img = pred_img / 255.

#         pred = model_best.predict(pred_img)
#         print("Pred")
#         print(pred)

#         if math.isnan(pred[0][0]) and math.isnan(pred[0][1]) and \
#                 math.isnan(pred[0][2]) and math.isnan(pred[0][3]):
#             pred = np.array([0.05, 0.05, 0.05, 0.07, 0.09, 0.19, 0.55, 0.0, 0.0, 0.0, 0.0])

#         top = pred.argsort()[0][-3:]
#         print("top",top)
#         label.sort()
#         _true = label[top[2]]
#         pa['image'] = f'{UPLOAD_FOLDER}/{i + 500}.jpg'
#         x = dict()
#         x[_true] = float("{:.2f}".format(pred[0][top[2]] * 100))
#         x[label[top[1]]] = float("{:.2f}".format(pred[0][top[1]] * 100))
#         x[label[top[0]]] = float("{:.2f}".format(pred[0][top[0]] * 100))
#         pa['result'] = x
#         pa['nutrition'] = nutrition_table[_true]
#         pa['food'] = f'{nu_link}{_true}'
#         pa['idx'] = i - start[0]
#         pa['quantity'] = 100

#         pack[0].append(pa)
#         passed[0] += 1

#     start[0] = passed[0]
#     print('successfully packed')
#     # compute the average source of calories
#     for p in pack[0]:
#         nutrients[0]['value'] = (nutrients[0]['value'] + p['nutrition'][0]['value'])
#         nutrients[1]['value'] = (nutrients[1]['value'] + p['nutrition'][1]['value'])
#         nutrients[2]['value'] = (nutrients[2]['value'] + p['nutrition'][2]['value'])
#         nutrients[3]['value'] = (nutrients[3]['value'] + p['nutrition'][3]['value'])
#         nutrients[4]['value'] = (nutrients[4]['value'] + p['nutrition'][4]['value'])

#     nutrients[0]['value'] = nutrients[0]['value'] / num[0]
#     nutrients[1]['value'] = nutrients[1]['value'] / num[0]
#     nutrients[2]['value'] = nutrients[2]['value'] / num[0]
#     nutrients[3]['value'] = nutrients[3]['value'] / num[0]
#     nutrients[4]['value'] = nutrients[4]['value'] / num[0]

#     return render_template('results.html', pack=pack[0], whole_nutrition=nutrients)


# @app.route('/update', methods=['POST'])
# def update():
#     return render_template('index.html', img='static/P2.jpg')


# if __name__ == "__main__":
#     import click

#     @click.command()
#     @click.option('--debug', is_flag=True)
#     @click.option('--threaded', is_flag=True)
#     @click.argument('HOST', default='127.0.0.1')
#     @click.argument('PORT', default=5000, type=int)
#     def run(debug, threaded, host, port):
#         """
#         This function handles command line parameters.
#         Run the server using
#             python server.py
#         Show the help text using
#             python server.py --help
#         """
#         HOST, PORT = host, port
#         app.run(host=HOST, port=PORT, debug=debug, threaded=threaded)
#     run()
