import os
import views
from flask import Flask

app = Flask(__name__)

app.config['IMAGE_FOLDER'] = 'images'
image_folder_path = os.path.join(app.root_path, app.config['IMAGE_FOLDER'])
if not os.path.exists(image_folder_path):
    os.makedirs(image_folder_path)

views.init_app(app)


if __name__ == '__main__':
    app.run(debug=True)
