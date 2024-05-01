from flask import request, send_from_directory, jsonify, render_template, redirect, url_for
import os
# import uuid
from werkzeug.utils import secure_filename
import process_image


def init_app(app):
    @app.route('/')
    def home():
        return render_template('home.html')

    @app.route('/edit')
    def edit():
        data = request.args.get('data', None)
        return render_template('edit.html', data=data)

    @app.route('/display')
    def display():
        data = request.args.get('data', None)
        return render_template('display.html', data=data)

    @app.route('/navigate', methods=['POST'])
    def navigate():
        data = request.json
        dest = data.get('dest')
        page_data = data.get('page_data', '')
        if dest == 'edit':
            return redirect(url_for('edit', data=page_data))
        else:
            return redirect(url_for('display', data=page_data))

    @app.route('/process_upload', methods=['POST'])
    def upload_file():
        print("Uploading File...")

        if 'file' not in request.files:
            print("No file provided!")
            return jsonify({'error': "No file provided!"}), 400

        file = request.files['file']
        if file.filename == '':
            print("No selected file!")
            return jsonify({'error': "No selected file!"}), 400

        # Convert the input PDF into a PNG image
        image = process_image.convert_to_png(file.stream)

        # Proceed with preprocessing if conversion was successful
        if image:
            # Make near-black pixels black and all others white
            preprocessed_image = process_image.preprocess_image(image)

            # Convert outside white to transparent
            processed_image = process_image.make_white_areas_transparent(preprocessed_image)

            # Use the original filename but ensure it is secure
            original_name = secure_filename(file.filename)
            filename_base, _ = os.path.splitext(original_name)
            filename = f"{filename_base}.png"
            file_path = os.path.join(app.config['IMAGE_FOLDER'], filename)
            processed_image.save(file_path)

            # Return the URL for the processed image
            image_url = '/' + app.config['IMAGE_FOLDER'] + '/' + filename
            return jsonify({'image_url': image_url})
        else:
            return jsonify({'error': "Failed to convert PDF to PNG!"}), 500

    @app.route('/process_directory_upload', methods=['POST'])
    def upload_directory():
        print("Uploading Directory...")

        # 'files[]' is the name attribute for the <input> tag for directory uploads
        uploaded_files = request.files.getlist('files[]')

        for file in uploaded_files:
            if file.filename == '':
                print("No selected file!")
                continue

            # Convert the input PDF into a PNG image
            image = process_image.convert_to_png(file.stream)

            # Proceed with preprocessing if conversion was successful
            if image:
                # Make near-black pixels black and all others white
                preprocessed_image = process_image.preprocess_image(image)

                # Convert outside white to transparent
                processed_image = process_image.make_white_areas_transparent(preprocessed_image)

                # Generate a filename and save the image
                original_name = secure_filename(file.filename)
                filename_base, _ = os.path.splitext(original_name)
                filename = f"{filename_base}.png"
                file_path = os.path.join(app.config['IMAGE_FOLDER'], filename)
                processed_image.save(file_path)

                images = process_image.split_image(file_path)
                for index, image in enumerate(images):
                    original_name = secure_filename(filename)
                    filename_base, _ = os.path.splitext(original_name)
                    new_filename = f"{filename_base}_{index + 1}.png"
                    file_path = os.path.join(app.config['IMAGE_FOLDER'], new_filename)
                    image.save(file_path)
                    cropped_image = process_image.crop_image(file_path)
                    cropped_image.save(file_path)
            else:
                print(f"Failed to convert {file.filename} to PNG!")

        return jsonify({'image_dir': app.config['IMAGE_FOLDER']})


    @app.route('/process_click', methods=['POST'])
    def process_click():
        print("Processing click...")

        data = request.json  # Get JSON data from the request
        x = data.get('x')
        y = data.get('y')
        filename = data.get('filename').split('?')[0]
        file_path = os.path.join(app.config['IMAGE_FOLDER'], filename)

        image = process_image.select_block(file_path, x, y)
        image.save(file_path)

        image_url = '/' + app.config['IMAGE_FOLDER'] + '/' + filename
        return jsonify({'image_url': image_url})

    @app.route('/process_submit', methods=['POST'])
    def process_submit():
        print("Processing submit..")

        data = request.json
        filename = data.get('filename').split('?')[0]
        split = data.get('split')
        file_path = os.path.join(app.config['IMAGE_FOLDER'], filename)

        image = process_image.remove_non_selected_blocks(file_path)
        image.save(file_path)

        image_urls = []
        if split:
            images = process_image.split_image(file_path)
            image_urls = []  # List to hold the URLs of the saved images
            for index, image in enumerate(images):
                original_name = secure_filename(filename)  # Assuming 'file' is the uploaded file
                filename_base, _ = os.path.splitext(original_name)
                new_filename = f"{filename_base}_{index + 1}.png"
                file_path = os.path.join(app.config['IMAGE_FOLDER'], new_filename)
                image.save(file_path)
                cropped_image = process_image.crop_image(file_path)
                cropped_image.save(file_path)
                image_url = '/' + app.config['IMAGE_FOLDER'] + '/' + new_filename
                image_urls.append(image_url)
        else:
            image_url = '/' + app.config['IMAGE_FOLDER'] + '/' + filename
            image_urls.append(image_url)

        # Return a list of URLs for all processed images
        return jsonify({'image_urls': image_urls})

    @app.route('/images/<filename>')
    def serve_image(filename):
        return send_from_directory(app.config['IMAGE_FOLDER'], filename)
