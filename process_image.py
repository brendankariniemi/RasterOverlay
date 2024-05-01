import numpy as np
import fitz  # PyMuPDF
import io
from PIL import Image
from scipy.ndimage import label, find_objects
import cv2

white_color = [255, 255, 255]  # White in BGR
red_color = [195, 195, 255]  # Red in BGR
black_color = [0, 0, 0]  # Black in BGR


def crop_image(img_path):
    img = Image.open(img_path)
    img = img.convert('RGBA')
    bbox = img.getbbox()
    if bbox:
        cropped_image = img.crop(bbox)
        return cropped_image
    else:
        return None


def split_image(img_path):
    # Load the image
    img = Image.open(img_path)
    img_array = np.array(img)

    # Only consider the alpha channel to identify the islands of non-transparent pixels
    alpha_channel = img_array[:, :, 3]

    # Use a threshold to create a binary mask (0 for transparent, 1 for non-transparent)
    binary_mask = alpha_channel > 0

    # Label connected components
    labeled_array, num_features = label(binary_mask)

    # Split the image into multiple images based on the labeled islands
    split_images = []
    for i in range(1, num_features + 1):
        # Create a mask for the current island
        island_mask = labeled_array == i

        # Create a new image array filled with transparent pixels
        new_img_array = np.zeros(img_array.shape, dtype=img_array.dtype)

        # Copy the island's pixels to the new image array
        for j in range(3):  # For RGB channels
            new_img_array[:, :, j] = img_array[:, :, j] * island_mask

        # Set the alpha channel
        new_img_array[:, :, 3] = alpha_channel * island_mask

        # Convert the array back to an image
        new_img = Image.fromarray(new_img_array)
        split_images.append(new_img)

    # Return the list of split images
    return split_images


def convert_to_png(file_stream):
    # Read the stream into bytes
    file_bytes = file_stream.read()

    try:
        doc = fitz.open("pdf", file_bytes)
        if len(doc) > 0:
            page = doc[0]
            pix = page.get_pixmap()
            data = pix.tobytes("png")
            img = Image.open(io.BytesIO(data))

            doc.close()
            return img
        else:
            doc.close()
            return None
    except fitz.FileDataError as e:
        print(f"Failed to open the PDF: {e}")
        return None


def preprocess_image(img, threshold=254):
    try:
        # Convert the image to grayscale and then to a numpy array
        grayscale = img.convert('L')
        data = np.array(grayscale)

        # Convert pixels below the threshold to white, and the rest to black
        black_white = np.where(data < threshold, 0, 255).astype(np.uint8)

        # Return the PIL image
        return Image.fromarray(black_white, 'L')
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        return None


def make_white_areas_transparent(img):
    try:
        # Convert the image to RGBA and then to a NumPy array
        img = img.convert("RGBA")
        data = np.array(img)

        # Identify all white pixels
        white_pixels = (data[:, :, 0] == 255) & (data[:, :, 1] == 255) & (data[:, :, 2] == 255)

        # Label connected regions of white_pixels
        structure = np.ones((3, 3), dtype=int)  # Define connectivity
        labeled, ncomponents = label(white_pixels, structure)

        # Calculate the size of each component and get the largest one
        sizes = np.bincount(labeled.ravel())
        background_label = sizes[1:].argmax() + 1

        # Make the background label transparent
        background = (labeled == background_label)
        data[background] = (255, 255, 255, 0)

        # Return modified image
        return Image.fromarray(data, 'RGBA')
    except Exception as e:
        print(f"Error in making white areas transparent: {e}")
        return None


def select_block(img_path, x, y):
    target_colors = [white_color, red_color]

    # Open and convert the image to RGBA
    img = Image.open(img_path)
    img = img.convert('RGBA')

    # Convert image to NumPy array
    data = np.array(img)

    # Get the clicked pixel's RGBA values
    clicked_pixel = data[y, x]

    # If the clicked pixel is transparent, return the original image
    if clicked_pixel[3] == 0:
        return img
    else:
        # Convert to BGRA for OpenCV processing
        cv_data = cv2.cvtColor(data, cv2.COLOR_RGBA2BGRA)

        # Ensure the image is in an appropriate format
        cv_data = cv_data.astype(np.uint8)
        cv_data = np.ascontiguousarray(cv_data)

        if cv_data.shape[2] != 4:
            print("Image does not have an alpha channel.")
            return img

        bgr_image = cv_data[:, :, :3]
        alpha_channel = cv_data[:, :, 3]

        rows, cols, _ = bgr_image.shape
        filled = np.zeros((rows, cols), dtype=np.bool_)
        to_fill = [(x, y)]

        while to_fill:
            x, y = to_fill.pop()
            if not (0 <= x < cols and 0 <= y < rows):
                continue  # Out of bounds
            if filled[y, x] or alpha_channel[y, x] == 0:
                continue  # Already filled or transparent

            current_color = bgr_image[y, x].tolist()
            if current_color not in target_colors and current_color != black_color:
                continue  # Skip if not target color and not black

            new_color = None
            # Determine new color based on current color
            if current_color == white_color:
                new_color = red_color
            elif current_color == red_color:
                new_color = white_color

            # Apply new color if a change is determined
            if new_color is not None:
                bgr_image[y, x] = new_color
            filled[y, x] = True

            # Add neighboring pixels (8-connectivity)
            to_fill.extend([(x - 1, y - 1), (x, y - 1), (x + 1, y - 1),
                            (x - 1, y), (x + 1, y),
                            (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)])

        final_image = cv2.merge([bgr_image[:, :, 0], bgr_image[:, :, 1], bgr_image[:, :, 2], alpha_channel])
        recolored_data_rgba = cv2.cvtColor(final_image, cv2.COLOR_BGRA2RGBA)
        recolored_img = Image.fromarray(recolored_data_rgba)
        return recolored_img


def remove_non_selected_blocks(img_path):
    # Load the image and convert to RGBA
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] < 4:
        raise ValueError("Image does not have an alpha channel.")

    # Create a binary mask of non-transparent pixels
    alpha_channel = img[:, :, 3]
    non_transparent_mask = alpha_channel > 0

    # Find connected components in the mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(non_transparent_mask.astype(np.uint8),
                                                                            connectivity=8)

    # Prepare a mask to keep islands that are a mix of red and black
    keep_mask = np.zeros_like(non_transparent_mask)

    # Analyze each island
    for label in range(1, num_labels):  # Skip background
        component_mask = labels == label
        island_pixels = img[component_mask]

        # Criteria for red and black pixels
        is_red = np.all(island_pixels[:, :3] == red_color, axis=1)
        is_black = np.all(island_pixels[:, :3] == black_color, axis=1)

        # Check for the presence of both red and black pixels
        has_red = np.any(is_red)
        has_black = np.any(is_black)

        # Keep the island only if it contains both red and black pixels
        if has_red and has_black:
            keep_mask[component_mask] = 1

    # Apply the keep mask to the alpha channel
    img[:, :, 3] = img[:, :, 3] * keep_mask

    # Convert every non-black pixel to white, considering non-transparency
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if keep_mask[y, x] and not np.array_equal(img[y, x, :3], [0, 0, 0]):
                img[y, x] = [255, 255, 255, img[y, x, 3]]  # Set to white, preserving original alpha

    # Convert back to PIL Image and return
    cleaned_image = Image.fromarray(img)
    return cleaned_image


"""
def make_outside_of_borders_transparent(img_path):
    img = Image.open(img_path)
    img = img.convert('RGBA')  # Ensure image is in RGBA format
    border_color = (0, 0, 255, 255)  # The unique, specified border color

    data = np.array(img)  # Convert image to a NumPy array for processing
    height, width = data.shape[:2]
    queue = deque()
    visited = set()

    # Check if a pixel matches the border color
    def is_border(pixel):
        return np.array_equal(pixel, border_color)

    # Start marking from the edge if the pixel is not a border, aiming to find and mark outside background
    for x in range(width):
        if not is_border(data[0, x]): queue.append((x, 0))
        if not is_border(data[height - 1, x]): queue.append((x, height - 1))
    for y in range(height):
        if not is_border(data[y, 0]): queue.append((0, y))
        if not is_border(data[y, width - 1]): queue.append((width - 1, y))

    # Process pixels to find and mark the outside area
    while queue:
        x, y = queue.popleft()
        if (x, y) in visited or x < 0 or x >= width or y < 0 or y >= height: continue
        current_pixel = data[y, x]
        if is_border(current_pixel): continue  # Skip border pixels

        visited.add((x, y))
        data[y, x][3] = 0  # Make non-border, outside pixels transparent

        # Add neighboring pixels to the queue for processing
        for nx, ny in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                queue.append((nx, ny))

    # Convert the modified NumPy array back to an image and return it
    return Image.fromarray(data)


def make_colored_borders(data, x, y, border_color):
    height, width = data.shape[:2]
    queue = deque([(x, y)])
    visited = set()

    def is_transparent(pixel):
        return pixel[3] == 0

    while queue:
        x, y = queue.popleft()
        if (x, y) in visited or x < 0 or x >= width or y < 0 or y >= height:
            continue
        current_pixel = data[y, x]
        if is_transparent(current_pixel):
            continue
        visited.add((x, y))

        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1),
                     (x - 1, y - 1), (x + 1, y + 1), (x - 1, y + 1), (x + 1, y - 1)]

        is_border_pixel = any(
            is_transparent(data[ny, nx]) for nx, ny in neighbors if 0 <= nx < width and 0 <= ny < height)

        if is_border_pixel:
            data[y, x] = border_color

        for nx, ny in neighbors:
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                queue.append((nx, ny))

    return data
"""
