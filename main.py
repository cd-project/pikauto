import shutil
import string
import time
from collections import OrderedDict, deque
import pyautogui
import tkinter as tk
import cv2
import os
from PIL import Image, ImageTk
import numpy as np

root_folder = os.path.dirname(os.path.abspath(__file__))
print(root_folder)


def clear_folder(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder does not exist: {folder_path}")
        return

    # Clear the folder contents
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

    print(f"Folder cleared: {folder_path}")


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        clear_folder(folder_path)


def cut(img_path, image_dir, pokemon_height, pokemon_width):
    img = cv2.imread(img_path)
    resize_size = (1008, 675)
    img_resize = cv2.resize(img, resize_size)
    height, width, _ = img_resize.shape
    # Loop through each Pokemon image and separate it
    for row in range(0, height, pokemon_height):
        for col in range(0, width, pokemon_width):
            # Crop the image to get the current Pokemon
            pokemon_img = img_resize[row:row+pokemon_height, col:col+pokemon_width]
            hi, wi = pokemon_img.shape[:2]

            # Define the crop area
            x, y, w, h = 4, 2, wi-8, hi-4

            # Crop the image
            crop_img = pokemon_img[y:y+h, x:x+w]

            # Save the current Pokemon as a separate file
            cv2.imwrite(f'{image_dir}/pokemon_{row//pokemon_height+1}_{col//pokemon_width+1}.jpg', crop_img)


# Define the size of each Pokemon image
pokemon_height = 75
pokemon_width = 63
image_dir = root_folder + '/dir'
template_dir = root_folder + '/template'
create_folder(image_dir)

img_path = "/home/deeteecee/PycharmProjects/pikauto/pikakaka.jpg"
main_img = cv2.imread(img_path)
cut(img_path, image_dir, pokemon_height, pokemon_width)

normalize_size = (75, 63)
template_images = {}
for file in os.listdir(template_dir):
    if file.endswith('.png') or file.endswith('.jpg'):
        # Read the image and resize it to a common size (e.g., 100x100)
        img = cv2.imread(os.path.join(template_dir, file), cv2.IMREAD_GRAYSCALE)
        # Extract the pokemon type from the filename (assuming the filename format is 'type.png')
        pokemon_type = file.split('.')[0]

        # Add the image to the template_images dictionary with the pokemon type as the key
        template_images[pokemon_type] = img


def compare_edges_ncc(edges1, edges2):
    edges1_norm = cv2.normalize(edges1.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
    edges2_norm = cv2.normalize(edges2.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)

    ncc = cv2.matchTemplate(edges1_norm, edges2_norm, cv2.TM_CCORR_NORMED)
    simi = np.max(ncc)

    return simi


def compare_images_by_color(img_path1, img_path2):

    image1 = cv2.imread(img_path1)
    image2 = cv2.imread(img_path2)
    # Convert images to Lab color space
    image1_lab = cv2.cvtColor(image1, cv2.COLOR_BGR2Lab)
    image2_lab = cv2.cvtColor(image2, cv2.COLOR_BGR2Lab)

    # Compute the average color of each image
    avg_color1 = np.mean(image1_lab, axis=(0, 1))
    avg_color2 = np.mean(image2_lab, axis=(0, 1))

    # Compute the mean squared error (MSE)
    mse = np.mean((avg_color1 - avg_color2) ** 2)

    # Return the MSE value
    return mse


for file in os.listdir(image_dir):
    # Check that the file is an image
    if file.endswith('.png') or file.endswith('.jpg'):
        # Read the image and resize it to a common size
        img = cv2.imread(os.path.join(image_dir, file), cv2.IMREAD_GRAYSCALE)

        # Calculate the similarity between the image and each template image
        similarities = {}
        for pokemon_type, template_img in template_images.items():

            edge_img = cv2.Canny(img, 100, 300)
            edge_template = cv2.Canny(template_img, 100, 300)
            similarities[pokemon_type] = compare_edges_ncc(edge_img, edge_template)
        # Get the pokemon type with the highest similarity
            sorted_similarities = dict(sorted(similarities.items(), key=lambda x: x[1]))
            values = list(sorted_similarities.values())[-2:]


        pokemon_type = max(similarities, key=similarities.get)

        # Create a directory for the pokemon type if it doesn't exist
        if not os.path.exists(os.path.join(image_dir, pokemon_type)):
            os.makedirs(os.path.join(image_dir, pokemon_type))

        # Move the image to the corresponding directory
        shutil.move(os.path.join(image_dir, file), os.path.join(image_dir, pokemon_type, file))


def transform_to_matrix(image_path):
    rows, columns = 9, 16
    matrix = [["" for _ in range(columns)] for _ in range(rows)]
    prefix = image_dir + '/'
    split_symbol = '/'
    to_remove = {".jpg": ""}
    file_names = []
    for root, dirs, files in os.walk(image_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if os.path.isfile(file_path):
                fp = file_path[len(prefix):]
                file_names.append(fp)

    for fn in file_names:
        split_parts = fn.split(split_symbol)
        poke_type = split_parts[0]
        pos_string = None
        for key, val in to_remove.items():
            pos_string = split_parts[1].replace(key, val)

        pos_string = pos_string.lstrip("pokemon_")
        ps = pos_string.split('_')
        r = int(ps[0]) - 1
        c = int(int(ps[1]) - 1)
        matrix[r][c] = poke_type

    return matrix


def bfs1(matrix, start, end):
    rows = len(matrix)
    cols = len(matrix[0])
    visited = [[False] * cols for _ in range(rows)]
    queue = deque([(start, [], 0)])

    while queue:
        current, path, turns = queue.popleft()
        x, y = current

        if current == end:
            path.append(current)
            return path, turns

        if x < 0 or x >= rows or y < 0 or y >= cols:
            continue

        if visited[x][y] or matrix[x][y] != 1:
            continue

        visited[x][y] = True
        path.append(current)

        # Only allow two turns
        if turns >= 2:
            continue

        # Add new paths with possible turns
        queue.append(((x + 1, y), path[:], turns + (0 if len(path) < 2 else 1)))
        queue.append(((x - 1, y), path[:], turns + (0 if len(path) < 2 else 1)))
        queue.append(((x, y + 1), path[:], turns + (0 if len(path) < 2 or path[-1][0] == path[-2][0] else 1)))
        queue.append(((x, y - 1), path[:], turns + (0 if len(path) < 2 or path[-1][0] == path[-2][0] else 1)))

    return None, None

def bfs(matrix, start, end):
    rows = len(matrix)
    cols = len(matrix[0])
    visited = [[False] * cols for _ in range(rows)]  # Initialize visited matrix
    parent = [[None] * cols for _ in range(rows)]  # Initialize parent matrix
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Define possible movements: up, down, left, right
    queue = deque([(start[0], start[1], None, 0)])  # Queue to store cells to visit, along with previous direction and number of turns

    while queue:
        row, col, prev_direction, turns = queue.popleft()

        if (row, col) == end:
            break  # Found the end point

        visited[row][col] = True  # Mark current cell as visited

        for dx, dy in directions:
            new_row, new_col = row + dx, col + dy

            if 0 <= new_row < rows and 0 <= new_col < cols and not visited[new_row][new_col]:
                new_turns = turns + int((prev_direction is not None and (dx, dy) != prev_direction))

                if new_turns <= 2 and (new_row, new_col) == end or matrix[new_row][new_col] == 1:
                    queue.append((new_row, new_col, (dx, dy), new_turns))
                    visited[new_row][new_col] = True
                    parent[new_row][new_col] = (row, col)

    if parent[end[0]][end[1]] is None:
        return []  # Path not found

    path = []
    curr = end
    while curr:
        path.append(curr)
        curr = parent[curr[0]][curr[1]]
    path.reverse()

    return path



rows = 11
cols = 18

# Create an empty matrix filled with zeros
match_m = [[0] * cols for _ in range(rows)]

# Set border elements to 1
for idx in range(rows):
    for jdx in range(cols):
        if idx == 0 or idx == rows - 1 or jdx == 0 or jdx == cols - 1:
            match_m[idx][jdx] = 1


def matchable(matrix, _x, _y, x, y):
    start = (_x, _y)
    end = (x, y)
    path = bfs(matrix, start, end)
    if len(path) > 0:
        return path, True
    else:
        return [], False


m = transform_to_matrix(image_dir)


def is_matrix_all_ones(mat):
    for row in mat:
        for element in row:
            if element != 1:
                return False
    return True


pts_storage = []


def paint_black(image, i, j):
    top_left = (i*pokemon_width, j*pokemon_height)
    bot_right = ((i+1)*pokemon_width-20, (j+1)*pokemon_height-20)
    cv2.rectangle(image, top_left, bot_right, (0, 0, 0), -1)


padding = 50


def get_center(i, j):
    return (i*pokemon_width+pokemon_width//2, j*pokemon_height+pokemon_height//2)


def draw_path(img, i, j, i_d, j_d):
    center_1 = get_center(i, j)
    center_2 = get_center(i_d, j_d)

    cv2.line(img, center_1, center_2, (0, 0, 255), thickness=2)


def play():
    while not is_matrix_all_ones(match_m):
        for i, row_m in enumerate(m):
            for j, element in enumerate(row_m):
                for i_x, r in enumerate(m):
                    for j_x, element_x in enumerate(r):
                        if (i, j) != (i_x, j_x) and element == element_x and element != "0" and element_x != "0":
                            if (i, j) in pts_storage or (i_x, j_x) in pts_storage:
                                continue

                            path, match = matchable(match_m, i+1, j+1, i_x+1, j_x+1)

                            if match is True:
                                print(i+1, j+1, i_x+1, j_x+1, path, match)
                                # for i, p in enumerate(path):
                                #     if i < len(path)-1:
                                #         p1 = path[i]
                                #         p2 = path[i+1]
                                #         draw_path(main_img, p1[0], p1[1], p2[0], p2[1])
                                # cv2.imshow('res', main_img)
                                # cv2.waitKey(500)
                                m[i][j] = "0"
                                m[i_x][j_x] = "0"
                                match_m[i+1][j+1] = 1
                                match_m[i_x+1][j_x+1] = 1
                                pts_storage.append((i, j))
                                pts_storage.append((i_x, j_x))
                                print(len(pts_storage))
                                paint_black(main_img, j, i)
                                paint_black(main_img, j_x, i_x)
                                cv2.imshow('res', main_img)
                                cv2.waitKey(500)


play()


# print(cnt)
# while True:
#     play()
# opencv_image = cv2.imread(img_path)
# pil_image = Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))
#
# # Create a Tkinter window
# window = tk.Tk()
#
# window.width = pil_image.width + 2*pokemon_width
# window.height = pil_image.height + 2*pokemon_height
# # Create a Tkinter-compatible image
# tk_image = ImageTk.PhotoImage(pil_image)
#
# # Create a label and display the image
# label = tk.Label(window, image=tk_image)
# label.place(x=pokemon_width, y=pokemon_height)
#
# # Run the main event loop
# window.mainloop()


