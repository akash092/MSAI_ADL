import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def get_image_path(info_path: str, view_index: int):
    info_path = Path(info_path)
    base_name = info_path.stem.replace("_info", "")
    return list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 100)
        img_height: Height of the image (default: 150)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """

    info_path = Path(info_path)
    base_name = info_path.stem.replace("_info", "")
    image_path = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    # print(f"{img_width=} {img_height=}")
    img_center_x, img_center_y = img_width/2, img_height/2
    # print(f"{img_center_x=} {img_center_y=}")
    center = np.array([img_center_x, img_center_y])

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)
    
    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT
    # print(f"{scale_x=} {scale_y=}")

    # Draw each detection
    result = []
    min_center_distance = -1
    ego_cart_track_id = -1
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)
        # print(f"{x1_scaled=} {y1_scaled=} {x2_scaled=} {y2_scaled=}")
        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue
        
        center_x, center_y = (x2_scaled + x1_scaled)/2, (y2_scaled + y1_scaled)/2
        # print(f"{center_x=} {center_y=}")
        p2 = np.array([center_x, center_y])
        center_distance = np.linalg.norm(center - p2)
        # print(f"{center_distance=} {ego_cart_track_id=} {min_center_distance=}")
        if ego_cart_track_id == -1 or center_distance < min_center_distance:
            min_center_distance = center_distance
            ego_cart_track_id = track_id
        result.append({"instance_id":track_id, "kart_name": info["karts"][track_id],
                       "center": (center_x, center_y), "is_center_kart": False})
    
    # based on ego_cart_track_id, adjust the element's is_center_kart in list
    for idx in range(len(result)):
        if result[idx]["instance_id"] == ego_cart_track_id:
            result[idx]["is_center_kart"] = True
            break
    
    return result

def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    return info["track"]

def get_dict(question: str, answer: str, image_file: str):
    return {"question":question, "answer": answer, "image_file": image_file}

def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 100)
        img_height: Height of the image (default: 150)

    Returns:
        List of dictionaries, each containing a question and answer
    """

    

    track_info = extract_track_info(info_path)
    karts_info = extract_kart_objects(info_path, view_index)
    result = []
    ego_cart_entry = None
    # Iterate through each entry and get ego cart
    for entry in karts_info:
        if entry["is_center_kart"] == True:
            ego_cart_entry = entry
            break

    if ego_cart_entry is None:
        return []
    image_path = str(get_image_path(info_path, view_index)).split("/",1)[1]
    # 1. Ego car question
    # What kart is the ego car?
    result.append(get_dict("What kart is the ego car?", ego_cart_entry["kart_name"], image_path))

    # 2. Total karts question
    # How many karts are there in the scenario?
    result.append(get_dict("How many karts are there in the scenario?", str(len(karts_info)), image_path))

    # 3. Track information questions
    # What track is this?
    result.append(get_dict("What track is this?", track_info, image_path))

    # 4. Relative position questions for each kart
    # Is {kart_name} to the left or right of the ego car?
    # Is {kart_name} in front of or behind the ego car?
    # Where is {kart_name} relative to the ego car?"

    left, right, front, behind = 0, 0, 0, 0
    for entry in karts_info:
        if entry["is_center_kart"] == True:
            continue
        if entry["center"][0] <= ego_cart_entry["center"][0]:
            x_alignment = "left"
            left += 1
        else:
            x_alignment = "right"
            right += 1
        result.append(get_dict("Is "+entry["kart_name"]+" to the left or right of the ego car?", x_alignment,image_path))
    
        if entry["center"][1] < ego_cart_entry["center"][1]:
            y_alignment = "front"
            front += 1
        else:
            y_alignment = "back"
            behind +=1 
        result.append(get_dict("Is "+entry["kart_name"]+" in front of or behind the ego car?", y_alignment, image_path))
        result.append(get_dict("Where is "+entry["kart_name"]+" relative to the ego car?", y_alignment + " and " +x_alignment, image_path))

    # 5. Counting questions
    # How many karts are to the left of the ego car?
    # How many karts are to the right of the ego car?
    # How many karts are in front of the ego car?
    # How many karts are behind the ego car?
    # No need to add those QA where answer is 0 since the test dataset doesn't have such questions
    if left != 0:
        result.append(get_dict("How many karts are to the left of the ego car?", str(left), image_path))
    if right != 0:
        result.append(get_dict("How many karts are to the right of the ego car?", str(right), image_path))
    if front != 0:
        result.append(get_dict("How many karts are in front of the ego car?", str(front), image_path))
    if behind != 0:
        result.append(get_dict("How many karts are behind the ego car?", str(behind), image_path))
    return result

def generate(data_dir: str, output_file: str):
    # Get all info files
    info_files = Path(data_dir).glob("*_info.json")
    qa_data = []
    for file in info_files:
        with open(file) as f:
            info = json.load(f)

        # Get the max detection frame
        for view_index in range(len(info["detections"])):
            
            # Generate QA pairs
            qa_pairs = generate_qa_pairs(file, view_index)
            qa_data.extend(qa_pairs)

    print(f"{output_file=}")
    with open(output_file, 'w') as file:
        json.dump(qa_data, file, indent=4)


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # # Print QA pairs
    # print("\nQuestion-Answer Pairs:")
    # print("-" * 50)
    # for qa in qa_pairs:
    #     print(f"Q: {qa['question']}")
    #     print(f"A: {qa['answer']}")
    #     print("-" * 50)


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_qa_pairs, "generate": generate})


if __name__ == "__main__":
    main()
