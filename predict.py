import json
import os
from pathlib import Path
import torch
from PIL import Image
from ultralytics import YOLO
from argparse import ArgumentParser


def main(args):
    # Load the model (ensure that you have trained or downloaded a model)
    model = YOLO(args.modelpath)  # Replace with the path to your trained model

    # Path to the folder containing images
    image_folder = Path(args.imgpath)  # Replace with your folder path

    # Initialize an empty list to store results
    all_results = []
    total_infer_time = 0.0
    avg_infer_time = 0.0
    img_sum = 0
    # Iterate over all images in the folder
    for image_path in image_folder.glob('*.*'):  # This will match all files with any extension
        # infer_time = 0.0
        img_sum += 1
        if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:  # Filter image extensions
            print(f"Processing {image_path.name}...")

            # Perform inference
            results = model(image_path, conf=0.01, iou=0.25)

            # print("results:", results)
            # Iterate through results and format them
            for result in results:
                boxes = result.boxes  # Boxes object for bounding box outputs
                # print("boxes:", boxes)
                json_result = result.to_json()
                # print("json_result:", json_result)
                json_result = json.loads(json_result)
                # Extract total inference time by summing all the timing values in the 'speed' dict
                # infer_time = sum(result.speed.values())  # Sum 'preprocess', 'inference', and 'postprocess' times
                total_infer_time += result.speed["inference"]  # Add it to the total inference time

                for entry in json_result:
                    if isinstance(entry, dict):
                        image_id = image_path.name  # Image name as image_id
                        category_id = entry["class"]  # Class id (category_id)
                        bbox = [entry["box"]["x1"], entry["box"]["y1"], entry["box"]["x2"] - entry["box"]["x1"],
                                entry["box"]["y2"] - entry["box"]["y1"]]  # [xmin, ymin, width, height]
                        score = entry["confidence"]  # Confidence score

                        # Append formatted result to all_results
                        all_results.append({
                            "image_id": image_id,
                            "category_id": category_id,
                            "category_name": entry["name"],
                            "bbox": bbox,
                            "score": score
                        })
                    else:
                        print(f"Warning: Expected a dictionary, but found {type(entry)}")
    avg_infer_time = total_infer_time/img_sum
    print("avg_infer_time:", avg_infer_time, "ms")
    # Ensure the directory exists
    os.makedirs(args.savepath, exist_ok=True)


    output_json_path = args.savepath+'/best_predictions.json'
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print(f"All inference results saved to {output_json_path}")
if __name__ == "__main__":

    Model_PATH = './runs/train_smart/weights/best.pt'
    IMG_PATH = './images/test'
    SAVE_PATH = './runs/predict_smart/exp'

    parser = ArgumentParser()
    parser.add_argument('--modelpath', help='modelpath', default=Model_PATH)
    parser.add_argument('--imgpath', help='imgpath', default=IMG_PATH)
    parser.add_argument('--savepath', help='savepath', default=SAVE_PATH)
    get_params = parser.parse_args()
    main(get_params)
