import argparse
import os
import cv2
from ultralytics import YOLO


def test_image(weights_path, image_path, conf_threshold, save_dir):
    model = YOLO(weights_path)

    results = model.predict(
        source=image_path,
        conf=conf_threshold,   # minimum confidence to count as a detection
        save=False,
    )

    os.makedirs(save_dir, exist_ok=True)

    for result in results:
        num_detections = len(result.boxes)
        print(f"\nImage: {image_path}")
        print(f"Bees detected: {num_detections}")
        print(f"Confidence threshold used: {conf_threshold}")

        if num_detections > 0:
            confidences = result.boxes.conf.tolist()
            print(f"Confidence range: {min(confidences):.2f} - {max(confidences):.2f}")

        # Draw boxes onto the image and save it under the same filename
        annotated = result.plot()  # returns a numpy array (BGR) with boxes drawn
        filename = os.path.basename(image_path)
        out_path = os.path.join(save_dir, filename)
        cv2.imwrite(out_path, annotated)

        print(f"\nAnnotated image saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test bee detection model on an image")
    parser.add_argument("--weights", required=True, help="Path to trained .pt weights file")
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--conf", type=float, default=0.25,
                         help="Confidence threshold (0-1). Lower catches more but risks false positives")
    parser.add_argument("--save-dir", default="bee_results", help="Folder to save annotated output into")
    args = parser.parse_args()

    test_image(args.weights, args.image, args.conf, args.save_dir)