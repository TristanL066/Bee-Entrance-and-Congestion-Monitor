import argparse
from ultralytics import YOLO
 
 
def train(data_yaml, model_size, epochs, imgsz, batch):
    # "n" = nano, smallest/fastest -- good starting point, especially since
    # this will eventually run on a Raspberry Pi 5.
    model = YOLO(f"yolo11{model_size}.pt")  # downloads pretrained weights once, then caches locally
 
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=20,        # stop early if val performance plateaus
        project="runs",
        name="bee_detector",
    )
 
    # Validate on the val split and print metrics (precision, recall, mAP)
    metrics = model.val()
    print(metrics)
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a bee detection YOLO model")
    parser.add_argument("--data", required=True, help="Path to Roboflow-exported data.yaml")
    parser.add_argument("--model-size", default="n", choices=["n", "s", "m"],
                         help="Model size: n=nano (fastest, best for Pi), s=small, m=medium")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16, help="Lower this if you run out of memory")
    args = parser.parse_args()
 
    train(args.data, args.model_size, args.epochs, args.imgsz, args.batch)
 
