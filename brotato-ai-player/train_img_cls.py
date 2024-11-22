from ultralytics import YOLO

def train_image_classification():
    DATA_PATH = "brotato-cls"

    # Load a model
    model = YOLO("yolo11-cls.yaml")  # build a new model from YAML
    # model = YOLO("yolo11-cls.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data=DATA_PATH, epochs=100, imgsz=640)

    # Export the model
    path = model.export(format="onnx")
    print(f"model export to: {path}")

def predict(model_path, source):
    model = YOLO(model_path)

    # Predict with the model
    results = model(source)  # predict on an image

    # Process results list
    for result in results:
        probs = result.probs  # Probs object for classification outputs

        # ultralytics.engine.results.Probs object with attributes:
        #
        # data: tensor([2.2521e-05, 4.7274e-06, 3.6310e-09, 2.0759e-05, 3.6600e-09, 5.4356e-08, 6.3017e-10, 1.2820e-08, 1.0162e-05, 9.9993e-01, 9.2681e-06], device='cuda:0')
        # orig_shape: None
        # shape: torch.Size([11])
        # top1: 9
        # top1conf: tensor(0.9999, device='cuda:0')
        # top5: [9, 0, 3, 8, 10]
        # top5conf: tensor([9.9993e-01, 2.2521e-05, 2.0759e-05, 1.0162e-05, 9.2681e-06], device='cuda:0')
        print(probs)

        result.show()  # display to screen
        # result.save(filename="result.jpg")  # save to disk

if __name__ == '__main__':
    train_image_classification()

    # model_path = "runs/classify/train/weights/best.onnx"
    # source = "datasets/brotato-cls/test/05_WAVE_END/000589.jpg"
    # predict(model_path, source)
