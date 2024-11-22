from rapidocr_onnxruntime import RapidOCR

class OCR:
    def __init__(self):
        self.engine = RapidOCR()

    # RapidOCR use_det=True results:
    # [
    #   [[[54.0, 28.0], [81.0, 28.0], [81.0, 36.0], [54.0, 36.0]], '11/19', 0.99749345]
    # ]
    #
    # RapidOCR use_det=False results:
    # [['20 / 20', 0.9370327]]
    def recognize(self, image, use_det=False):
        return self.engine(image, use_det, use_cls=False, use_rec=True)
