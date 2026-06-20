import cv2
import numpy as np
import supervision as sv


from rfdetr import RFDETRMedium, RFDETRSegMedium
from rfdetr.assets.coco_classes import COCO_CLASSES
from typing import Optional, Any

image_path = "E:/Git/LabelPaw/images/微信图片_20260619181646_25_66.jpg"


def detection():
    """
    目标检测函数
    """
    model = RFDETRMedium()

    detections: sv.Detections | Any = model.predict(image_path, threshold=0.5)

    class_ids = []

    if detections.class_id is not None:
        class_ids = detections.class_id

    labels = [f"{COCO_CLASSES[class_id]}" for class_id in class_ids]

    annotated_image = sv.BoxAnnotator().annotate(
        detections.metadata["source_image"], detections
    )

    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
    annotated_image = np.array(annotated_image)

    cv2.imshow("Object Detection", cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def segmentation():
    """
    示例分割函数
    """

    model = RFDETRSegMedium()

    detections: sv.Detections | Any = model.predict(image_path, threshold=0.5)

    class_ids = []

    if detections.class_id is not None:
        class_ids = detections.class_id

    labels = [f"{COCO_CLASSES[class_id]}" for class_id in class_ids]

    annotated_image = sv.MaskAnnotator().annotate(
        detections.metadata["source_image"], detections
    )
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

    annotated_image = np.array(annotated_image)

    cv2.imshow("Instance Segmentation", cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detection()
    segmentation()