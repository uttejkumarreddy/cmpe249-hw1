_base_ = "./cascade_mask_rcnn_r50_fpn_1x_coco.py"

dataset_type = "CocoDataset"

classes = (
    "Pedestrian",
    "Truck",
    "Car",
    "Cyclist",
    "DontCare",
    "Misc",
    "Van",
    "Tram",
    "Person_sitting",
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file="D:\\present\\cmpe249-hw1\\dataset\\coco-annotations\\annotations_train.json",
        img_prefix="D:\\present\\cmpe249-hw1\\dataset\\images\\train",
    ),
    val=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file="D:\\present\\cmpe249-hw1\\dataset\\coco-annotations\\annotations_test.json",
        img_prefix="D:\\present\\cmpe249-hw1\\dataset\\images\\test",
    ),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file="D:\\present\\cmpe249-hw1\\dataset\\coco-annotations\\annotations_test.json",
        img_prefix="D:\\present\\cmpe249-hw1\\dataset\\images\\test",
    ),
)

model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type="Shared2FCBBoxHead",
                # explicitly over-write all the `num_classes` field from default 80 to 5.
                num_classes=8,
            ),
            dict(
                type="Shared2FCBBoxHead",
                # explicitly over-write all the `num_classes` field from default 80 to 5.
                num_classes=8,
            ),
            dict(
                type="Shared2FCBBoxHead",
                # explicitly over-write all the `num_classes` field from default 80 to 5.
                num_classes=8,
            ),
        ],
        # explicitly over-write all the `num_classes` field from default 80 to 5.
        mask_head=dict(num_classes=8),
    )
)
