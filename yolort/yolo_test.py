import argparse
from yolort.models import Darknet
from yolort.utils.datasets import LoadImages
from yolort.utils.utils import non_max_suppression, scale_coords, load_classes
import torch


def detect():
    # (320, 192) or (416, 256) or (608, 352) for (height, width)
    img_size, out, source, weights, half, device = opt.img_size, opt.output, opt.source, opt.weights, opt.half, opt.device

    # Initialize model
    model = Darknet(opt.cfg, img_size)
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
    model.to(device).eval()

    # Half precision
    half = half
    if half:
        model.half()

    # Set Dataloader
    # torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadImages(source, img_size=img_size)

    # Get names and colors
    names = load_classes(opt.names)

    # Warm up:
    _ = model(torch.zeros((1, 3, img_size, img_size), device=device))

    # Run inference
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=opt.augment)[0]

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolort/cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='yolort/data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='yolort/weights/yolov3-spp-ultralytics.pt', help='weights path')
    parser.add_argument('--source', type=str, default='yolort/data/samples', help='source')
    parser.add_argument('--output', type=str, default='yolort/output', help='output folder')
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()

    opt.device = 'cuda:0'
    opt.source = '/home/darkalert/KazendiJob/Data/HoloVideo/Data/frames/person_7/light-100_temp-5601/garments_3/rotation/cam5/image-00151.jpeg'

    with torch.no_grad():
        detect()
