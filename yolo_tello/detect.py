"""Detection module"""
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import cv2
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.download_weights import download
from utils.general import check_img_size, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, set_logging, increment_path, strip_optimizer
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, TracedModel


def detect(args):
    """Detection function"""
    global modelc
    source, weights, view_img, save_txt, imgsz, trace, blur = \
        (args.source, args.weights,
         args.view_img, args.save_txt,
         args.img_size, not args.no_trace, args.blur
         )
    save_img = not args.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(args.project) /
                                   args.name, exist_ok=args.exist_ok)
                    )  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(args.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, args.img_size)

    if half:
        model.half()  # to FP16

    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        (modelc.load_state_dict(
            torch.load('weights/resnet101.pt', map_location=device)['model'])
         .to(device).eval())

    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        if (device.type != 'cpu' and
                (old_img_b != img.shape[0] or
                 old_img_h != img.shape[2] or
                 old_img_w != img.shape[3])
        ):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                _ = model(img, augment=args.augment)[0]

        pred = model(img, augment=args.augment)[0]

        pred = non_max_suppression(pred,
                                   args.conf_thres,
                                   args.iou_thres,
                                   classes=args.classes,
                                   agnostic=args.agnostic_nms
                                   )

        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        for i, det in enumerate(pred):
            if webcam:
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = (str(save_dir / 'labels' / p.stem) +
                        ('' if dataset.mode == 'image' else f'_{frame}'))
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                for *xyxy, conf, cls in reversed(det):

                    if blur:
                        crop_obj = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                        blur = cv2.blur(crop_obj, (60, 60))
                        im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])] = blur

                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if args.save_conf else (cls, *xywh)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(
                            xyxy,
                            im0,
                            label=label,
                            color=colors[int(cls)],
                            line_thickness=1
                        )

            if view_img:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    raise StopIteration

            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 10, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(
                            save_path,
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            fps,
                            (w, h)
                        )
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = (f"\n{len(list(save_dir.glob('labels/*.txt')))} "
             f"labels saved to {save_dir / 'labels'}") if save_txt else ''
        print(f"Results saved to {save_dir}{s}")


def parse_args():
    """Parsing arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--download', action='store_true',
                        help='download model weights automatically', default=True)
    parser.add_argument('--no-download', dest='download',
                        action='store_false')
    parser.add_argument('--source', type=str, default='0',
                        help='source')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                        help='IOU threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true',
                        help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3', default=0)
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='object_tracking',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true',
                        help='don`t trace model')
    parser.add_argument('--blur', action='store_true',
                        help='blur detections')
    parser.set_defaults(download=True)

    return parser.parse_args()


def main() -> int:
    """Main function"""
    args = parse_args()
    if args.download and not os.path.exists(str(args.weights)):
        print('Model weights not found. Attempting to download now...')
        download('./')

    with torch.no_grad():
        if args.update:
            for args.weights in ['yolov7.pt']:
                detect(args)
                strip_optimizer(args.weights)
        else:
            detect(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
