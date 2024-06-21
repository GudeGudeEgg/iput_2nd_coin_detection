import csv
import os
import sys
from pathlib import Path
import torch
import gradio as gr

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_img_size,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


def run(source, iou_thres, one_thres, five_thres, ten_thres, fifty_thres, hundred_thres, fivehundred_thres, classes):
    coin_classes = []
    coin_dict = {
        "1": 0,
        "5": 1,
        "10": 2,
        "50": 3,
        "100": 4,
        "500": 5
    }
    for i in classes:
        coin_classes.append(coin_dict[str(i)])

    weights = r"C:\Users\ok230116\Desktop\Python\人工知能\yolov5-master\runs\train\outdir11\weights\best.pt"
    data = ROOT / "data/coco128.yaml"  # dataset.yaml path
    imgsz = (640, 640)  # inference size (height, width)
    max_det = 1000  # maximum detections per image
    device = ""  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img = False  # show results
    save_txt = False  # save results to *.txt
    save_csv = False  # save results in CSV format
    save_conf = False  # save confidences in --save-txt labels
    save_crop = False  # save cropped prediction boxes
    nosave = False  # do not save images/videos
    agnostic_nms = False  # class-agnostic NMS
    augment = False  # augmented inference
    visualize = False  # visualize features
    update = False  # update all models
    project = ROOT / "runs/detect"  # save results to project/name
    name = "exp"  # save results to project/name
    exist_ok = False  # existing project/name ok, do not increment
    line_thickness = 10  # bounding box thickness (pixels)
    hide_labels = False  # hide labels
    hide_conf = False  # hide confidences
    half = False  # use FP16 half-precision inference
    dnn = False  # use OpenCV DNN for ONNX inference
    vid_stride = 1  # video frame-rate stride
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else 1, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)

        # NMS
        thres_list = [one_thres, five_thres, ten_thres, fifty_thres, hundred_thres, fivehundred_thres]
        pred_list = [0, 0, 0, 0, 0, 0]
        sub_pred_list = []
        with dt[2]:
            for j in coin_classes:
                i = int(j)
                sub_pred = non_max_suppression(pred, thres_list[i], iou_thres, [i], agnostic_nms, max_det=max_det)
                if len(sub_pred) > 0:
                    pred_list[i] = sub_pred
            for sub in pred_list:
                if sub != 0:
                    sub_pred_list.append(sub[0])
            if sub_pred_list:
                pred = [torch.cat(sub_pred_list, dim=0)]
        print(pred)
        pred_0 = pred[0]
        print(pred[0])
        coin_num = [0, 0, 0, 0, 0, 0]
        value_list = [1, 5, 10, 50, 100, 500]
        total = 0

        for i in pred_0:
            coin_pred = int(i[5].item())
            coin_num[coin_pred] += 1
        for i in range(6):
            total += coin_num[i] * value_list[i]

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "%gx%g " % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

            # Save results (image with detections)
            if save_img:
                cv2.imwrite(save_path, im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

    return save_path, f"{coin_num[0]}枚", f"{coin_num[1]}枚", f"{coin_num[2]}枚", f"{coin_num[3]}枚", f"{coin_num[4]}枚", f"{coin_num[5]}枚", f"{total}円"


def gui():
    with gr.Blocks() as detection:
        # 画面の配置を決定する手段をGPT4-oに聞きました
        with gr.Row():
            with gr.Column(scale=2):
                image = gr.Image(label="Upload Image", type="filepath")

                iou_thres_input = gr.Slider(minimum=0., maximum=1., label="NMSの閾値")
                one_thres_input = gr.Slider(minimum=0., maximum=1., label="1円の信頼閾値")
                five_thres_input = gr.Slider(minimum=0., maximum=1., label="5円の信頼閾値")
                ten_thres_input = gr.Slider(minimum=0., maximum=1., label="10円の信頼閾値")
                fifty_thres_input = gr.Slider(minimum=0., maximum=1., label="50円の信頼閾値")
                hundred_thres_input = gr.Slider(minimum=0., maximum=1., label="100円の信頼閾値")
                fivehundred_thres_input = gr.Slider(minimum=0., maximum=1., label="500円の信頼閾値")

                classes_input = gr.CheckboxGroup(choices=[1, 5, 10, 50, 100, 500],
                                                 label="硬貨選択")
                execution_btn = gr.Button("検出")

            with gr.Column(scale=1):
                display = gr.Image(label="", type="filepath", interactive=False)
                one_coin_count = gr.Textbox(label="1円硬貨の枚数", interactive=False)
                five_coin_count = gr.Textbox(label="5円硬貨の枚数", interactive=False)
                ten_coin_count = gr.Textbox(label="10円硬貨の枚数", interactive=False)
                fifty_coin_count = gr.Textbox(label="50円硬貨の枚数", interactive=False)
                hundred_coin_count = gr.Textbox(label="100円硬貨の枚数", interactive=False)
                fivehundred_coin_count = gr.Textbox(label="500円硬貨の枚数", interactive=False)
                total_count = gr.Textbox(label="合計金額", interactive=False)

        execution_btn.click(fn=run,
                            inputs=[image,
                                    iou_thres_input,
                                    one_thres_input,
                                    five_thres_input,
                                    ten_thres_input,
                                    fifty_thres_input,
                                    hundred_thres_input,
                                    fivehundred_thres_input,
                                    classes_input],
                            outputs=[display,
                                     one_coin_count,
                                     five_coin_count,
                                     ten_coin_count,
                                     fifty_coin_count,
                                     hundred_coin_count,
                                     fivehundred_coin_count,
                                     total_count]
                            )
    detection.launch()


if __name__ == "__main__":
    gui()
