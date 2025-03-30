import keyboard
import win32gui
import win32con
import win32api
from pynput.mouse import Controller

import numpy as np
import time

import cv2
from mss import mss
from ultralytics import YOLO

# Due to different screen sizes and slight deviations in obtaining the CSGO window with win32gui,
# updating CSGO_WIN_WH in real-time may lead to inaccurate aiming.
# Please adjust CSGO_WIN_WH to the current CSGO window size in advance.
CSGO_WIN_WH = (1440, 900)
# This is the image size passed to the model, currently set to a 16:10 ratio, matching CSGO_WIN_WH.
# If your screen is 16:9, please adjust it to (768, 432) in advance.
INFERENCE_WH = (768, 480)
SCALE_RATIO = CSGO_WIN_WH[0] / INFERENCE_WH[0]

mouse = Controller()

# Load the YOLO model as before
model = YOLO('./models/v8s_300_epoch_prune_finetune.pt')

# classes = {0: "ct_body", 1: "ct_head", 2: t_body, 3: "t_head"}
detection_modes = {
    "CT Camp": {"classes": [2, 3]},
    "T Camp": {"classes": [0, 1]},
    "Solo": {"classes": [0, 1, 2, 3]}
}
current_mode = detection_modes["Solo"]
mode_type = "Solo"
auto_shot = False
auto_mouse_move = True

# keyboard_switch_camp = {'f5': "CT Camp", 'f6': "T Camp", 'f7': "Solo", "f8": "Auto Mouse Move", "f9": "Auto Shot"}
def switch_detection_mode(event):
    global current_mode, mode_type, auto_shot, auto_mouse_move
    if event.name == 'f5':
        mode_type = "CT Camp"
        current_mode = detection_modes[mode_type]
        print(f"Switching team: {mode_type}")
    elif event.name == 'f6':
        mode_type = "T Camp"
        current_mode = detection_modes[mode_type]
        print(f"Switching team: {mode_type}")
    elif event.name == 'f7':
        mode_type = "Solo"
        current_mode = detection_modes[mode_type]
        print(f"Switching team: {mode_type}")
    elif event.name == 'f8':
        if auto_mouse_move:
            print("Disabling auto mouse movement")
            auto_mouse_move = False
        else:
            print("Enabling auto mouse movement")
            auto_mouse_move = True
    elif event.name == 'f9':
        if auto_shot:
            print("Disabling auto shooting")
            auto_shot = False
        else:
            print("Enabling auto shooting")
            auto_shot = True

def find_csgo_window():
    # global SCALE_RATIO, CSGO_WIN_WH
    hwnd = win32gui.FindWindow(None, "Counter-Strike 2")
    if hwnd == 0:
        print("No running CS:GO window detected")
        exit(0)

    if hwnd != 0 and win32gui.IsWindowVisible(hwnd):
        rect = win32gui.GetWindowRect(hwnd)
        if rect[0] != -32000 and rect[1] != -32000:
            # CSGO_WIN_WH = rect[2] - rect[0], rect[3] - rect[1]
            # SCALE_RATIO = CSGO_WIN_WH[0] / INFERENCE_WH[0]
            # print("##################################################")
            # print(CSGO_WIN_WH)
            # print(SCALE_RATIO)
            # print("##################################################")
            print(f"Detected CS:GO window information at the forefront: {rect}")
            return rect
        else:
            cv2.destroyAllWindows()
    return None

def screenshot():
    rect = find_csgo_window()
    if rect is None:
        return
    with mss() as sct:
        monitor = {"top": rect[1], "left": rect[0], "width": rect[2] - rect[0], "height": rect[3] - rect[1]}
        img = sct.grab(monitor)
        img_np = np.array(img)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
        img_np = cv2.resize(img_np, (INFERENCE_WH[0], INFERENCE_WH[1]))

        detect(img_np)
        print("-------------------------------------------------------------------------------------------------")

def mouse_move(boxes):
    boxes_distance = []
    rect = find_csgo_window()
    if rect is None:
        return

    win_left, win_top, win_right, win_bottom = rect[0], rect[1], rect[2], rect[3]
    current_x, current_y = mouse.position
    if current_x <= win_left or current_x >= win_right or current_y <= win_top or current_y >= win_bottom:
        print("Target detected, please move the mouse within the CS:GO window.")
        return

    start_time = time.time()
    for box in boxes:
        cls = int(box.cls.tolist()[0])
        if cls == 0 or cls == 3:
            continue
        img_box_xyxy = box.xyxy.tolist()[0]
        scale_box_xyxy = [i * SCALE_RATIO for i in img_box_xyxy]
        projection_box_xyxy = [
            win_left + scale_box_xyxy[0],
            win_top + scale_box_xyxy[1],
            win_left + scale_box_xyxy[2],
            win_top + scale_box_xyxy[3]
        ]
        projection_box_middle = ((projection_box_xyxy[0] + projection_box_xyxy[2]) / 2,
                                 (projection_box_xyxy[1] + projection_box_xyxy[3]) / 2)
        euclidian_distance = (current_x - projection_box_middle[0]) ** 2 + (current_y - projection_box_middle[1]) ** 2
        boxes_distance.append([cls, projection_box_xyxy, projection_box_middle, euclidian_distance])

    if boxes_distance:
        target_x = 0
        target_y = 0
        boxes_distance.sort(key=lambda x: x[3])
        if boxes_distance[0][0] == 2 or boxes_distance[0][0] == 5:
            target_x = int(boxes_distance[0][2][0])
            target_y = int(boxes_distance[0][2][1])
        else:
            if len(boxes_distance) > 1:
                if boxes_distance[1][0] == 2 or boxes_distance[1][0] == 5:
                    target_x = int(boxes_distance[1][2][0])
                    target_y = int(boxes_distance[1][2][1])
            else:
                target_x = int(boxes_distance[0][2][0])
                target_y = int(boxes_distance[0][1][1])

        if not target_x and not target_y:
            _ = time.time()
            return
        if auto_mouse_move:
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, target_x - current_x + 8, target_y - current_y + 30, 0, 0)
        if auto_shot:
            time.sleep(0.5)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    end_time = time.time()
    period = end_time - start_time
    print(f"Calculating bounding boxes mapping and moving the mouse took {period} seconds")

def detect(img):
    infer_region = img
    infer_region = np.ascontiguousarray(infer_region, dtype=np.uint8)
    results = model(infer_region, classes=current_mode["classes"],
                    conf=0.65, iou=0.7, imgsz=(INFERENCE_WH[1], INFERENCE_WH[0]),
                    device=0, agnostic_nms=True)
    r = results[0]
    img_ss = r.plot()
    cv2.imshow("CS:GO Screenshot", img_ss)

    if r.boxes:
        mouse_move(r.boxes)

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        exit(0)

if __name__ == '__main__':
    keyboard.on_press(switch_detection_mode)
    while True:
        screenshot()