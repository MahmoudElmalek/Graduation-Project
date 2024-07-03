import cv2
# import time
import numpy as np

__all__ = ["vis"]


text_scale = 1
text_thickness = 1
line_thickness = 1


def plot_tracking(
    image, tlwhs, faces_id, scores=None, frame_id=0, fps=0.0, ids2=None, cards=[]):
    
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    cv2.putText(
        im,
        "fps: %.2f num: %d" % (fps, len(tlwhs)),
        (0, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        thickness=2,
    )
   
    for i, tlwh in enumerate(tlwhs):
        color=(0,0,255)
        colordist=(0,0,255)
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(faces_id[i])
        upper_text = "{}".format(int(obj_id))
        dist_text = "Dist: "

        if (obj_id) in cards:
            if cards[obj_id][6]=='UN_KNOWN':
                color=(0,0,255)
                upper_text = str(cards[obj_id][2]) + ": UnKnown"
            else:
                color=(0,255,0)
                upper_text = str(cards[obj_id][2]) + ": " + str(cards[obj_id][0])+", Score="+str((cards[obj_id][1]))+"%"
            
            dist_text=dist_text+str(round(cards[obj_id][3],2))+"cm"

            if cards[obj_id][3]<40:
                colordist=(0,255,0)
            else:
                colordist=(0,0,255)
        
        if ids2 is not None:
            upper_text = upper_text + ", {}".format(int(ids2[i]))
        
        cv2.rectangle(
            im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness
        )
        cv2.putText(
            im,
            upper_text,
            (int(x1), int(y1)),
            cv2.FONT_HERSHEY_PLAIN,
            text_scale,
            color,
        )

        cv2.putText(
            im,
            dist_text,
            (int(x1), int(y1+h)),
            cv2.FONT_HERSHEY_PLAIN,
            text_scale,
            colordist,
        )
        
    return im


# def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
#     for i in range(len(boxes)):
#         box = boxes[i]
#         cls_id = int(cls_ids[i])
#         score = scores[i]
#         if score < conf:
#             continue
#         x0 = int(box[0])
#         y0 = int(box[1])
#         x1 = int(box[2])
#         y1 = int(box[3])

#         color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
#         text = "{}:{:.1f}%".format(class_names[cls_id], score * 100)
#         txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
#         font = cv2.FONT_HERSHEY_SIMPLEX

#         txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
#         cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

#         txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
#         cv2.rectangle(
#             img,
#             (x0, y0 + 1),
#             (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
#             txt_bk_color,
#             -1,
#         )
#         cv2.putText(
#             img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1
#         )

#     return img


# _COLORS = (
#     np.array(
#         [
#             0.000,
#             0.447,
#             0.741,
#             0.850,
#             0.325,
#             0.098,
#             0.929,
#             0.694,
#             0.125,
#             0.494,
#             0.184,
#             0.556,
#             0.466,
#             0.674,
#             0.188,
#             0.301,
#             0.745,
#             0.933,
#             0.635,
#             0.078,
#             0.184,
#             0.300,
#             0.300,
#             0.300,
#             0.600,
#             0.600,
#             0.600,
#             1.000,
#             0.000,
#             0.000,
#             1.000,
#             0.500,
#             0.000,
#             0.749,
#             0.749,
#             0.000,
#             0.000,
#             1.000,
#             0.000,
#             0.000,
#             0.000,
#             1.000,
#             0.667,
#             0.000,
#             1.000,
#             0.333,
#             0.333,
#             0.000,
#             0.333,
#             0.667,
#             0.000,
#             0.333,
#             1.000,
#             0.000,
#             0.667,
#             0.333,
#             0.000,
#             0.667,
#             0.667,
#             0.000,
#             0.667,
#             1.000,
#             0.000,
#             1.000,
#             0.333,
#             0.000,
#             1.000,
#             0.667,
#             0.000,
#             1.000,
#             1.000,
#             0.000,
#             0.000,
#             0.333,
#             0.500,
#             0.000,
#             0.667,
#             0.500,
#             0.000,
#             1.000,
#             0.500,
#             0.333,
#             0.000,
#             0.500,
#             0.333,
#             0.333,
#             0.500,
#             0.333,
#             0.667,
#             0.500,
#             0.333,
#             1.000,
#             0.500,
#             0.667,
#             0.000,
#             0.500,
#             0.667,
#             0.333,
#             0.500,
#             0.667,
#             0.667,
#             0.500,
#             0.667,
#             1.000,
#             0.500,
#             1.000,
#             0.000,
#             0.500,
#             1.000,
#             0.333,
#             0.500,
#             1.000,
#             0.667,
#             0.500,
#             1.000,
#             1.000,
#             0.500,
#             0.000,
#             0.333,
#             1.000,
#             0.000,
#             0.667,
#             1.000,
#             0.000,
#             1.000,
#             1.000,
#             0.333,
#             0.000,
#             1.000,
#             0.333,
#             0.333,
#             1.000,
#             0.333,
#             0.667,
#             1.000,
#             0.333,
#             1.000,
#             1.000,
#             0.667,
#             0.000,
#             1.000,
#             0.667,
#             0.333,
#             1.000,
#             0.667,
#             0.667,
#             1.000,
#             0.667,
#             1.000,
#             1.000,
#             1.000,
#             0.000,
#             1.000,
#             1.000,
#             0.333,
#             1.000,
#             1.000,
#             0.667,
#             1.000,
#             0.333,
#             0.000,
#             0.000,
#             0.500,
#             0.000,
#             0.000,
#             0.667,
#             0.000,
#             0.000,
#             0.833,
#             0.000,
#             0.000,
#             1.000,
#             0.000,
#             0.000,
#             0.000,
#             0.167,
#             0.000,
#             0.000,
#             0.333,
#             0.000,
#             0.000,
#             0.500,
#             0.000,
#             0.000,
#             0.667,
#             0.000,
#             0.000,
#             0.833,
#             0.000,
#             0.000,
#             1.000,
#             0.000,
#             0.000,
#             0.000,
#             0.167,
#             0.000,
#             0.000,
#             0.333,
#             0.000,
#             0.000,
#             0.500,
#             0.000,
#             0.000,
#             0.667,
#             0.000,
#             0.000,
#             0.833,
#             0.000,
#             0.000,
#             1.000,
#             0.000,
#             0.000,
#             0.000,
#             0.143,
#             0.143,
#             0.143,
#             0.286,
#             0.286,
#             0.286,
#             0.429,
#             0.429,
#             0.429,
#             0.571,
#             0.571,
#             0.571,
#             0.714,
#             0.714,
#             0.714,
#             0.857,
#             0.857,
#             0.857,
#             0.000,
#             0.447,
#             0.741,
#             0.314,
#             0.717,
#             0.741,
#             0.50,
#             0.5,
#             0,
#         ]
#     )
#     .astype(np.float32)
#     .reshape(-1, 3)
# )
