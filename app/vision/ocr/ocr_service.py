import os
import time
import numpy as np
import cv2
from typing import List, Tuple

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

from app.extentions import settings


def get_character_dict(dict_path: str) -> List[str]:
    """Load character dictionary from file."""
    character_str = []
    with open(dict_path, 'rb') as fin:
        lines = fin.readlines()
        for line in lines:
            line = line.decode('utf-8').strip('\n').strip('\r\n')
            character_str += line
    return character_str


class BaseRecLabelDecode:
    """Base class for recognition label decoding."""
    
    def __init__(self, character_dict=None, character_type='ch', use_space_char=False):
        self.beg_str = 'sos'
        self.end_str = 'eos'
        if character_type == 'en':
            self.character_str = '0123456789abcdefghijklmnopqrstuvwxyz'
            dict_character = list(self.character_str)
        else:
            self.character_str = []
            self.character_str.extend(character_dict)
            if use_space_char:
                self.character_str.append(' ')
            dict_character = list(self.character_str)
        
        dict_character = self.add_special_char(dict_character)
        self.dict = {char: i for i, char in enumerate(dict_character)}
        self.character = dict_character

    def add_special_char(self, dict_character):
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """Decode text from indices."""
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate and idx > 0 and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]:
                    continue
                char_list.append(self.character[int(text_index[batch_idx][idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, float(np.mean(conf_list)) if conf_list else float('nan')))
        return result_list

    def get_ignored_tokens(self):
        return [0]


class CTCLabelDecode(BaseRecLabelDecode):
    """CTC label decoding for recognition."""
    
    def __init__(self, character_dict=None, character_type='ch', use_space_char=False, **kwargs):
        super().__init__(character_dict, character_type, use_space_char)
        self.char_mask = None

    def __call__(self, preds, label=None, *args, **kwargs):
        if self.char_mask is not None:
            preds[:, :, ~self.char_mask] = 0
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        text = [(t, float(conf)) for (t, conf) in text]
        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character


def get_rotate_crop_image(img, points: np.ndarray):
    """Rotate and crop image based on points."""
    img_crop_width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(img, M, (img_crop_width, img_crop_height), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def sorted_boxes(dt_boxes):
    """Sort detected boxes by position."""
    num_boxes = dt_boxes.shape[0]
    sorted_boxes_list = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes_list)
    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


class TextRecognizer:
    """Text recognition using ONNX model."""
    
    def __init__(self, rec_model_path=None, target_size=[3, 32, 480], dict_path=None, device_id=-1, rec_batch_num=8):
        self.device_id = device_id
        self.rec_image_shape = target_size
        self.rec_batch_num = rec_batch_num
        self.character_dict = get_character_dict(dict_path)
        self.postprocess_op = CTCLabelDecode(character_dict=self.character_dict, character_type='vn', use_space_char=True)
        
        if rec_model_path is None:
            raise ValueError('TEXT_REC_MODEL_PATH is not set')
        
        with open(rec_model_path, 'rb') as f:
            model_bytes = f.read()

        if ort is None:
            raise RuntimeError('onnxruntime is required for TextRecognizer')

        so = ort.SessionOptions()
        so.log_severity_level = 3
        
        if settings.DEVICE_GPU >= 0:
            providers = [('CUDAExecutionProvider', {'device_id': settings.DEVICE_GPU})]
        else:
            providers = ['CPUExecutionProvider']

        sess = ort.InferenceSession(model_bytes, so, providers=providers)
        self.predictor = sess
        self.input_tensor = sess.get_inputs()[0]
        self.output_tensors = None

    def resize_norm_img(self, img, max_wh_ratio):
        """Resize and normalize image for recognition."""
        imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[2]
        h, w = img.shape[:2]
        ratio = w / float(h)
        if int(np.ceil(imgH * ratio)) > imgW:
            resized_w = imgW
        else:
            resized_w = int(np.ceil(imgH * ratio))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255.0
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, img_list):
        """Recognize text from image list."""
        img_num = len(img_list)
        width_list = [img.shape[1] / float(img.shape[0]) for img in img_list]
        indices = np.argsort(np.array(width_list))
        rec_res = [['', 0.0]] * img_num
        batch_num = self.rec_batch_num
        elapse = 0
        
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]], max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()
            starttime = time.time()
            input_dict = {self.input_tensor.name: norm_img_batch}
            outputs = self.predictor.run(self.output_tensors, input_dict)
            preds = outputs[0]
            rec_result = self.postprocess_op(preds)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
            elapse += time.time() - starttime
        return rec_res, elapse


class TextDetectPdparam:
    """Text detection using PaddleOCR."""
    
    def __init__(self, model_path, unclip_ratio=2.0, box_thresh=0.3):
        if PaddleOCR is None:
            raise RuntimeError('paddleocr is required for detection')
        use_gpu = settings.DEVICE_GPU >= 0
        self.text_detector = PaddleOCR(
            det_model_dir=model_path,
            use_gpu=use_gpu,
            lang='en',
            use_angle_cls=True,
            det_db_unclip_ratio=unclip_ratio,
            det_db_box_thresh=box_thresh
        )

    def __call__(self, image=None):
        """Detect text in image."""
        start_time = time.time()
        result_text_dt = self.text_detector.ocr(image, rec=False)
        result_text_dt = np.asarray(result_text_dt)
        return result_text_dt[0], time.time() - start_time


class TextSystem:
    """Complete OCR system combining detection and recognition."""
    
    def __init__(self, rec_model_path, det_model_path_pdparam, dict_path, 
                 text_rec_targetsize, box_thresh=0.3, unclip_ratio=2.0, 
                 drop_score=0.6, device_id=-1):
        self.text_detector_pdparam = TextDetectPdparam(
            model_path=det_model_path_pdparam,
            unclip_ratio=unclip_ratio,
            box_thresh=box_thresh
        )
        self.text_recognizer = TextRecognizer(
            rec_model_path=rec_model_path,
            target_size=text_rec_targetsize,
            dict_path=dict_path,
            device_id=device_id
        )
        self.use_angle_cls = True
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio
        self.drop_score = drop_score

    def ocr(self, img, cls=True, det=True, rec=True):
        """Run OCR pipeline."""
        if det and rec:
            drop_score = self.drop_score
            ori_im = img.copy()
            dt_boxes, elapse_text_det = self.text_detector_pdparam(img)
            if dt_boxes is None:
                return [], 0, 0
            img_crop_list = []
            dt_boxes = sorted_boxes(dt_boxes)
            for bno in range(len(dt_boxes)):
                tmp_box = np.array(dt_boxes[bno], np.float32)
                img_crop = get_rotate_crop_image(ori_im, tmp_box)
                img_crop_list.append(img_crop)
            rec_res, elapse_text_rec = self.text_recognizer(img_crop_list)
            res = []
            for box, rec_result in zip(dt_boxes, rec_res):
                text, score = rec_result
                if score >= drop_score:
                    res.append([np.array(box).tolist(), rec_result])
            return [res], elapse_text_det, elapse_text_rec
        elif det and not rec:
            dt_boxes, elapse_text_det = self.text_detector_pdparam(img)
            res = []
            if dt_boxes is None:
                return res, 0, 0
            dt_boxes = sorted_boxes(dt_boxes)
            for item_box in dt_boxes:
                res.append([item_box.tolist(), ('', 0)])
            return [res], elapse_text_det, 0
        elif not det and rec:
            if type(img) != list:
                img = [img]
            rec_res, elapse_text_rec = self.text_recognizer(img)
            return [rec_res], 0, elapse_text_rec


# Global OCR system instance
_text_system = None


def get_ocr_system() -> TextSystem:
    """Get or initialize OCR system (singleton)."""
    global _text_system
    if _text_system is None:
        _text_system = initialize_ocr()
    return _text_system


def initialize_ocr() -> TextSystem:
    """Initialize OCR system with settings."""
    return TextSystem(
        rec_model_path=settings.TEXT_REC_MODEL_PATH,
        det_model_path_pdparam=settings.TEXT_DET_PDPARAM_MODEL_PATH,
        dict_path=settings.TEXT_REC_DICT_PATH,
        text_rec_targetsize=settings.rec_target_size,
        box_thresh=settings.TEXT_DET_CONF_THRESHOLD,
        drop_score=settings.TEXT_REC_CONF_THRESHOLD,
        device_id=settings.DEVICE_GPU
    )
