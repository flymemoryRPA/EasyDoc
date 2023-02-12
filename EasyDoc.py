import os
from pathlib import Path

import pandas as pd
from paddleocr import PaddleOCR

from pdf2image import convert_from_path

import cv2
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from sentence_transformers import SentenceTransformer, util
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import easyocr
import spacy


class EasyDoc:
    def __init__(self, file_path, lang='en', page=1, temp_folder='tmp', tmp_prefix='image',
                 poppler_path="poppler-0.68.0/bin"):

        self.df_nearby_below = None
        self.df_nearby_above = None
        self.df_nearby = None
        self.df_crop = None
        self.ocr_img_path = None
        self.extraction = None
        self.width = None
        self.height = None
        self.ocr_result = None
        self.region = (0, 0, 0, 0)
        self.file_path = file_path
        self.lang = lang
        self.page = page
        self.temp_folder = temp_folder
        self.tmp_prefix = tmp_prefix
        self.poppler_path = poppler_path
        self.model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')

        if not Path(file_path).exists:
            raise ('File not found: ' + file_path)
        self.ocr = PaddleOCR(use_angle_cls=True, lang=self.lang)

    def reset_region(self):
        self.region = (0, 0, self.width, self.height)

    def get_ocr_result(self):
        if not Path(self.temp_folder).is_dir():
            Path(self.temp_folder).mkdir()
            print('Created temp folder ' + self.temp_folder)
        if Path(self.file_path).suffix.lower() in ['.jpg', '.png', '.jpeg', '.gif', '.tiff']:
            ocr_img_path = os.path.abspath(self.file_path)
            print(ocr_img_path)
        else:
            pages = convert_from_path(self.file_path, poppler_path=self.poppler_path)
            pages[self.page - 1].save(f'{self.temp_folder}\\{self.tmp_prefix}_{self.page}.png', 'PNG')
            ocr_img_path = f'{self.temp_folder}\\{self.tmp_prefix}_{self.page}.png'
            print(ocr_img_path)

        try:
            self.ocr_img_path = ocr_img_path
            im = cv2.imread(ocr_img_path)
            self.width, self.height, c = im.shape
            self.region = (0, 0, self.width, self.height)

            result = self.ocr.ocr(ocr_img_path, cls=True)
            df = pd.DataFrame(result[0], columns=['bboxes', 'words'])
            df['top_left_x'] = df['bboxes'].apply(lambda x: x[0][0])
            df['top_left_y'] = df['bboxes'].apply(lambda x: x[0][1])
            df['top_right_x'] = df['bboxes'].apply(lambda x: x[1][0])
            df['top_right_y'] = df['bboxes'].apply(lambda x: x[0][1])
            df['bottom_right_x'] = df['bboxes'].apply(lambda x: x[2][0])
            df['bottom_right_y'] = df['bboxes'].apply(lambda x: x[2][1])
            df['bottom_left_x'] = df['bboxes'].apply(lambda x: x[3][0])
            df['bottom_left_y'] = df['bboxes'].apply(lambda x: x[3][1])
            df['confidence'] = df['words'].apply(lambda x: x[1])
            df['words'] = df['words'].apply(lambda x: x[0])
            df = df.drop('bboxes', axis=1)
            sentences = list(df.iloc[:]['words'])
            sentences_lower = list(df.iloc[:]['words'].str.lower())
            sentences_lower_trim = list(df['words'].str.replace(' ', '').str.lower())
            embeddings = self.model.encode(sentences)
            embeddings_lower = self.model.encode(sentences_lower)
            embeddings_lower_trim = self.model.encode(sentences_lower_trim)
            df['embedding'] = [*embeddings]
            df['embeddings_lower'] = [*embeddings_lower]
            df['embeddings_lower_trim'] = [*embeddings_lower_trim]
            self.ocr_result = df
            df.to_csv(f'{self.temp_folder}\\{self.tmp_prefix}_{self.page}.csv',
                      columns=['words', 'top_left_x', 'top_left_y', 'bottom_right_x', 'bottom_right_y', 'confidence'])
            return df
        except Exception as e:
            raise ('Failed to extract text from path: ' + ocr_img_path + ', err msg: ' + str(e))

    def find_text(self, keyword, fuzzy=0.65, position='top', first_only=True):
        df = self.ocr_result
        df['text_contains'] = df.apply(lambda x: keyword.lower() in x['words'].lower(), axis=1)
        keyword_embedding = self.model.encode(keyword)
        keyword_embedding_lower = self.model.encode(keyword.lower())
        keyword_embedding_lower_trim = self.model.encode(keyword.replace(' ', '').lower())
        df['fuzzy_matching'] = df.apply(
            lambda x: util.pytorch_cos_sim(x['embedding'], keyword_embedding).item(), axis=1)
        df['fuzzy_matching_lower'] = df.apply(
            lambda x: util.pytorch_cos_sim(x['embeddings_lower'], keyword_embedding_lower).item(), axis=1)
        df['fuzzy_matching_lower_trim'] = df.apply(
            lambda x: util.pytorch_cos_sim(x['embeddings_lower_trim'], keyword_embedding_lower_trim).item(), axis=1)

        df_fuzzy = df[
            df['text_contains'] | df['fuzzy_matching_lower'].ge(float(fuzzy)) | df['fuzzy_matching_lower_trim'].ge(
                float(fuzzy))]
        if position == 'top':
            df_fuzzy.sort_values(by=['fuzzy_matching_lower_trim', 'top_left_y'], inplace=True, ascending=[False, True])
        if position == 'bottom':
            df_fuzzy.sort_values(by=['fuzzy_matching_lower_trim', 'top_left_y'], inplace=True, ascending=[False, False])
        if position == 'left':
            df_fuzzy.sort_values(by=['fuzzy_matching_lower_trim', 'top_left_x'], inplace=True, ascending=[False, True])
        if position == 'right':
            df_fuzzy.sort_values(by=['fuzzy_matching_lower_trim', 'top_left_x'], inplace=True, ascending=[False, False])

        print(df_fuzzy)
        if first_only:
            return df_fuzzy.iloc[0][:]
        else:
            return df_fuzzy

    def set_region(self, element=None, text=None, relation='above', offset=0, position='whole'):
        if text:
            try:
                element = self.find_text(keyword=text)
            except:
                raise ('Failed to find text: ' + text)
        top_left_x = element.top_left_x
        top_left_y = element.top_left_y
        bottom_right_x = element.bottom_right_x
        bottom_right_y = element.bottom_right_y
        a, b, c, d = 0, 0, 0, 0
        if relation == 'above':
            if position == 'whole':
                a, b, c, d = 0, 0, self.width, top_left_y - offset
            if position == 'bottom':
                a, b, c, d = 0, 0, self.width, bottom_right_y - offset
        if relation == 'below':
            if position == 'whole':
                a, b, c, d = 0, bottom_right_y + offset, self.width, self.height
            if position == 'top':
                a, b, c, d = 0, top_left_y + offset, self.width, self.height
        if relation == 'left':
            if position == 'whole':
                a, b, c, d = 0, 0, top_left_x - offset, self.height
            if position == 'right':
                a, b, c, d = 0, 0, bottom_right_x - offset, self.height
        if relation == 'right':
            if position == 'whole':
                a, b, c, d = bottom_right_x + offset, 0, self.width, self.height
            if position == 'left':
                a, b, c, d = top_left_x + offset, 0, self.width, self.height

        self.region = (float(max(self.region[0], a)), float(max(self.region[1], b)),
                       float(min(self.region[2], c)), float(min(self.region[3], d)))

        df = self.ocr_result
        df_region = df[df.apply(lambda x: (x['top_left_x'] >= self.region[0]) & (x['top_left_y'] >= self.region[1]) & (
                    x['bottom_right_x'] <= self.region[2]) & (x['bottom_right_y'] <= self.region[3]), axis=1)]
        self.extraction = df_region
        return df_region

    def on_the_same_row(self, element=None, text=None, offset=(0, 0), relation=None):
        if text:
            try:
                element = self.find_text(text)
            except:
                raise ('Failed to find text: ' + text)
        top_left_x = element.top_left_x
        top_left_y = element.top_left_y
        bottom_right_x = element.bottom_right_x
        bottom_right_y = element.bottom_right_y

        a, b, c, d = 0, top_left_y - offset[0], self.width, bottom_right_y + offset[1]

        if relation == 'left':
            c = top_left_x - offset[2]
        if relation == 'right':
            a = bottom_right_x + offset[0]

        self.region = (float(max(self.region[0], a)), float(max(self.region[1], b)),
                       float(min(self.region[2], c)), float(min(self.region[3], d)))

    def on_the_same_column(self, element=None, text=None, offset=(0, 0), relation=None):
        if text:
            try:
                element = self.find_text(text)
            except:
                raise ('Failed to find text: ' + text)
        top_left_x = element.top_left_x
        top_left_y = element.top_left_y
        bottom_right_x = element.bottom_right_x
        bottom_right_y = element.bottom_right_y

        a, b, c, d = top_left_x - offset[0], 0, bottom_right_x + offset[1], self.height

        if relation == 'above':
            d = top_left_y - offset[0]
        if relation == 'below':
            b = bottom_right_y + offset[1]

        self.region = (float(max(self.region[0], a)), float(max(self.region[1], b)),
                       float(min(self.region[2], c)), float(min(self.region[3], d)))

    def get_nearyby_paragraph(self, element, w, h, separator):
        top_left_x = element.top_left_x
        top_left_y = element.top_left_y
        bottom_right_x = element.bottom_right_x
        bottom_right_y = element.bottom_right_y
        df = self.ocr_result
        df_below = df[
            df.apply(lambda x: (abs(x.top_left_x - top_left_x) <= w) & (x.top_left_y >= bottom_right_y) & ((x.top_left_y - bottom_right_y) <= h) & (x.top_left_y < self.region[3]) & (x.words != element.words), axis=1)]
        if len(df_below) > 0 and self.df_nearby_below:
            df_below = df_below.iloc[0][:]
            if abs(df_below.bottom_right_x - self.df_nearby.bottom_right_x) > w:
                self.df_nearby_below = False
            self.df_nearby.bottom_right_y = df_below.bottom_right_y
            self.df_nearby.words = self.df_nearby.words + separator + df_below.words
            return self.get_nearyby_paragraph(element=self.df_nearby, w=w, h=h, separator=separator)

        df_above = df[
            df.apply(lambda x: (abs(x.top_left_x - top_left_x) <= w) & (x.bottom_right_y <= top_left_y) & (
                        (top_left_y - x.bottom_right_y) <= h) & (x.bottom_right_y > self.region[1]) & (x.words != element.words), axis=1)]
        if len(df_above) > 0 and self.df_nearby_above:
            df_above = df_above.iloc[-1][:]
            if abs(df_above.bottom_right_x - self.df_nearby.bottom_right_x) <= w:
                self.df_nearby.top_left_y = df_above.top_left_y
                self.df_nearby.top_left_x = (self.df_nearby.top_left_x + df_above.top_left_x) / 2
                self.df_nearby.words = df_above.words + separator + self.df_nearby.words
                return self.get_nearyby_paragraph(element=self.df_nearby, w=w, h=h, separator=separator)
            else:
                self.df_nearby_above = False
        if self.df_nearby_below == False and self.df_nearby_above == False:
            print('Paragraph: ', self.df_nearby.words)
            return self.df_nearby[0][:]

    def get_paragraph(self, element=None, w=100, h=100, separator=' '):
        self.df_nearby_below = True
        self.df_nearby_above = True
        if element is None:
            element = self.get_df_from_region().iloc[0][:]
        self.df_nearby = element
        #print(element.words)
        self.get_nearyby_paragraph(element, w=w, h=h, separator=separator)
        return self.df_nearby.words

    def get_df_from_region(self):
        df = self.ocr_result
        df_region = df[df.apply(lambda x: (x['top_left_x'] >= self.region[0]) & (x['top_left_y'] >= self.region[1]) & (x['bottom_right_x'] <= self.region[2]) & (x['bottom_right_y'] <= self.region[3]), axis=1)]
        if len(df) > 0:
            return df_region
        else:
            raise ('Cannot get OCR text within region: ', self.region)

    def get_text_from_region(self):
        df = self.ocr_result
        df_region = df[df.apply(lambda x: (x['top_left_x'] >= self.region[0]) & (x['top_left_y'] >= self.region[1]) & (
                    x['bottom_right_x'] <= self.region[2]) & (x['bottom_right_y'] <= self.region[3]), axis=1)]
        if len(df_region) > 0:
            return ' '.join(df_region.words)
        else:
            print(self.region)
            raise Exception('Cannot get OCR text within region')

    def nlp(self, text=None, element=None):
        nlp = None
        if self.lang == 'en':
            nlp = spacy.load('en_core_web_trf')
        if self.lang == 'ch':
            nlp = spacy.load('zh_core_web_trf')
        if element:
            text = self.get_text_from_region()
        doc = nlp(text)
        nlp_analysis = pd.DataFrame({}, columns=['Text', 'Label', 'Start', 'End'])
        for i in doc.ents:
            nlp_analysis = nlp_analysis.append(
                {'Text': i.text,
                 'Start': i.start_char,
                 'End': i.end_char,
                 'Label': i.label_}, ignore_index=True)
        return nlp_analysis

    def get_entity(self, text=None, element=None, label='DATE'):
        nlp_analysis = self.nlp(text=text, element=element)
        df_entity = nlp_analysis[nlp_analysis.apply(lambda x: x['Label'] == label, axis=1)]
        return df_entity

    '''
    def find_paragraphs(self,min_words=10):
        df = self.ocr_result
        for index,item in df.iterrows():
            df_nearby = 
    '''

    def extract_ocr(self, engine='PaddleOCR'):
        im = cv2.imread(self.ocr_img_path)
        region = im[int(self.region[1]):int(self.region[3]), int(self.region[0]):int(self.region[2])]
        crop_image = os.path.abspath(f'{self.temp_folder}\\target_region.png')
        print(crop_image)
        cv2.imwrite(crop_image, region)
        if engine == 'PaddleOCR':
            try:
                result = self.ocr.ocr(crop_image)
                df_crop = pd.DataFrame(result[0], columns=['bboxes', 'words'])
                df_crop['confidence'] = df_crop['words'].apply(lambda x: x[1])
                df_crop['words'] = df_crop['words'].apply(lambda x: x[0])
                df_crop = df_crop.drop('bboxes', axis=1)
                self.df_crop = df_crop
                print(df_crop)
                return df_crop
            except:
                return None
        if engine == 'EasyOCR':
            if self.lang == 'en':
                reader = easyocr.Reader([self.lang])
            if self.lang == 'ch':
                reader = easyocr.Reader(['ch_sim', 'en'])
            if self.lang == 'cht':
                reader = easyocr.Reader(['ch_tra', 'en'])
            try:
                result = reader.readtext(crop_image)
                df_crop = pd.DataFrame(result, columns=['bboxes', 'words', 'confidence'])
                df_crop = df_crop.drop('bboxes', axis=1)
                self.df_crop = df_crop
                print(df_crop)
                return df_crop
            except:
                return None
        if engine == 'TrOCR-handwritten':
            try:
                image = Image.open(crop_image).convert("RGB")
                processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
                model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
                pixel_values = processor(images=image, return_tensors="pt").pixel_values
                generated_ids = model.generate(pixel_values)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                df_crop = pd.DataFrame({'words': generated_text, 'confidence': None}, index=[0])
                self.df_crop = df_crop
                print(df_crop)
                return df_crop
            except:
                return None

    def draw_region(self, label=None, show_image=False):
        start_point = (int(self.region[0]), int(self.region[1]))
        end_point = (int(self.region[2]), int(self.region[3]))
        color = (255, 0, 0)
        thickness = 2
        im = cv2.imread(self.ocr_img_path)
        image = cv2.rectangle(im, start_point, end_point, color, thickness)

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (int(self.region[2]) + 5, int(self.region[3]) - 10)
        fontScale = 1
        fontColor = color
        thickness = 3
        lineType = 2
        try:
            if label:
                text = label + ": " + self.df_crop.iloc[0][:].words
            else:
                text = self.df_crop.iloc[0][:].words
        except:
            text = 'not found'

        cv2.putText(image, text, bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)
        cv2.imwrite(f'{self.temp_folder}\\output.png', image)
        print(os.path.abspath(f'{self.temp_folder}\\output.png'))
        if show_image:
            dim = None
            (h, w) = image.shape[:2]

            width = 1500
            r = width / float(w)
            dim = (width, int(h * r))
            resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

            cv2.imshow('Image', resized_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
