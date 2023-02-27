import os
from pathlib import Path
import numpy as np
from collections import Counter

import pandas as pd
from paddleocr import PaddleOCR

import pypdfium2 as pdfium

import cv2
from PIL import Image

import pdfplumber

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from sentence_transformers import SentenceTransformer, util
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import easyocr
import spacy


class EasyDoc:
    def __init__(self, file_path, lang='en', page=1, temp_folder='tmp', tmp_prefix='image', apply_angle_rotate=False):

        self.table_h = None
        self.table_v = None
        self.average_line_space = None
        self.w = None
        self.h = None
        self.apply_ocr = None
        self.df_nearby_right = None
        self.df_nearby_left = None
        self.PaddleOCR = None
        self.engine = None
        self.df_nearby_below = None
        self.df_nearby_above = None
        self.df_nearby = None
        self.df_crop = None
        self.ocr_img_path = None
        self.extraction = None
        self.width = None
        self.height = None
        self.ocr_result = None
        self.region = None
        self.file_path = file_path
        self.lang = lang
        self.page = page
        self.temp_folder = temp_folder
        self.tmp_prefix = tmp_prefix
        self.model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')
        self.apply_angle_rotate = apply_angle_rotate

        if not Path(file_path).exists:
            raise ('File not found: ' + file_path)

        if not Path(self.temp_folder).is_dir():
            Path(self.temp_folder).mkdir()
            print('Created temp folder ' + self.temp_folder)

    def reset_region(self):
        # page_width = self.ocr_result['bottom_right_x'].max()-self.ocr_result['top_left_x'].min()
        # page_height = self.ocr_result['bottom_right_y'].max() - self.ocr_result['top_left_y'].min()
        self.region = (0, 0, self.width, self.height)

    def extract_words(self, apply_ocr=True, engine='PaddleOCR', crop_region=False, embedding=True):

        self.engine = engine
        self.apply_ocr = apply_ocr
        df = None

        if not self.ocr_img_path:

            if apply_ocr:

                if Path(self.file_path).suffix.lower() in ['.jpg', '.png', '.jpeg', '.gif', '.tiff']:
                    ocr_img_path = os.path.abspath(self.file_path)

                else:
                    pdf = pdfium.PdfDocument(self.file_path)
                    page = pdf.get_page(self.page-1)
                    pil_image = page.render_topil(
                        scale=1,  # 72dpi resolution
                        rotation=0,  # no additional rotation
                        # ... further rendering options
                    )

                    ocr_img_path = f'{self.temp_folder}/{self.tmp_prefix}_{self.page}.png'
                    pil_image.save(ocr_img_path)
                    im = cv2.imread(ocr_img_path)
                    self.height, self.width, c = im.shape
                    self.region = (0, 0, self.width, self.height)

                print(ocr_img_path)
                im = cv2.imread(ocr_img_path)
                self.height, self.width, c = im.shape
                self.region = (0, 0, self.width, self.height)
                self.ocr_img_path = ocr_img_path

        if crop_region:
            im = cv2.imread(self.ocr_img_path)
            region = im[int(self.region[1]):int(self.region[3]), int(self.region[0]):int(self.region[2])]
            crop_image = os.path.abspath(f'{self.temp_folder}\\target_region.png')
            print(crop_image)
            cv2.imwrite(crop_image, region)

        if engine == 'PaddleOCR' and apply_ocr:
            self.PaddleOCR = PaddleOCR(use_angle_cls=True, lang=self.lang)
            try:
                if crop_region:
                    result = self.PaddleOCR.ocr(crop_image)
                else:
                    result = self.PaddleOCR.ocr(self.ocr_img_path)

                df = pd.DataFrame(result[0], columns=['bboxes', 'words'])
                df['top_left_x'] = df['bboxes'].apply(lambda x: x[0][0])
                df['top_left_y'] = df['bboxes'].apply(lambda x: x[0][1])
                df['bottom_right_x'] = df['bboxes'].apply(lambda x: x[2][0])
                df['bottom_right_y'] = df['bboxes'].apply(lambda x: x[2][1])
                df['confidence'] = df['words'].apply(lambda x: x[1])
                df['words'] = df['words'].apply(lambda x: x[0])
                df = df.drop('bboxes', axis=1)

            except Exception as e:
                raise Exception('Failed to run OCR: ', str(e))

        if engine == 'EasyOCR' and apply_ocr:
            if self.lang == 'en':
                reader = easyocr.Reader([self.lang])
            elif self.lang == 'ch':
                reader = easyocr.Reader(['ch_sim', 'en'])
            elif self.lang == 'cht':
                reader = easyocr.Reader(['ch_tra', 'en'])
            else:
                reader = easyocr.Reader([self.lang, 'en'])
            try:
                if crop_region:
                    result = reader.readtext(crop_image)
                else:
                    result = reader.readtext(self.ocr_img_path)
                df = pd.DataFrame(result, columns=['bboxes', 'words', 'confidence'])
                df['top_left_x'] = df['bboxes'].apply(lambda x: x[0][0])
                df['top_left_y'] = df['bboxes'].apply(lambda x: x[0][1])
                df['bottom_right_x'] = df['bboxes'].apply(lambda x: x[2][0])
                df['bottom_right_y'] = df['bboxes'].apply(lambda x: x[2][1])
                df = df.drop('bboxes', axis=1)

            except Exception as e:
                raise Exception('Failed to run OCR: ', str(e))

        if engine == 'TrOCR-handwritten' and apply_ocr:
            if not crop_region:
                raise Exception(
                    'TrOCR-handwritten can only be used when handwriting is extracted with region related methods, and setting crop_region=True.')
            else:
                try:
                    image = Image.open(crop_image).convert("RGB")
                    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
                    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
                    pixel_values = processor(images=image, return_tensors="pt").pixel_values
                    generated_ids = model.generate(pixel_values)
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    df = pd.DataFrame({'words': generated_text, 'confidence': None}, index=[0])

                except Exception as e:
                    raise Exception('Failed to run OCR: ', str(e))

        if not apply_ocr:

            if Path(self.file_path).suffix.lower() in ['.jpg', '.png', '.jpeg', '.gif', '.tiff']:
                image = Image.open(self.file_path)
                im = image.convert('RGB')
                pdf_path = f'{self.temp_folder}\\{self.tmp_prefix}_{self.page}.pdf'
                im.save(pdf_path)
                self.ocr_img_path = self.file_path
            else:
                pdf_path = self.file_path
                ocr_img_path = f'{self.temp_folder}/{self.tmp_prefix}_{self.page}.png'
                pdf = pdfium.PdfDocument(self.file_path)
                page = pdf.get_page(self.page - 1)
                pil_image = page.render_topil(
                    scale=1,  # 72dpi resolution
                    rotation=0,  # no additional rotation
                    # ... further rendering options
                )
                pil_image.save(ocr_img_path)
                self.ocr_img_path = ocr_img_path

            with pdfplumber.open(pdf_path) as pdf:
                page = pdf.pages[self.page - 1]

                result = page.extract_words()
                df = pd.DataFrame(result,
                                  columns=['text', 'x0', 'x1', 'top', 'doctop', 'bottom', 'upright', 'direction'])
                df['words'] = df['text']
                df = df.drop('text', axis=1)

                im = cv2.imread(ocr_img_path)
                im_h, im_w, im_c = im.shape
                self.width, self.height = im_w, im_h
                iw, ih = page.width, page.height
                scale_w, scale_y = im_w / iw, im_h / ih

                df['top_left_x'] = df['x0'] * scale_w
                df['top_left_y'] = df['top'] * scale_y
                df['bottom_right_x'] = df['x1'] * scale_w
                df['bottom_right_y'] = df['bottom'] * scale_y
                df['confidence'] = 1.0
                df = df.drop(['x0', 'x1', 'top', 'doctop', 'bottom', 'upright', 'direction'], axis=1)
                if crop_region:
                    df = df[df.apply(
                        lambda x: (x['top_left_x'] >= self.region[0]) & (x['top_left_y'] >= self.region[1]) & (
                                x['bottom_right_x'] <= self.region[2]) & (x['bottom_right_y'] <= self.region[3]),
                        axis=1)]

        self.ocr_result = df

        if embedding:
            df = self.calculate_embeddings()

        if crop_region:
            self.df_crop = df
            print(df)
        else:
            # df[['top_left_x', 'top_left_y', 'bottom_right_x', 'bottom_right_y']] = df[['top_left_x', 'top_left_y', 'bottom_right_x', 'bottom_right_y']].astype(int)

            df['w'] = (df['bottom_right_x'] - df['top_left_x'])
            df['w'] = df.apply(lambda x: x['w']/len(x['words']), axis=1)
            w = df['w'].mean()
            df['h'] = df['bottom_right_y'] - df['top_left_y']
            df['h'] = df['h'].astype(int)
            h = df['h'].mean()
            df['lines'] = 1
            df['space'] = 0

            self.ocr_result = df
            self.average_line_space = self.get_average_line_space()
            self.w, self.h = w, h
            self.reset_region()
            self.table_h, self.table_v = self.detect_lines()

        if engine == 'TrOCR-handwritten':
            df.to_csv(f'{self.temp_folder}\\{self.tmp_prefix}_{self.page}.csv',
                      columns=['words', 'confidence'])
        else:
            df.to_csv(f'{self.temp_folder}\\{self.tmp_prefix}_{self.page}.csv',
                      columns=['words', 'top_left_x', 'top_left_y', 'bottom_right_x', 'bottom_right_y', 'confidence','w','h','lines','space'])
            df.to_csv(f'{self.temp_folder}\\{self.tmp_prefix}_{self.page}_full.csv')

        return df

    def calculate_embeddings(self):
        df = self.ocr_result
        sentences = list(df.iloc[:]['words'])
        sentences_lower_trim = list(df['words'].str.replace(' ', '').str.lower())
        embeddings = self.model.encode(sentences)
        embeddings_lower_trim = self.model.encode(sentences_lower_trim)
        df['embedding'] = [*embeddings]
        df['embeddings_lower_trim'] = [*embeddings_lower_trim]
        self.ocr_result = df
        return df

    def find_text(self, keyword, fuzzy=0.65, position=None, nth=1, sort_by='fuzzy_matching_lower_trim'):
        df = self.ocr_result
        df['text_contains'] = df.apply(lambda x: keyword.lower() in x['words'].lower(), axis=1)
        keyword_embedding = self.model.encode(keyword)
        keyword_embedding_lower_trim = self.model.encode(keyword.replace(' ', '').lower())
        df['fuzzy_matching'] = df.apply(
            lambda x: util.pytorch_cos_sim(x['embedding'], keyword_embedding).item(), axis=1)
        df['fuzzy_matching_lower_trim'] = df.apply(
            lambda x: util.pytorch_cos_sim(x['embeddings_lower_trim'], keyword_embedding_lower_trim).item(), axis=1)

        df_fuzzy = df[
            df['text_contains'] | df['fuzzy_matching'].ge(float(fuzzy)) | df['fuzzy_matching_lower_trim'].ge(
                float(fuzzy))]
        if position is None:
            df_fuzzy.sort_values(by=[sort_by], inplace=True, ascending=[False])
        if position == 'top':
            df_fuzzy.sort_values(by=['top_left_y', sort_by], inplace=True, ascending=[True, False])
        if position == 'bottom':
            df_fuzzy.sort_values(by=['top_left_y', sort_by], inplace=True, ascending=[False, False])
        if position == 'left':
            df_fuzzy.sort_values(by=['top_left_x', sort_by], inplace=True, ascending=[True, False])
        if position == 'right':
            df_fuzzy.sort_values(by=['top_left_x', sort_by], inplace=True, ascending=[False, False])

        print(df_fuzzy)
        if nth >= 1:
            return df_fuzzy.iloc[nth - 1][:]
        else:
            return df_fuzzy

    def set_region(self, element=None, text=None, relation='above', offset=0.0, position='whole'):
        if type(element) == str:
            text = element
            element = None
        if text and element is None:
            try:
                element = self.find_text(keyword=text)
            except:
                print(self.region)
                raise Exception('Failed to find text: ' + text)
        top_left_x = element.top_left_x
        top_left_y = element.top_left_y
        bottom_right_x = element.bottom_right_x
        bottom_right_y = element.bottom_right_y
        a, b, c, d = 0, 0, self.width, self.height
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

    def on_the_same_row(self, element=None, text=None, offset=(0, 0), relation=None, relation_offset=0.0):
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
            c = top_left_x - relation_offset
        if relation == 'right':
            a = bottom_right_x + relation_offset

        self.region = (float(max(self.region[0], a)), float(max(self.region[1], b)),
                       float(min(self.region[2], c)), float(min(self.region[3], d)))

    def on_the_same_column(self, element=None, text=None, offset=(0, 0), relation=None, relation_offset=0):
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
            d = top_left_y - relation_offset
        if relation == 'below':
            b = bottom_right_y + relation_offset

        self.region = (float(max(self.region[0], a)), float(max(self.region[1], b)),
                       float(min(self.region[2], c)), float(min(self.region[3], d)))

    def get_df_from_region(self):
        df = self.ocr_result
        df_region = df[df.apply(
            lambda x: (int(x['top_left_x']) >= int(self.region[0])) & (int(x['top_left_y']) >= int(self.region[1])) & (
                    int(x['bottom_right_x']) <= int(self.region[2])) & (
                              int(x['bottom_right_y']) <= int(self.region[3])), axis=1)]
        if len(df) > 0:
            return df_region
        else:
            raise ('Cannot get OCR text within region: ', self.region)

    def extract_text(self, apply_ocr=True, engine='PaddleOCR', separator=' ', offset=5):

        if self.region == (0, 0, self.width, self.height):
            crop_region = False
        else:
            crop_region = True

        if engine == self.engine:
            if not crop_region:
                df_region = self.ocr_result
            else:
                df_region = self.get_df_from_region()
        else:
            df_region = self.extract_words(apply_ocr=apply_ocr, engine=engine, crop_region=crop_region, embedding=False)
        if len(df_region) > 0:
            return separator.join(df_region.words)
        else:
            region = self.region
            self.region = (region[0] - offset, region[1] - offset, region[2] + offset, region[3] + offset)
            df_region = self.extract_words(apply_ocr=apply_ocr, engine=engine, crop_region=crop_region,
                                           embedding=False)
            if len(df_region) > 0:
                return separator.join(df_region.words)
            else:
                print(self.region)
                raise Exception('Cannot get OCR text within region')

    def NER(self, text=None):
        nlp = None
        if self.lang == 'en':
            nlp = spacy.load('en_core_web_trf')
        if self.lang == 'ch':
            nlp = spacy.load('zh_core_web_trf')
        doc = nlp(text)
        nlp_analysis = pd.DataFrame({}, columns=['Text', 'Label', 'Start', 'End'])
        for i in doc.ents:
            nlp_analysis = nlp_analysis.append(
                {'Text': i.text,
                 'Start': i.start_char,
                 'End': i.end_char,
                 'Label': i.label_}, ignore_index=True)
        return nlp_analysis

    def get_entity_by_label(self, text=None, labels=['DATE']):
        NER_analysis = self.NER(text=text)
        df_entity = NER_analysis[NER_analysis.apply(lambda x: x['Label'] in labels, axis=1)]
        return df_entity

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

    def extract_tables(self):
        if Path(self.file_path).suffix.lower() in ['.jpg', '.png', '.jpeg', '.gif', '.tiff']:
            image = Image.open(self.file_path)
            im = image.convert('RGB')
            pdf_path = f'{self.temp_folder}\\{self.tmp_prefix}_{self.page}.pdf'
            im.save(pdf_path)
        else:
            pdf_path = self.file_path

        pdf = pdfplumber.open(pdf_path)
        page = pdf.pages[self.page - 1]
        tables = page.extract_tables()

        return None

    def align_horizontal_line(self, to_int=False):
        df = self.ocr_result.sort_values(by=['top_left_y'], ascending=[True])
        updated = []

        for idx, element in df.iterrows():
            if idx not in updated:
                top_left_y = element.top_left_y
                bottom_right_y = element.bottom_right_y
                self.reset_region()

                self.on_the_same_row(element=element, relation='right')
                df_row = self.get_df_from_region()

                if len(df_row) > 0:
                    for idx_left, element_right in df_row.iterrows():
                        if element.name != element_right.name:
                            top_left_y = min(top_left_y, element_right.top_left_y)
                            bottom_right_y = max(bottom_right_y, element_right.bottom_right_y)

                    for idx_left, element_right in df_row.iterrows():
                        self.ocr_result.at[element_right.name, 'top_left_y'] = top_left_y
                        self.ocr_result.at[element_right.name, 'bottom_right_y'] = bottom_right_y
                        updated.append(element_right.name)

                    self.ocr_result.at[element.name, 'top_left_y'] = top_left_y
                    self.ocr_result.at[element.name, 'bottom_right_y'] = bottom_right_y

        if to_int:
            df[['top_left_x', 'top_left_y', 'bottom_right_x', 'bottom_right_y']] = df[
                ['top_left_x', 'top_left_y', 'bottom_right_x', 'bottom_right_y']].astype(int).astype(float)

        self.ocr_result.to_csv(f'{self.temp_folder}\\{self.tmp_prefix}_{self.page}_predict.csv',
                               columns=['words', 'top_left_x', 'top_left_y', 'bottom_right_x', 'bottom_right_y',
                                        'confidence'])

    def update_df_after_merge(self,action,separator,element,element_df,direction):

        bottom_right_x = self.ocr_result.loc[element.name][:].bottom_right_x
        bottom_right_y = self.ocr_result.loc[element.name][:].bottom_right_y
        bottom_right_x = max(bottom_right_x, element_df.bottom_right_x)
        bottom_right_y = max(bottom_right_y, element_df.bottom_right_y)
        top_left_x = self.ocr_result.loc[element.name][:].top_left_x
        top_left_y = self.ocr_result.loc[element.name][:].top_left_y
        top_left_x = min(top_left_x, element_df.top_left_x)
        top_left_y = min(top_left_y, element_df.top_left_y)
        old_words = self.ocr_result.loc[element.name]['words']
        words = old_words + separator + element_df.words
        lines = self.ocr_result.at[element.name, 'lines']
        if self.ocr_result.at[element.name, 'space']:
            space = self.ocr_result.at[element.name, 'space']
        else:
            space = 0
        print(direction + ' ' + action + ":" + str(element.name) + ' with '+ str(element_df.name))

        self.ocr_result.drop(element_df.name, inplace=True)
        self.ocr_result.at[element.name, 'bottom_right_x'] = bottom_right_x
        self.ocr_result.at[element.name, 'bottom_right_y'] = bottom_right_y
        self.ocr_result.at[element.name, 'top_left_x'] = top_left_x
        self.ocr_result.at[element.name, 'top_left_y'] = top_left_y
        if direction == 'vertical':
            self.ocr_result.at[element.name, 'lines'] = lines + 1
            if space == 0:
                self.ocr_result.at[element.name, 'space'] = element_df.top_left_y - element.bottom_right_y
            else:
                self.ocr_result.at[element.name, 'space'] = (space + element_df.top_left_y - element.bottom_right_y)/2
        self.ocr_result.at[element.name, 'words'] = words
        print(self.ocr_result.loc[element.name].words)
        self.ocr_result.to_csv(f'{self.temp_folder}\\{self.tmp_prefix}_{self.page}_{direction}_{action}.csv',
                               columns=['words', 'top_left_x', 'top_left_y', 'bottom_right_x',
                                        'bottom_right_y',
                                        'confidence'])
        return bottom_right_x, bottom_right_y

    def get_average_line_space(self):
        df = self.ocr_result.sort_values(by=['top_left_y'], ascending=[True])
        first_df = df.iloc[0][:]
        start_y = first_df.bottom_right_y
        space = []
        while start_y + 1 < self.height:
            second_df = df[df.apply(lambda x:x['top_left_y'] > start_y, axis=1)].sort_values(by=['top_left_y'], ascending=[True])
            if len(second_df) > 0:
                second_y = second_df.iloc[0][:].top_left_y
                space.append(int(second_y - start_y))
                self.ocr_result.at[second_df.iloc[0][:].name, 'space'] = int(second_y - start_y)
                start_y = second_df.iloc[0][:].bottom_right_y
            else:
                break

        counts = np.bincount(space)
        return np.argmax(counts)

        #df_region  = df[df.apply(lambda x: x['bottom_right_y']>=y, axis=1)]
    def merge_texts(self, w=1.0, h=1.0, separator=' ', action='merge_overlap', p=0.7, direction=''):
        df = self.ocr_result
        #w = self.w * w
        old_w = w

        page_width = df['bottom_right_x'].max() - df['top_left_x'].min()

        i = 0

        while i < max(df.index):
            df = self.ocr_result
            if i in df.index:
                element = df.loc[i][:]
            else:
                i = i + 1
                continue
            print(i)
            top_left_x = element.top_left_x
            top_left_y = element.top_left_y
            bottom_right_x = element.bottom_right_x
            bottom_right_y = element.bottom_right_y
            lines = element.lines
            width = bottom_right_x - top_left_x
            height = (bottom_right_y - top_left_y) / lines
            width_p = width / page_width
            old_space = element.space
            w = (element.bottom_right_x-element.top_left_x)/len(element.words) * old_w


            if width_p < p and action == 'merge_paragraph' and direction == 'vertical':
                i = i + 1
                continue

            df_action = None

            if action == 'merge_overlap' and direction == 'vertical':
                df_action = df[df.apply(lambda x: bottom_right_y >= x.top_left_y >= top_left_y and abs(x.top_left_x - top_left_x) <= old_w*self.w, axis=1)].sort_values(by=['top_left_x','top_left_y'], ascending=[True,True])

            if action == 'merge_nearby' and direction == 'vertical':
                df_action = df[df.apply(lambda x: bottom_right_y < x.top_left_y and abs(bottom_right_y-x.top_left_y) <= 0.25 * height and abs(x.top_left_x - top_left_x) <= old_w*self.w, axis=1)].sort_values(by=['top_left_x','top_left_y'], ascending=[True,True])

            if action == 'merge_nearby' and direction == 'horizontal':
                self.reset_region()
                self.on_the_same_row(element=element, offset=(0.5*self.h, 0.5*self.h), relation='right', relation_offset=-w)
                #self.set_region(element=element,relation='above',offset=average_line_space)
                df_action = self.get_df_from_region().sort_values(by=['top_left_x'], ascending=[True])

            if len(df_action) > 0:
                match = False
                print(element.name)
                for idx, element_df in df_action.iterrows():
                    new_height = (element_df.bottom_right_y - element_df.top_left_y) / element_df.lines
                    new_space = element_df.top_left_y - bottom_right_y
                    if element.name == element_df.name:
                        continue
                    if direction == 'horizontal':
                        gap = (element_df.top_left_x - self.ocr_result.loc[element.name].bottom_right_x)
                        if action == 'merge_nearby' and gap <= w:
                            a = self.table_v[int(element_df.top_left_y):int(self.ocr_result.loc[i].bottom_right_y), int(self.ocr_result.loc[i].bottom_right_x -w):int(element_df.top_left_x+w)].nonzero()

                            if a[0].sum() == 0 and a[1].sum() == 0:

                                j = 3
                                a = self.ocr_result.loc[i].bottom_right_x
                                b = self.ocr_result.loc[i].bottom_right_y
                                c = element_df.top_left_x
                                d = b + j * self.h

                                df = self.ocr_result
                                df_filter = df[df.apply(lambda x: x['top_left_y'] > b and x['bottom_right_x'] >= c and x['bottom_right_y'] <= d, axis=1)]

                                proceed = False
                                print(gap, self.w)
                                if gap < self.w:
                                    proceed = True
                                else:
                                    print(df_filter)
                                    if len(df_filter) == 0:
                                        proceed = True

                                if proceed:
                                    self.update_df_after_merge(action=action, separator=separator, element=element, element_df=element_df, direction='horizontal')
                                    match = True
                                    break
                                else:
                                    continue
                            else:
                                continue
                        else:
                            continue
                    if direction == 'vertical':
                        proceed = True
                        if new_height / height >= h or height / new_height >= h:
                            print(f"new_height {new_height}, height {height}, {new_height / height}, {height / new_height}")
                            proceed = False
                        if action == 'merge_nearby' and old_space > 0:
                            if new_space / old_space > 1.1 and abs(new_space-old_space) > 1:
                                proceed = False
                        if action == 'merge_nearby':
                            df = self.ocr_result
                            a = self.ocr_result.loc[i].top_left_x
                            b = self.ocr_result.loc[i].top_left_y
                            c = self.ocr_result.loc[i].bottom_right_x
                            d = self.ocr_result.loc[i].bottom_right_y
                            df_filter = df[df.apply(lambda x: c <= x['top_left_x'] <= c+10*self.w and b-2 <= x['top_left_y'] <= b+2 and d-2 <= x['bottom_right_y'] <= d+2, axis=1)]
                            if len(df_filter) == 0:
                                proceed = False
                            else:
                                print(df_filter)
                        if proceed:
                            match = True
                            self.update_df_after_merge(action=action,separator='\n',element=element,element_df=element_df,direction=direction)
                            break
                        else:
                            continue
                if not match:
                    i = i + 1
            else:
                i = i + 1

        self.ocr_result.to_csv(f'{self.temp_folder}\\{self.tmp_prefix}_{self.page}_{direction}_{action}.csv',
                               columns=['words', 'top_left_x', 'top_left_y', 'bottom_right_x', 'bottom_right_y',
                                        'confidence'])

    def merge_paragraph(self, w=1, h=1, p=0.7, separator=' '):
        self.merge_texts(w=w, h=h, p=p, separator=separator, action='merge_paragraph', direction='vertical')

    def draw_bboxes(self, show_image=False):
        df = self.ocr_result
        color = (255, 0, 0)
        thickness = 2
        im = cv2.imread(self.ocr_img_path)
        print((self.width, self.height))
        '''if resize or not self.apply_ocr:
            im = cv2.resize(im, (int(self.width), int(self.height)),None, 0, 0, cv2.INTER_AREA)'''

        for idx, element in df.iterrows():

            start_point = (int(element.top_left_x), int(element.top_left_y))
            end_point = (int(element.bottom_right_x), int(element.bottom_right_y))

            image = cv2.rectangle(im, start_point, end_point, color, thickness)

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

    def analyze_layout(self, w=4, h=1.4):
        self.ocr_result = pd.read_csv(f'{self.temp_folder}\\{self.tmp_prefix}_{self.page}_full.csv')
        #self.align_horizontal_line(to_int=True)
        self.merge_texts(w=w, h=h, action='merge_nearby', direction='horizontal')

        self.calculate_embeddings()

        self.merge_texts(w=w, h=h, action='merge_overlap', direction='vertical')
        self.merge_texts(w=w, h=h, action='merge_nearby', direction='vertical')

    # %% 最小包裹正矩形 ================================================================
    def boundingRect(self, image_path):
        # 读入图片3通道 [[[255,255,255],[255,255,255]],[[255,255,255],[255,255,255]]]
        image = cv2.imread(image_path)
        # 转为灰度单通道 [[255 255],[255 255]]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 黑白颠倒
        gray = cv2.bitwise_not(gray)
        # 二值化
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # 获取最小包裹正矩形 x-x轴位置, y-y轴位置, w-宽度, h-高度
        x, y, w, h = cv2.boundingRect(thresh)
        left, top, right, bottom = x, y, x + w, y + h

        # 把框画在图上
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

        # 写入文件
        # cv2.imwrite('img2_1_rotate.jpg', image)
        # 弹出展示图片
        # cv2.imshow("output", image)
        # cv2.waitKey(0)

    # %% 最小面积矩形 ========================================================================
    def minAreaRect(self, image_path):
        # 读入图片3通道 [[[255,255,255],[255,255,255]],[[255,255,255],[255,255,255]]]
        image = cv2.imread(image_path)
        # 转为灰度单通道 [[255 255],[255 255]]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 黑白颠倒
        gray = cv2.bitwise_not(gray)
        # 二值化
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # %% 把大于0的点的行列找出来
        ys, xs = np.where(thresh > 0)
        # 组成坐标[[306  37][306  38][307  38]],里面都是非零的像素
        coords = np.column_stack([xs, ys])
        # 获取最小矩形的信息 返回值(中心点，长宽，角度)
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]  # 最后一个参数是角度
        # print(rect, angle)  # ((26.8, 23.0), (320.2, 393.9), 63.4)

        # %%  通过换算，获取四个顶点的坐标
        box = cv2.boxPoints(rect)
        box = np.int0(cv2.boxPoints(rect))
        # print(box)  # [[15 181][367  5][510 292][158 468]]

        cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
        cv2.imshow("output", image)
        cv2.waitKey(0)

        return angle

    # %% 霍夫线 ========================================================================

    # 计算一条直线的角度
    def calculateAngle(self, x1, y1, x2, y2):
        x1 = float(x1)
        x2 = float(x2)
        y1 = float(y1)
        y2 = float(y2)

        if x2 - x1 == 0:
            result = 90  # 直线是竖直的
        elif y2 - y1 == 0:
            result = 0  # 直线是水平的
        else:
            # 计算斜率
            k = -(y2 - y1) / (x2 - x1)
            # 求反正切，再将得到的弧度转换为度
            result = np.arctan(k) * 57.29577
        return result

    # 霍夫线获得角度
    def houghImg(self, image_path):
        # 读入图片3通道 [[[255,255,255],[255,255,255]],[[255,255,255],[255,255,255]]]
        image = cv2.imread(image_path)
        # 转为灰度单通道 [[255 255],[255 255]]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 处理边缘
        edges = cv2.Canny(gray, 500, 200, 3)
        # 求得
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, maxLineGap=200)
        print(lines)

        # 得到所有线段的端点
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255))
            angle = self.calculateAngle(x1, y1, x2, y2)
            angles.append(round(angle))

        mostAngle = Counter(angles).most_common(1)[0][0]
        # print("mostAngle:", mostAngle)

        # cv2.imshow("output", image)
        # cv2.waitKey(0)

        return mostAngle

    # %% 图片旋转 ============================================================================

    def rotate_bound(self, image, angle):
        # 获取宽高
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        # 提取旋转矩阵 sin cos
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # 计算图像的新边界尺寸
        nW = int((h * sin) + (w * cos))
        nH = h
        # 调整旋转矩阵
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def doc_rotate(self):
        self.boundingRect(self.ocr_img_path)
        # 【C】霍夫线，获得角度
        mostAngle = self.houghImg(self.ocr_img_path)
        # 旋转图片,查看效果
        image = self.rotate_bound(cv2.imread(self.ocr_img_path), mostAngle)
        # cv2.imshow("output", image)
        # cv2.waitKey(0)
        cv2.imwrite(self.ocr_img_path, image)

    def detect_lines(self):
        image = cv2.imread(self.ocr_img_path, 1)
        # 灰度图片
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 二值化
        binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -5)
        # ret,binary = cv2.threshold(~gray, 127, 255, cv2.THRESH_BINARY)
        #cv2.imshow("二值化图片：", binary)  # 展示图片
        #cv2.waitKey(0)

        rows, cols = binary.shape
        scale = 40
        # 识别横线
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
        eroded = cv2.erode(binary, kernel, iterations=1)
        # cv2.imshow("Eroded Image",eroded)
        dilatedcol = cv2.dilate(eroded, kernel, iterations=1)
        # cv2.imshow("表格横线展示：", dilatedcol)
        # cv2.waitKey(0)
        cv2.imwrite('tmp/h.png', dilatedcol)

        # 识别竖线
        scale = 20
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
        eroded = cv2.erode(binary, kernel, iterations=1)
        dilatedrow = cv2.dilate(eroded, kernel, iterations=1)
        # cv2.imshow("表格竖线展示：", dilatedrow)
        # cv2.waitKey(0)
        cv2.imwrite('tmp/v.png', dilatedrow)

        return dilatedcol, dilatedrow
