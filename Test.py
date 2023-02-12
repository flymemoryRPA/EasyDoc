from EasyDoc import EasyDoc

doc = EasyDoc("doc/Test.pdf", page=1)

ocr_result = doc.get_ocr_result()
doc.on_the_same_column(text='Name (as shown', relation='below')
doc.set_region(text='Business name', relation='above')
doc.extract_ocr(engine='TrOCR-handwritten')
doc.draw_region('Name', show_image=True)
#print(doc.region)

doc = EasyDoc("doc/Test.pdf", page=6)

ocr_result = doc.get_ocr_result()
doc.reset_region()
doc.set_region(text='Bill To', relation='below')
address = doc.get_paragraph(separator='\n', h=50, w=500)
print(address)

doc.reset_region()
doc.on_the_same_column(text='FEE', offset=(300, 10))
doc.on_the_same_row(text='Total Fee including Tax', offset=(10, 10))
print(doc.get_text_from_region())
#print(doc.region)