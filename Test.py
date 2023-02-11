from EasyDoc import EasyDoc

doc = EasyDoc(r"Test.pdf")

ocr_result = doc.get_ocr_result()
doc.on_the_same_column(text='Name (as shown', relation='below')
doc.set_region(text='Business name', relation='above')
doc.extract_ocr(engine='TrOCR-handwritten')
doc.draw_region('Name', show_image=True)
#print(doc.region)

doc.reset_region()
doc.on_the_same_column(text='Social security number', relation='below', offset=(0, 0, 420, 0))
doc.set_region(text='Employer Identification number', relation='above', offset=-30)
doc.extract_ocr(engine='TrOCR-handwritten')
doc.draw_region('SSN', show_image=True)
#print(doc.region)