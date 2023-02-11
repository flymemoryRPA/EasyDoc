# EasyDoc


```Python
from EasyDoc import EasyDoc
doc = EasyDoc(r"C:\Users\ccigc\Downloads\00612270.pdf")
ocr_result = doc.get_ocr_result()
account_number = doc.find_text('Account Number').iloc[0][:]
doc.set_region(account_number, 'above', 10)
doc.set_region(account_number, 'below', -50, 'top')
doc.set_region(account_number, 'right', -10, 'left')
doc.set_region(account_number, 'left', -500, 'right')
print(doc.extraction)
doc.draw_region('Account Number')

```