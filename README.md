# text_deskewer
Text alignment for OCR using a FFT

## Usage
```python
from deskewer import deskew_view, deskew
import cv2 as cv


file_name = '1.jpg'
img = cv.imread(file_name)
rotated_img, rotated_angle = deskew_view(img)
```

![Alt text](/deskewer.png "Optional title")
![Alt text](/deskewer3.png "Optional title")
