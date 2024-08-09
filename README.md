# Developing an ALPR System with Three Different OCR Systems
In this study, license plate detection is performed on images and videos. Additionally, the aim is to compare the performance of three different OCR systems for license plate text recognition.
## Test ##
Install the project dependencies

```shell
pip install -r requirements.txt
```


## License plate detection in video ##
Execute the following command.

```shell
python video_anpr_detector.py "C:\path\to\your\image\video_name.mp4"
```
**Result:**
The results are saved in a file named *detected_video.mp4*

## Detect license plates in an image using EasyOCR ##
Execute the following command.

```shell
python img_anpr_detector.py "C:\path\to\your\image\image_name.png"
```
**Result:**
The results have been saved in the folder named *output_images*

## Detect license plates in an image using Surya OCR ##
**Installation:** 
```shell
pip install surya-ocr
```

Execute the following command.

```shell
python surya_ocr_anpr.py "C:\path\to\your\image\image_name.png"
```
**Result:**
The results have been saved in the folder named *output_images*

## Detect license plates in an image using Tesseract OCR ##
**Installation:** 
Ensure that Tesseract OCR is installed on your machine. You can download it from [here](https://github.com/tesseract-ocr/tesseract). Update the *tesseract_cmd* in the script if necessary:
```shell
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

```
Execute the following command.

```shell
python tesseract_ocr_anpr.py "C:\path\to\your\image\image_name.png"
```
**Result:**
The results have been saved in the folder named *output_images*

