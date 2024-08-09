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

