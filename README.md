# automated_numberPlate_annotator
number_plate_annotation created with https://huggingface.co/keremberke/yolov5m-license-plate



- Installation
```
pip install -U yolov5
```


- Usage 
    - Change the ```img_sources``` in ```annotator.py```
        ```
        python3 annotator.py
        ```

    - Cropped ROI Images will be generated in ```./cropped``` folder
    - ```./annotations.csv``` file will contain the annotations
        - each row is:
            - {source_file},{x1},{x2},{y1},{y2},{confidence of annotation}, {cropped number plate}