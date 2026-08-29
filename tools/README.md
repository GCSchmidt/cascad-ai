# Tools Guide

## Dataset Generator

1. Clone repo into Collab. In Collab terminal run:

    ```
    git clone https://github.com/GCSchmidt/cascad-ai.git
    ```

2. CD into cloned repos and Install requirements (packages). In Collab terminal run:

    ```bash
    cd cascad-ai
    pip install -e .
    ```

3. Generate Dataset with ():

    ```
    python3 tools/dataset_genetor.py -train <training sample>
    ```  
  
    💡 By default the amount of validations samples are equal to 20% of training sample amount 


## Annotation Inspector

Given a sample from the YOLO dataset an image is saved to visualise its corresponding annotations. A sample of the YOLO dataset consists of 1 image file and a corresponding annotiontionn (.txt) file.
This tool is useful to verify annotations created during dataset generation. 
The output is saved to the `output` dir with the same name as the image file.

1. Run with

    ```
    python3 tools/annotation_inspector.py -i <image file path> -a <label file path>
    ```

