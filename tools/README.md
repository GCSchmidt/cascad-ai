# Tools Guide

## Dataset Generator

1. Clone repo into Collab. In Collab terminal run:

    ```git clone https://github.com/GCSchmidt/cascad-ai.git```

2. Install requirements (packages). In Collab terminal run:

    ```pip install -e .```

3. Generate Dataset with ():

    ```python3 tools/dataset_genetor.py -train <training sample>```  
  
    💡 By default the amount of validations samples are equal to 20% of training sample amount 

