# random forest
An image segmentation algorithm for the MLExcahgne platform adapted from [Dash Plotly example ](https://github.com/plotly/dash-sample-apps/blob/d96997bd269deb4ff98b810d32694cc48a9cb93e/apps/dash-image-segmentation/trainable_segmentation.py#L130).

## Getting started
To get started, you will need:
  - [Docker](https://docs.docker.com/get-docker/)

## Running
First, build the segMSDnet image in terminal:  
`cd mlex_random_forest`    
`make build_docker`
  
Once built, you can run the following examples:   
`make train_example`  
`make test_example`  

These examples utilize the information stored in the folder /data. The trained model and the segmented images will be stored in /data/model and /data/out, respectively.

Alternatively, you can run the container interactively as follows:
```
make run_docker
```

While running interactively, you can perform training and testin processes using random forest.



## Copyright
MLExchange Copyright (c) 2021, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights.  As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit others to do so.
