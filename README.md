# Airbus Ship Detection Challenge
This repository aimes to create a neural network, that is capable of detecting ships on satellite images.
This repository is a homework for this course: [http://smartlab.tmit.bme.hu/oktatas-deep-learning](http://smartlab.tmit.bme.hu/oktatas-deep-learning).
For detailed description about the project and the team members, please check our [wiki](https://github.com/Deep-Learning-GAZ/Airbus-Ship-Detection-Challenge/wiki).

If you want to learn more about the data, check out these:
* [Vizualization.ipynb](https://github.com/Deep-Learning-GAZ/Airbus-Ship-Detection-Challenge/blob/master/Vizualization.ipynb) show a sigle image and it's annotation from the dataset.
* [VisualizeShipOccurances.ipynb](https://github.com/Deep-Learning-GAZ/Airbus-Ship-Detection-Challenge/blob/master/VisualizeShipOccurances.ipynb) show heatmaps about where the ships are located.
* [DataExploration.ipynb](https://github.com/Deep-Learning-GAZ/Airbus-Ship-Detection-Challenge/blob/master/DataExploration.ipynb) calculates various statistiscs about the dataset, like ship counts area etc.

To solve the proble we try various methods:
* Our first solution is a segmentation using an object classifier changing its last layers, implemented in [train_increasing_rcm.ipynb](https://github.com/Deep-Learning-GAZ/Airbus-Ship-Detection-Challenge/blob/master/train_increasing_rcm.ipynb).

* For image segmentation SegNet is a popular choiche. [SegNetModel.py](https://github.com/Deep-Learning-GAZ/Airbus-Ship-Detection-Challenge/blob/master/Model/SegNetModel.py) implements this solution with the help of [model.py](https://github.com/Deep-Learning-GAZ/Airbus-Ship-Detection-Challenge/blob/master/model.py) and [layers.py](https://github.com/Deep-Learning-GAZ/Airbus-Ship-Detection-Challenge/blob/master/layers.py). The solution uses argmax for MaxPooling and has residual connections as well, we modified [the implementation of ykamikawa](https://github.com/ykamikawa/SegNet) to help us create our SegNet. The result of the training and evaluation can be found in [train_segnet.ipynb](https://github.com/Deep-Learning-GAZ/Airbus-Ship-Detection-Challenge/blob/master/train_segnet.ipynb). Some predictions are visible at [show_prediction_segnet.ipynb](https://github.com/Deep-Learning-GAZ/Airbus-Ship-Detection-Challenge/blob/master/show_prediction_segnet.ipynb).


