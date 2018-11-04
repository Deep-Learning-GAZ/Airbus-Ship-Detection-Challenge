# Airbus Ship Detection Challenge
This repositorz aimes to create a neural network, that is capable, to detect ships on satelite images.
This reposetory is a homework for this course: [http://smartlab.tmit.bme.hu/oktatas-deep-learning](http://smartlab.tmit.bme.hu/oktatas-deep-learning).
For detaild description about the project and the team members, pleas check our [wiki](https://github.com/Deep-Learning-GAZ/Airbus-Ship-Detection-Challenge/wiki).

If zou want to learn more about the data, check out these:
* [Vizualization.ipynb](https://github.com/Deep-Learning-GAZ/Airbus-Ship-Detection-Challenge/blob/master/Vizualization.ipynb) show a sigle image and it's annotation from the dataset.
* [VisualizeShipOccurances.ipynb](https://github.com/Deep-Learning-GAZ/Airbus-Ship-Detection-Challenge/blob/master/VisualizeShipOccurances.ipynb) show heatmaps about where the ships are located.
* [DataExploration.ipynb](https://github.com/Deep-Learning-GAZ/Airbus-Ship-Detection-Challenge/blob/master/DataExploration.ipynb) calculates various statistiscs about the dataset, like ship counts area etc.

To solve the proble we try various methods:
* Our first solution is a segmentation using an object classifier changing its last layers, implemented in [segmentation.ipynb](https://github.com/Deep-Learning-GAZ/Airbus-Ship-Detection-Challenge/blob/master/segmentation.ipynb).

* For image segmentation SegNet is a popular choiche. [Train.ipynb](https://github.com/Deep-Learning-GAZ/Airbus-Ship-Detection-Challenge/blob/master/Train.ipynb) implements this solution leightweightly with the help of model.py and layers.py. We executed the stored specific run on batches sized 1, the losses are 0 if the image doesn't contain any ships, otherwise not zero(it doesn't make sense in sight of reaching any result, but that is going to be the task of milestone 3). Our plans later are expanding it to be bigger and do MaxPooling with argmax.

* RetinaNet object detector: The annotations are defenetly rectangles, so it makes sense to try to predict the parameters of that rectangle. There are many frameworks to do object detection. After looking at the literature, it seems, that RetinaNet outperformes every other attempts in terms of accuracy. However all these methos are assuming, that the bounding boxes are paralell with the x and y axis. In this case our rectangles have an orientation too. I have searched an existing implementation of RetinaNet in keras and found the [keras-retinanet](https://github.com/fizyr/keras-retinanet). I have copied the code to our repository and modified it the predict orientations for the anchor boxes as well. To run this training fors run [create keras-retinanet csv.ipynb](https://github.com/Deep-Learning-GAZ/Airbus-Ship-Detection-Challenge/blob/Research-object-detection-solutions-%2311/create%20keras-retinanet%20csv.ipynb) notebook to generate the csv files containing the annotations. After that call `python keras_retinanet\bin\train.py csv train.csv ids.csv`.

