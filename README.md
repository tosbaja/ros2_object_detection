## ros2_object_detection

This is a ros2 repo for working with openCV it has two launch files.
```bash
$ ros2 launch opencv_package camera.launch.py
```
will launch a publisher that publishes 'video0' from your computer. If you're using a laptop this will probably be the webcam. You don't have to use this launch if you already have an image topic to work with.

```bash
$ ros2 launch opencv_package opencv_package.launch.py
```

Will launch the detection node which subcribes to an image topic and detects objects. It publishes a new image msg with boundaries around the detected object. 

### Modifications you can make

First one would be chancing the net itself to detect other things. You can do it by chaning the file paths to a path to new net, weights and classes.

Currently the net uses CUDA cores but if your computer does not have a NVDIA grapcis card you can comment out or delete the following lines.

```bash
net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
```
