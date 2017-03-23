# Image-Retrieval
Image retrieval program made in Tensorflow supporting VGG16, VGG19, InceptionV3 and InceptionV4 feature vectors.


### Requirements
* Python 3.*
* Tensorflow
* Pillow
* tqdm
* Pretrained VGG16, VGG19, InceptionV3 or InceptionV4 network.


### Usage
Firstly put your images in the **images** folder.

**Embdedding images and saving them**
Just do this command.
```
python3 vectorize_pretrained.py --model_path=<model_path> --model_type=<model_type> --layer_to_extract=<layer_to_extract>
```
What does these arguments mean?

**model_path**: Path to pretrained model. e.g ./inception_v4.ckpt

**model_type**: Type of model, either VGG16, VGG19, InceptionV3 or InceptionV4. e.g InceptionV4

**layer_to_extract**: Which layer to take vector from. e.g Mixed_7a

This command will save the vectors in a file in the vectors folder and will print out the path to the vectors for later
use or evaluation at the end of the program.

**Evaluating**
To evaluate your vectors you can do this command.
```
python3 evaluation.py --vectors_path=<vectors_path> --image_path=<image_path>
```
What does these arguments mean?

**vectors_path**: Where vectors are saved. e.g vectors/vectors_1

**image_path**: Image to evaluate on, i.e the image to check nearest neighbour on. e.g img.jpg

### Todos
* Add bottleneck layer from trained Convolutional Autoencoder as feature vector.
* More ways of doing NN search.

### Other
Made by Oliver Edholm, 14 years old.
