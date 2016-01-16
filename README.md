# wm

The goal of this tool is to provide watermark services to owners of copyrighted artwork to prevent tampering and
edition without detection.
The tool started as a project of the Forensic Cybersecurity course at Instituto Superior Técnico, and now we are
open source it.


Future work
-----------

* Major code refactoring to add a more permanent structure to the code.
* Implement test-first development.
* Write complete documentation.
* Tweak the implemented algorithms to erase some of the restrictions we have currently.
* Develop an user interface.


Project Setup
-------------

* Get Python 2.7 with pip.
* Install virtualenv ```pip install virtualenv```
* Run ```virtualenv venv```
* Run ```source venv/bin/activate``` to activate virtual environment
* Run ```pip install -r requirements.txt``` to install project dependencies

Make sure you have the virtual environment activated before issuing any command, or ImportErrors will occur.

Operations
----------

The template to perform an operation is the following:

```python run.py <operation> -a <algorithm> <image_file> <watermark>```

where the ```<watermark>`` parameter is optional depending on the algorithm and operation.

The operations available are:`

* ``<operation> = embed``: Embed a watermark

    ```python run.py embed -a cox img/lena.png```

    ```python run.py embed -a dwt img/lena.png img/milk.png```

    ```python run.py embed -a recover img/lena.png```

* ``<operation> = extract``: Extract a watermark

    ```python run.py extract -a cox wm-img/lena.png wm-img/lena_wm.json```
    
    where ```wm-img/lena.png``` is the watermarked image and ``wm-img/lena_wm.json`` is the
    watermark file produced by embed operation.

    ```python run.py extract -a dwt wm-img/lena.png```
    
    where ```wm-img/lena.png``` is the watermarked image
    
    ```python run.py extract -a recover wm-img/lena.png wm-img/lena_k```
    
    where ```wm-img/lena.png``` is the watermarked image and ```wm-img/lena_k``` is a file that contains the key to 
    build the sequence of blocks used in the embed phase of the recovery algorithm.


* ``<operation> = benchmarks``: Run the benchmarks
See next section.

Benchmarks
----------

The benchmarks automate the attacks on watermarked images. We currently implement these attacks:

* Add 20% of contrast
* Unsharp mask (a technique to increase sharpness in images).
* Mode Filter (a type of low-pass filter).
* Median Filter (a type of low-pass filter).
* Rotate 3 degrees counterclockwise.
* Add noise.
* Add gausian blur.
* Use JPEG compression with 10% quality.
* Draw white squares at the middle of the image (sizes: 5%, 15% and 30%).
* Double the size of the image, then reduce to half again.
* Reduce the size of the image to half, then double the size.

To run the benchmarks, issue the following command:

```python run.py benchmarks -a <algorithm> <original_image> <watermark>```

The ``<algorithm>`` parameter refers to one of the algorithms that is implemented, the ``<original_image>`` parameter
to the path of the original image. The ``<watermark>`` is only used in DWT algorithm in the current version and is
the path of the watermark image.
We currently do not support automatic benchmarks for the recover algorithm.

Examples:

```python run.py benchmarks -a cox img/lena.png```

```python run.py benchmarks -a dwt img/lena.png img/milk.png```

