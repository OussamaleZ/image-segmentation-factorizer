# image-segmentation-factorizer
Paper's presentation for the course: Medical Image Analysis based on generative, geometric and biophysical models

Group members: Hamza Azzouzi, Oussama Zouhry

Paper: "Factorizer: A scalable interpretable approach to context modeling for medical image segmentation" (Ashtari et al.) 


Steps to follow:

1. **Install requirements**

2. **Import the datasets into a folder named `data`.** It should contain two subfolders:

   2.1 **isles:** [ISLES 2022: A multi-center magnetic resonance imaging stroke lesion segmentation dataset](https://zenodo.org/records/7153326)

   2.2 **brats:** [The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2) (Download data of task1)

In short, your `data` folder should be structured like this:

    data/
        isles/
            derivatives/
                sub-strokecase0001/
                sub-strokecase0002/
                .
                .
                sub-strokecase0250/
            sub-strokecase0001/
            sub-strokecase0002/
            .
            .
            sub-strokecase0250/
        brats/
            imagesTr/
            imagesTs/
            labelsTr/

