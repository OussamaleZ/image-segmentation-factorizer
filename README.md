# image-segmentation-factorizer
Paper's presentation for the course: Medical Image Analysis based on generative, geometric and biophysical models

Group members: Hamza Azzouzi, Oussama Zouhry

Paper: "Factorizer: A scalable interpretable approach to context modeling for medical image segmentation" (Ashtari et al.) 


## Steps to follow:

### 1. **Install requirements**

### 2. **Import the datasets into a folder named `data`.** It should contain two subfolders:

   2.1. **isles:** [ISLES 2022: A multi-center magnetic resonance imaging stroke lesion segmentation dataset](https://zenodo.org/records/7153326)

   2.2. **brats:** [The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2) (Download data of task1)

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

## Data description

### ISLES

The ISLES dataset focuses on acute ischemic stroke. Each subject includes three MRI sequences commonly used in stroke assessment: **FLAIR**, **DWI (b=1000)**, and the corresponding **ADC** map. Expert-annotated stroke lesion masks are provided for every case.  
All files follow the **BIDS** structure and are released in **NIfTI (.nii)** format. Images are kept in **native acquisition space** (no registration or spatial normalization). To ensure anonymity, all scans were **skull-stripped** before release.

- **Samples:** 250  
- **Modalities:** FLAIR, DWI, ADC  
- **Labels:** Stroke lesion segmentation  
- **Format:** `.nii` files  
- **Notes:** Clinical MRI with heterogeneous resolution

### BRATS

BRATS is a widely used dataset for **glioma brain tumor segmentation**. Each subject includes four standard MRI sequences: **T1**, **T1ce** (contrast-enhanced), **T2**, and **FLAIR**. The dataset provides expert delineations of three tumor regions: **Enhancing Tumor (ET)**, **Tumor Core (TC)**, and **Whole Tumor (WT)**.  
All scans are preprocessed in a consistent manner: **skull-stripping**, **co-registration**, **resampling to 1 mm³ isotropic**, and **intensity normalization**, which makes the dataset uniform across patients and centers. Data are distributed in **NIfTI (.nii)** format.

- **Samples:** varies by edition (~300 to >2000)  
- **Modalities:** T1, T1ce, T2, FLAIR  
- **Labels:** ET, TC, WT tumor subregions  
- **Format:** `.nii` files  
- **Notes:** Fully preprocessed, ready for direct use in segmentation pipelines



## Python version

Oussama: I am using 3.11.9. Factorizer requires at least >= 3.10.