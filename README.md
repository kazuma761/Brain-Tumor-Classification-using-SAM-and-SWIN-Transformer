# Brain Tumor Classification using SAM and SWIN Transformer

This project implements a brain tumor classification system using SAM (Segment Anything Model) and SWIN Transformer.

## Model Files
Due to size limitations on GitHub, the following model files are not included in the repository:
- `best_swin_transformer_model.pth` (319.68 MB)
- `sam_vit_b_01ec64.pth` (357.67 MB)

Please download these files separately from:
- SAM model: [Download from Meta AI](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
- For the SWIN transformer model, please train the model using the provided notebook.

## Project Structure
- `Brain_Tumor_Classification_Swin_Transformer.ipynb`: Main notebook for training and evaluation
- `complete_prediction_standalone.py`: Standalone prediction script
- `single_image_prediction.py`: Script for single image prediction
- `Testing/`: Test dataset
- `Training/`: Training dataset
