# Image-2-SMILES System Summary and Future Work
This document serves to summarize the work of the image2SMILES system, which comprises of Chem-Detectron2 and Chem-OSCAR separately, as well as some thoughts on future work.

## Flow of the two-part system
1. The images of chemical structure diagrams (in the forms of image files such as .png) are fed into Chem-Detectron2. Bounding boxes, image features of bounding boxes
as well as labels of bounding boxes are generated.
2. The image features of bounding boxes of each image is fed into Chem-OSCAR. Predicted SMILES are generated.

## Origins of the system
Both parts of the system are inspired by open-source research systems that perform image-related tasks on normal images, not chemistry-specific image processing tasks.
Both systems have no knowledge of the chemistry domain.
1. Chem-Detectron2 is based on an object detection framework called Detectron2 by FAIR (Facebook AI Research).
    - The official repo is here: [Detectron2](https://github.com/facebookresearch/detectron2)
    - A really good 5-part blog post explaining Detectron2 is here: [medium-Detectron2](https://medium.com/@hirotoschwert/digging-into-detectron-2-47b2e794fabd)
    - A version of Detectron2 that is modified to do chemical-related SMILES tasks is here: [DACON](https://github.com/Saurabh-Bagchi/LG_SMILES_1st)
3. Chem-OSCAR is based on an multimodal model that can do tasks such as image captioning, visual Q&A that work with both image and text data. 
    - The official repo is here: [OSCAR](https://github.com/microsoft/Oscar) 
    - The paper is here: [OSCAR-paper](https://arxiv.org/abs/2004.06165)

## Inspirations of the system
Image-2-SMILES and image captioning, a very popular and well researched task in the field of multimodal machine learning, share some underlying similarities. 
First, both tasks require an AI system to take image files as inputs and generate string descriptions as outputs.
Second, in order to generate relevant and correct captions, image captioning systems need to "understand" objects, relationships and actions in the pictures, i.e. 
an ability to identify important regions and capture the concepts and meanings of the important regions. This is similar to image-2-SMILES in that that task
also requires the system to identify sub-structures in the image and capture the chemical information of those sub-structures.

Image captioning systems nowadays rely heavily on a machine learning framework called transformers. A post that explains Transformers is [here](https://medium.com/inside-machine-learning/what-is-a-transformer-d07dd1fbec04).
Often, they attach a visual backbone to pre-process the image first to generate image features as inputs to the transformer network. A popular choice for the visual backbone is 
object detection networks. Detectron2 by Facebook is a common choice for that purpose. 

Therefore, we have the two-part system that work sequentially to do the image-2-SMILES task.

## Future Work
### Current State of the models
Chem-Detectron2 is trained with 500,000 chemical structure diagrams and Chem-OSCAR is trained with 300,000 image features of chemical structure diagrams. 
In both cases, the training set is small compared with how much Facebook and Microsoft trained their respective models in order to achieve state-of-the-art performances, which 
sit on a scale of 5 million images or more. 
When working on the demo, I discovered that both systems' performances are not stable. 
This is mostly due to the three reasons in my opinion:
1. Both systems are trained with small training sets due to time constraints.
2. Both systems are trained with a unified image format, which is not the case in actual patents.
3. The Chem-OSCAR system does not have additional information on the image features.

Therefore, I would propose the following approaches to further improve both models to get to production level ready for commercial use:
0. Train both systems with other hyperaparameters. Due to the time constraints, I was not able to try all the hyperparameters during training to figure which set of them would make the best models. This can be done in the future.
1. Train both systems with more images. The PubChem database has ~109 million chemicals as of May 2021. Training with at least 2 million images or more with both cases is required. With more data, both systems should benefit in performances.
2. Data augmentation on the images. The second point I brought up on the previous paragraph mentions that the system is not working in a stable fashion because of the image formats.
    - Enlarge the image sizes accepted by Chem-Detectron2. Currently, Chem-Detectron2 takes in image of sizes 300 * 300. This is low in resolution compared to real images in patents.
    - Visual AI Systems are easily influenced by images of different resolutions, sizes, fonts, bold/regular character weights etc,. It would useful to generate different distortions of the same image to train Chem-Detectron2. 
    - Another possible way to resolve this is to modify and transform the images from patents to a unified form before passing them into Chem-Detectron2 and consequently Chem-OSCAR. This, however, would require a new system that may or may not be AI-based. I do believe this would make the entire image-side chemical search system more robust. 
3. Improve the labeling of Chem-Detectron2 to support sub-structures. Chem-Detectron2 currently can only detect bounding boxes and label them on an atomic level. However, we can improve this by annotating the images with not atomic labels but sub-structure labels as well.
4. Change the inputs of Chem-OSCAR. Chem-OSCAR currently takes in only image features of 50 bounding boxes for each image. It does not, however, know the labels of the bounding boxes as well as the locations of the boundinx boxes with regards to the original image. Therefore, passing in labels of bounding boxes (which is supported by OSCAR currently but was not included in my work during the training) and locations of the bounding boxes would make the predicted SMILES better.
