#!/usr/bin/env python
# coding: utf-8


import cv2 
import albumentations as A
import json
import os
import argparse
import copy
import random


def read_json_file(path) : 
    '''
    description : function takes path to json file and return data as list 
    inputs : path as (str) 
    outputs : List that contains data
    
    '''
    manifest_file = []
    with open(path) as f :
        for line in f :
            manifest_file.append(json.loads(line))
    return manifest_file




def extract_data_from_ground_truth_json(path):
    '''
        discription : This function take as input path to json file  
        input : path to json file
        output:
            -> list that contains dictionary of 
                                                image_path:(string) & annotaions(list)
            -> sorted dictionary of label mapped to names
    '''
    manifest_file = read_json_file(path)
    extracted_data = []
    class_id_to_name_map = {}
    # extract data from json file 
    for i in manifest_file :
        data = {}
        data['image_path']=i['source-ref']
        annotations =[]
# note that next time remmeber not to agnore label job name and incluse it 
        for x in i['supermarket-dataset']['annotations'] :
            annotation = {}
            annotation['top'] = x['top']
            annotation['left'] = x['left']
            annotation['height'] = x['height']
            annotation['width'] = x['width']
            annotation['class_id'] = x['class_id']

            annotations.append(annotation)
        data['annotation'] = annotations
        extracted_data.append(data)
        
        class_id_to_map = {int(k): v for k, v in i['supermarket-dataset-metadata']['class-map'].items()}
        class_id_to_name_map.update(class_id_to_map)
    return extracted_data, dict(sorted(class_id_to_name_map.items()))



def extract_bboxes_and_class_ids(ground_truth_annotations):
    '''
    Description : Because of albumation we need to get bboxes and class_ids in seperatelists
    as albumation need data in that format
    
    inputs : dict contains bbox and class_ids
    
    output : - list of lists that contains bbox
             - list of class_ids   
    '''
    bboxes = []
    class_ids = []
    for gt_bbox in ground_truth_annotations['annotation']:
        xmin = gt_bbox["left"]
        ymin = gt_bbox["top"]
        width = gt_bbox["width"]
        height = gt_bbox["height"]
        bboxes.append([xmin, ymin, width, height])
        class_ids.append(gt_bbox["class_id"])
    return bboxes, class_ids




def train_validation_test_split(labeled_data, validation_sample=0.1, test_sample=0.1):
    '''
    Description : as we deal with json file we will not be able to use sklearn train_test_split()
    instead we will create our function 
    input : - Variable that holds all data 
            - validation_sample -> float that specify percentage of validation dataset
            - test_sample -> float that specify percentage of test dataset 
    output :
            - train_data -> list 
            - test_data  -> list
            - validation_data -> list
    '''
    random.shuffle(labeled_data)
    number_of_samples = len(labeled_data)
    num_validation_images = int(number_of_samples * validation_sample)
    num_test_images = int(number_of_samples * test_sample)
    num_train_images = number_of_samples - num_validation_images - num_test_images

    train_data = labeled_data[:num_train_images]
    validation_data = labeled_data[num_train_images:num_train_images + num_validation_images]
    test_data = labeled_data[num_train_images + num_validation_images:]

    return train_data, validation_data, test_data



def generate_manifest_for_augmented_data(img_label_dict, img_filename, bboxes,
                                         class_ids,
                                         output_s3_bucket_name,
                                         data_group="train"):
    augmented_img_label_dict = copy.deepcopy(img_label_dict)

    augmented_img_label_dict["image_path"] = (
        f"s3://{output_s3_bucket_name}/prepared_data/{data_group}/images/{img_filename}"
    )
    annotations = []
    for bbox, class_id in zip(bboxes, class_ids):
        annotations.append({
            "class_id": class_id,
            "left": bbox[0],
            "top": bbox[1],
            "width": bbox[2],
            "height": bbox[3],
        })
    augmented_img_label_dict["annotations"] = annotations
    return augmented_img_label_dict


def convert_bbox_from_gt_to_coco(gt_annotation, img_width, img_height):
    xmin = gt_annotation["left"] / img_width
    ymin = gt_annotation["top"] / img_height
    xmax = (gt_annotation["left"] + gt_annotation["width"]) / img_width
    ymax = (gt_annotation["top"] + gt_annotation["height"]) / img_height
    bbox = [xmin, ymin, xmax, ymax]
    return bbox


def grouth_truth_manifest_to_tf_object_detection_annotations(ground_truth_manifest, categories):
    ground_truth_manifest
    images = []
    annotations = []
    for idx, annotation in enumerate(ground_truth_manifest):
        curr_img = {}
        curr_img["id"] = idx
        curr_img["file_name"] = annotation['image_path'].split("/")[-1]
        curr_img.update({"height": 1944, "width": 2592})
        images.append(curr_img)
#         image_size = annotation[annotation_job_name]["image_size"][0]
        width, height = 2592, 1944   #image_size["width"], image_size["height"]
        for _annotation in annotation['annotation']:
            curr_annotation = {}
            curr_annotation["image_id"] = idx
            curr_annotation["category_id"] = _annotation["class_id"]
            curr_annotation["bbox"] = convert_bbox_from_gt_to_coco(_annotation, width, height)
            annotations.append(curr_annotation)

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }


def write_coco_annotations(annotations, filename):
    """Write coco annotations into disk."""
    with open(filename, "w") as coco_annotation_output:
        json.dump(annotations, coco_annotation_output)


def augment_and_save_dataset(dataset,
                             transform,
                             input_local_images_dir,
                             output_local_images_dir,
                             output_s3_bucket_name,
                             num_augmentations_per_img=10,
                             data_group="train",
                             save_original_img=True):

    # Iterate through dataset.
    augmented_manifest = []


    for img_label_dict in dataset:
        # Get the image name from S3 image path.
        img_name = img_label_dict["image_path"].split("/")[-1]
        # Construct the path to the image locally.
        # This path you specify it as argument when use estimator.run()
        image_local_path = os.path.join(input_local_images_dir, img_name)
        # Load image in memory.
        image = cv2.imread(image_local_path)
        # Convert to RGB, because OpenCV loads images in BGR
        # See https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get bounding boxes and class ids from SageMaker Ground Truth labeling format.
        ground_truth_annotations = img_label_dict["annotation"]
        bboxes, class_ids = extract_bboxes_and_class_ids(ground_truth_annotations)

        # Save original image to put all training/validation images
        # in the folder structure expected by Amazon SageMaker Object Detection - Tensorflow.
        if save_original_img:
            # Prepare the path to save the original image.
            # This is up to your chooice
            original_img_write_path = os.path.join(output_local_images_dir, img_name)
            # Convert to BGR, because opencv2 assumes channels are in BGRorder,
            # then converts to RGB before writing an image file.
            cv2.imwrite(original_img_write_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            # Add the annotations of the original image to the augmented annotations.
            augmented_manifest.append(
                generate_manifest_for_augmented_data(
                    img_label_dict,
                    img_name,
                    bboxes,
                    class_ids,
                    output_s3_bucket_name,
                    data_group=data_group
                )
            )

        base_img_name = img_name.split(".")[0]
        # Repeat image augmentation for each original images
        # `num_augmentations_per_img` times.
        for indx in range(num_augmentations_per_img):

            # Run the transformation to generate one augmented image.
            transformed = transform(image=image, bboxes=bboxes, class_ids=class_ids)
            # Save each transformation in the specified location.
            augmented_img_filename = f"{base_img_name}_aug_{indx + 1}.jpg"
            augmented_image_write_path = os.path.join(output_local_images_dir, augmented_img_filename)
            cv2.imwrite(augmented_image_write_path, cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR))

            # We create a new labels for the augmented image, by copying the labels of the original image.
            # Then update it with the new bboxes, class_ids, etc.
            curr_augmented_manifest = generate_manifest_for_augmented_data(
                    img_label_dict,
                    augmented_img_filename,
                    transformed['bboxes'],
                    transformed['class_ids'],
                    output_s3_bucket_name,
                    data_group=data_group
            )
            augmented_manifest.append(curr_augmented_manifest)

    return augmented_manifest


# In[276]:



def write_manifest_file_as_jsonlines(data, filename):
    """Write a list of dicts in `data` as a jsonlines file to `filename`"""
    with open(filename, 'w') as outfile:
        for line in data:
            json.dump(line, outfile)
            outfile.write('\n')


# In[ ]:





# In[278]:


# Let is write our script 
'''
                                             _____ class_maper  
                                            /
read_json_file --> extract useful data only
                                            \_____ extracted_data ---> train_validate_test_split
                                                             ____________________|_______________________
                                                            /                    |                      /              
                                                            
                                                        train_data          test_data       validation_data

                                                            \____________________|___________________/
                                                                                 |
                                                            perform transformations using albumation seperately
                                                                                 |
                                                                                 |
                                                                                 v
                                                                save_each_data_to_seperate_location    
                                                        
'''


if __name__=="__main__":
    
    
    base_dir = '/opt/ml/processing'
    manifest_file_local_path = f"{base_dir}/input/manifest/output.manifest"
    input_local_images_dir = f"{base_dir}/input/images"

    

    print("Listing job inputs:")
    # This location are specified when use .run() for your estimator
    # you pass your input data location whether images or manifest 
    # printing for making sure that data has been copied succefully and no errors
    print("Manifest:", os.listdir(f"{base_dir}/input/manifest"))
    print("Raw images:", os.listdir(f"{base_dir}/input/images")[:10])

    ######################################################################################################
    # Specify arguments you need for your script 
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--num_augmentations_per_img', type=int, default=10,
                        help='An integer for the number of augmentation per image.')
    parser.add_argument('--output_s3_bucket_name', type=str,
                        help='The S3 bucket where to store the augmented images.')

    args = parser.parse_args()
    
    output_s3_bucket_name = args.output_s3_bucket_name
    # The label attribute name used in the labeling job.
    # Number of augmentations per image.
    num_augmentations_per_img = args.num_augmentations_per_img
    #######################################################################################################
    labeled_data_manifest, labels_mapper = extract_data_from_ground_truth_json(manifest_file_local_path)
    
    print(f"The number of samples in input data is {len(labeled_data_manifest)}")

    print("Peek into a sample of input data")
    print(labeled_data_manifest[0])

    ########################################################################################################
    train_data, validation_data = train_validation_test_split(
        labeled_data_manifest, validation_sample=0.1
    )
    ########################################################################################################
    ### Apply Image Augmantaion 
    transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.Perspective(),
            A.RandomBrightnessContrast(p=0.3),
            A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_ids']))
    ########################################################################################################
    augmented_train_manifest = augment_and_save_dataset(
        train_data,
        transform,
        input_local_images_dir,
        f"{base_dir}/output/train/images/",
        output_s3_bucket_name,
        num_augmentations_per_img=num_augmentations_per_img,
        data_group="train",
        save_original_img=True)

    # Augment validation data.
    augmented_validation_manifest = augment_and_save_dataset(
        validation_data,
        transform,
        input_local_images_dir,
        f"{base_dir}/output/validation/images/",
        output_s3_bucket_name,
        num_augmentations_per_img=num_augmentations_per_img,
        data_group="validation",
        save_original_img=True)
    # # Augment test data.
    # augmented_test_manifest = augment_and_save_dataset(
    #     test_data,
    #     transform,
    #     input_local_images_dir,
    #     f"{base_dir}/output/augmented_test_images/",
    #     output_s3_bucket_name,
    #     num_augmentations_per_img=num_augmentations_per_img,
    #     data_group="test",
    #     save_original_img=True
    # )
    ##########################################################################################################
    # Write the three datasets into separate manifest files.
    train_data += augmented_train_manifest
    train_data = grouth_truth_manifest_to_tf_object_detection_annotations(train_data, labels_mapper)
    write_manifest_file_as_jsonlines(train_data, f"{base_dir}/output/train/annotations.json")

    validation_data += augmented_validation_manifest
    validation_data = grouth_truth_manifest_to_tf_object_detection_annotations(validation_data, labels_mapper)
    validation_data = write_manifest_file_as_jsonlines(validation_data,f"{base_dir}/output/validation/annotations.json")
    
    # test_data += augmented_test_manifest
    # write_manifest_file_as_jsonlines(test_data, f"{base_dir}/output/manifests/test.manifest")
    # Augment the images in train set.
    # augment the images in validation set.
    # Save augmented images to specified locations in S3.
    print("Finished data prepration job successfully.")