"""
This script transforms CelebA images to 64x64 resolution using the torchvision transforms. This is useful for computing the FID score as the model is trained on 64x64 images.
"""

import os
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import (
    Compose,
    ToTensor,
    ToPILImage,
    CenterCrop,
    Resize,
    RandomHorizontalFlip,
)

# Define the transform
image_size = 64
transform_train = Compose(
    [
        Resize(image_size),
        RandomHorizontalFlip(),
        CenterCrop(image_size),
        ToTensor(),
    ]
)


def transform_and_save_images(input_dir, output_dir, transform):
    """
    Apply the given transform on all images in the input directory and save in the output directory.

    Parameters:
    - input_dir: Directory containing the original images.
    - output_dir: Directory where the transformed images will be saved.
    - transform: torchvision transform to apply on images.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all files in the input directory
    image_files = [
        f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))
    ]

    for image_file in tqdm(image_files, desc="Processing images", unit="image"):
        # Check if the file is an image
        if image_file.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(input_dir, image_file)
            image = Image.open(image_path)

            # Apply transform
            transformed_image = transform(image)

            # Convert tensor back to PIL Image
            transformed_image = ToPILImage()(transformed_image)

            # Save the transformed image
            output_path = os.path.join(output_dir, image_file)
            transformed_image.save(output_path)

    print(f"Transformed images saved to {output_dir}")


# Sample usage
input_directory = "../data/img_align_celeba"
output_directory = "../data/img_align_celeba_64x64"
transform_and_save_images(input_directory, output_directory, transform_train)
