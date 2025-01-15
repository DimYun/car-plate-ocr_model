"""Lightning module for create dataset"""
import os
from typing import Optional, Tuple, Union

import albumentations as albu
import jpeg4py as jpeg
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

TRANSFORM_TYPE = Union[albu.BasicTransform, albu.BaseCompose]


class PlatesCodeDataset(Dataset):
    """Plates dataset module"""
    def __init__(
        self,
        data_folder: str,
        phase: str = "test",
        transforms: Optional[TRANSFORM_TYPE] = None,
        reset_flag: bool = False,
    ):
        self.data_folder = data_folder
        self.phase = phase
        self.transforms = transforms
        (image_paths, plate_numbers, regions) = (
            prepare_and_read_annotations(data_folder, phase, reset_flag)
        )
        self.image_paths = image_paths
        self.plate_numbers = plate_numbers
        self.regions = regions

    def __getitem__(self, idx: int) -> Tuple:
        """
        Function for getting a single data point
        :param idx: index of data point
        :return: tuple with image, test, text length and region
        """
        image_filename = self.image_paths[idx]
        plate_number = self.plate_numbers[idx]
        region = self.regions[idx]
        image_path = os.path.join(
            self.data_folder, "dataset-plates", region, self.phase, image_filename
        )
        image = jpeg.JPEG(image_path).decode()

        prep_data = {
            "image": image,
            "text": plate_number,  # 'region': region,
            "text_length": len(plate_number),
        }

        if self.transforms:
            prep_data = self.transforms(**prep_data)

        return prep_data["image"], prep_data["text"], prep_data["text_length"], region

    def __len__(self) -> int:
        """
        Function for getting the length of the dataset
        :return: length of dataset
        """
        return self.image_paths.shape[0]


def prepare_and_read_annotations(
    data_folder: str, phase: str = "train", reset_flag: bool = False
) -> Tuple:
    """
    Get data from disk and prepare it for dataset
    :param data_folder: string with path to dataset
    :param phase: string with type of dataset to load
    :param reset_flag: flag for read prepared data
    :return: tuple with numpy arrays
    """
    image_paths = []
    plate_numbers = []
    regions = []
    full_image_path = os.path.join(data_folder, f"{phase}_image_paths.npy")
    if (
        os.path.isfile(full_image_path)
        and not reset_flag
    ):
        with open(full_image_path, "rb") as f:
            image_paths = np.load(f)
        with open(os.path.join(data_folder, f"{phase}_plate_numbers.npy"), "rb") as f:
            plate_numbers = np.load(f)
        with open(os.path.join(data_folder, f"{phase}_regions.npy"), "rb") as f:
            regions = np.load(f)
    else:
        noisy_data = []
        with open(
                os.path.join(data_folder, f"{phase}_noisy_data.txt"),
                'r',
                encoding="utf-8"
        ) as fin:
            for line in fin:
                line = line.strip()
                noisy_data.append(line.split("/")[-1])
        noisy_data = list(set(noisy_data))
        annot_path = os.path.join(data_folder, f"annotations/{phase}")
        annot_filenames = os.listdir(annot_path)
        annot_filenames = list(filter(lambda x: x.endswith("_ref"), annot_filenames))
        annot_filenames.sort()
        counter = 1
        for a_file in annot_filenames:
            print(f"Parse file {a_file}, {counter} / {len(annot_filenames)}")
            counter += 1
            region = (
                a_file.replace("meta_", "").replace("_ref", "").replace("_test", "")
            )
            with open(
                    os.path.join(annot_path, a_file),
                    "r",
                    encoding='utf-8',
            ) as a_fin:
                for line in tqdm(a_fin.readlines(), position=0, desc="Images"):
                    image_filename, plate_number = line.strip().split("\t")
                    image_path = os.path.join(
                        data_folder, "dataset-plates", region, phase, image_filename
                    )
                    try:
                        image = jpeg.JPEG(image_path).decode()
                    except (AttributeError, jpeg.JPEGRuntimeError) as err:
                        print(f"Error: {err}, {image_filename}")
                        noisy_data.append(image_filename)
                        continue
                    if (
                            image_filename
                            not in noisy_data
                    ):  # data has images with low resolution and irrelevant text
                        image_paths.append(image_filename)
                        plate_numbers.append(plate_number)
                        regions.append(region)
                    else:
                        noisy_data.append(image_filename)
            with open(
                    os.path.join(data_folder, f"{phase}_noisy_data.txt"),
                    "w",
                    encoding="utf-8"
            ) as f:
                f.write("\n".join(noisy_data))
        with open(
                os.path.join(data_folder, f"{phase}_image_paths.npy"),
                "wb",
                encoding="utf-8",
        ) as f:
            image_paths = np.array(image_paths)
            np.save(f, image_paths)
        with open(
                os.path.join(data_folder, f"{phase}_plate_numbers.npy"),
                "wb",
                encoding="utf-8",
        ) as f:
            plate_numbers = np.array(plate_numbers)
            np.save(f, plate_numbers)
        with open(
                os.path.join(data_folder, f"{phase}_regions.npy"),
                "wb",
                encoding="utf-8",
        ) as f:
            regions = np.array(regions)
            np.save(f, regions)
    return image_paths, plate_numbers, regions
