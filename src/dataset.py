import os
from typing import Union, Optional

import albumentations as albu
import cv2
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import jpeg4py as jpeg

TRANSFORM_TYPE = Union[albu.BasicTransform, albu.BaseCompose]


class PlatesCodeDataset(Dataset):
    def __init__(
            self,
            data_folder: str,
            phase: str = 'test',
            transforms: Optional[TRANSFORM_TYPE] = None,
            reset_flag: bool = False
    ):
        self.data_folder = data_folder
        self.phase = phase
        self.transforms = transforms
        (
            self.image_paths,
            self.plate_numbers,
            self.regions
        ) = prepare_and_read_annotations(
            data_folder,
            phase,
            reset_flag
        )

    def __getitem__(self, idx):
        image_filename = self.image_paths[idx]
        plate_number = self.plate_numbers[idx]
        region = self.regions[idx]
        # print(image_filename)
        image_path = os.path.join(
            self.data_folder, 'dataset-plates', region, self.phase, image_filename
        )
        image = jpeg.JPEG(image_path).decode()

        data = {
            'image': image,
            'text': plate_number,
            # 'region': region,
            'text_length': len(plate_number),
        }

        if self.transforms:
            data = self.transforms(**data)

        return data['image'], data['text'], data['text_length'], region

    def __len__(self):
        return self.image_paths.shape[0]


def prepare_and_read_annotations(
        data_folder: str,
        phase: str = 'train',
        reset_flag: bool = False
) -> set:
    image_paths = []
    plate_numbers = []
    regions = []
    if os.path.isfile(os.path.join(data_folder, f'{phase}_image_paths.npy')) and not reset_flag:
        with open(os.path.join(data_folder, f'{phase}_image_paths.npy'), 'rb') as f:
            image_paths = np.load(f)
        with open(os.path.join(data_folder, f'{phase}_plate_numbers.npy'), 'rb') as f:
            plate_numbers = np.load(f)
        with open(os.path.join(data_folder, f'{phase}_regions.npy'), 'rb') as f:
            regions = np.load(f)
    else:
        noisy_data = []
        with open(os.path.join(data_folder, f"{phase}_noisy_data.txt")) as fin:
            for line in fin:
                line = line.strip()
                noisy_data.append(line.split('/')[-1])
        noisy_data = list(set(noisy_data))
        annot_path = os.path.join(data_folder, f"annotations/{phase}")
        annot_filenames = os.listdir(
            annot_path
        )
        annot_filenames = list(filter(lambda x: x.endswith("_ref"), annot_filenames))
        annot_filenames.sort()
        _counter = 1
        for a_file in annot_filenames:  # tqdm(annot_filenames, position=0, desc='Annot. File'):
            print(f'Parse file {a_file}, {_counter} / {len(annot_filenames)}')
            _counter += 1
            region = (
                a_file.replace("meta_", "")
                .replace("_ref", "")
                .replace("_test", "")
            )
            with open(os.path.join(annot_path, f"{a_file}")) as a_fin:
                for line in tqdm(a_fin.readlines(), position=0, desc='Images'):
                    image_filename, plate_number = line.strip().split("\t")
                    image_path = os.path.join(
                        data_folder, 'dataset-plates', region, phase, image_filename
                    )
                    try:
                        image = jpeg.JPEG(image_path).decode()
                        height, width = image.shape[:2]
                        if (
                                # width >= 50 and width <= 390 and
                                # height >= 10 and height <= 150 and
                                image_filename not in noisy_data
                        ):  # data has images with low resolution and irrelevant text
                            image_paths.append(image_filename)
                            plate_numbers.append(plate_number)
                            regions.append(region)
                        else:
                            noisy_data.append(image_filename)
                    except (AttributeError, jpeg.JPEGRuntimeError) as err:
                        print(f"Error: {err}, {image_filename}")
                        noisy_data.append(image_filename)
            with open(os.path.join(data_folder, f'{phase}_noisy_data.txt'), 'w') as f:
                f.write("\n".join(noisy_data))
        with open(os.path.join(data_folder, f'{phase}_image_paths.npy'), 'wb') as f:
            image_paths = np.array(image_paths)
            np.save(f, image_paths)
        with open(os.path.join(data_folder, f'{phase}_plate_numbers.npy'), 'wb') as f:
            plate_numbers = np.array(plate_numbers)
            np.save(f, plate_numbers)
        with open(os.path.join(data_folder, f'{phase}_regions.npy'), 'wb') as f:
            regions = np.array(regions)
            np.save(f, regions)
    return image_paths, plate_numbers, regions
