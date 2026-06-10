import os
import json
import cv2
import torch
from torch.utils.data import Dataset


def _as_list(cities):
    if isinstance(cities, (list, tuple)):
        return list(cities)
    return [cities]


class CVGTextPhotosTrain(Dataset):
    def __init__(self,
                 data_root,
                 city,
                 transforms_query=None,
                 transforms_reference=None,
                 split="train",
                 text_base_dir=None,
                 query_base_dir=None,
                 reference_base_dir=None):
        super().__init__()
        self.data_root = data_root
        self.cities = _as_list(city)
        self.transforms_query = transforms_query
        self.transforms_reference = transforms_reference

        # CVG-Text_full defaults
        self.text_base_dir = text_base_dir or os.path.join(data_root, "annotation", "texts")
        self.query_base_dir = query_base_dir or os.path.join(data_root, "data", "query")
        self.reference_base_dir = reference_base_dir or os.path.join(
            data_root,
            "mnt",
            "hwfile",
            "opendatalab",
            "air",
            "linhonglin",
            "CVG-text",
            "reference",
        )

        self.samples = []
        self.city_stats = {}

        for c in self.cities:
            text_path = os.path.join(self.text_base_dir, c, f"{split}.json")
            with open(text_path, "r", encoding="utf-8") as f:
                text_map = json.load(f)

            ground_dir = os.path.join(self.query_base_dir, f"{c}-ground")
            sat_dir = os.path.join(self.reference_base_dir, f"{c}-satellite")

            total = len(text_map)
            matched = 0
            for fname, text in text_map.items():
                g_path = os.path.join(ground_dir, fname)
                s_path = os.path.join(sat_dir, fname)
                if os.path.exists(g_path) and os.path.exists(s_path):
                    self.samples.append((f"{c}:{fname}", g_path, s_path, text))
                    matched += 1

            self.city_stats[c] = {"total": total, "matched": matched, "dropped": total - matched}

        self.idx2label = {s[0]: i for i, s in enumerate(self.samples)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_id, g_path, s_path, text = self.samples[index]

        q_img = cv2.imread(g_path)
        q_img = cv2.cvtColor(q_img, cv2.COLOR_BGR2RGB)

        r_img = cv2.imread(s_path)
        r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)

        if self.transforms_query is not None:
            q_img = self.transforms_query(image=q_img)["image"]

        if self.transforms_reference is not None:
            r_img = self.transforms_reference(image=r_img)["image"]

        label = torch.tensor(self.idx2label[sample_id], dtype=torch.long)
        return q_img, r_img, text, label


class CVGTextPhotosEval(Dataset):
    def __init__(self,
                 data_root,
                 city,
                 split="test",
                 img_type="query",
                 transforms=None,
                 text_base_dir=None,
                 query_base_dir=None,
                 reference_base_dir=None):
        super().__init__()
        self.data_root = data_root
        self.city = city
        self.transforms = transforms
        self.img_type = img_type

        self.text_base_dir = text_base_dir or os.path.join(data_root, "annotation", "texts")
        self.query_base_dir = query_base_dir or os.path.join(data_root, "data", "query")
        self.reference_base_dir = reference_base_dir or os.path.join(
            data_root,
            "mnt",
            "hwfile",
            "opendatalab",
            "air",
            "linhonglin",
            "CVG-text",
            "reference",
        )

        text_path = os.path.join(self.text_base_dir, city, f"{split}.json")
        with open(text_path, "r", encoding="utf-8") as f:
            text_map = json.load(f)

        ground_dir = os.path.join(self.query_base_dir, f"{city}-ground")
        sat_dir = os.path.join(self.reference_base_dir, f"{city}-satellite")

        self.items = []
        self.labels = []
        valid_names = []

        for fname in text_map.keys():
            if os.path.exists(os.path.join(ground_dir, fname)) and os.path.exists(os.path.join(sat_dir, fname)):
                valid_names.append(fname)

        valid_names.sort()
        self.name2label = {n: i for i, n in enumerate(valid_names)}

        for n in valid_names:
            if img_type == "query":
                self.items.append(os.path.join(ground_dir, n))
            elif img_type == "reference":
                self.items.append(os.path.join(sat_dir, n))
            else:
                raise ValueError("img_type must be query or reference")
            self.labels.append(self.name2label[n])

        self.stats = {"total": len(text_map), "matched": len(valid_names), "dropped": len(text_map) - len(valid_names)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        p = self.items[index]
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            img = self.transforms(image=img)["image"]
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return img, label
