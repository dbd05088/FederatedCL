# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import argparse
import sys
import json
import ast
import os
import sys
from typing import Optional
import cv2
import torch
from PIL import Image
import mmcv
from mmdet.core.visualization.image import imshow_det_bboxes
import numpy as np
import pycocotools.mask as maskUtils
import matplotlib.pyplot as plt

from transformers import CLIPProcessor, CLIPModel
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from transformers import BlipProcessor, BlipForConditionalGeneration
from cog import BasePredictor, Input, Path, BaseModel

sys.path.append("..")
from segment_anything import SamAutomaticMaskGenerator

sys.path.insert(0, "scripts")
from configs.ade20k_id2label import CONFIG as CONFIG_ADE20K_ID2LABEL
from configs.coco_id2label import CONFIG as CONFIG_COCO_ID2LABEL
from clip import clip_classification
from clipseg import clipseg_segmentation
from oneformer import oneformer_coco_segmentation, oneformer_ade20k_segmentation
from blip import open_vocabulary_classification_blip

from mobilesamv2.promt_mobilesamv2 import ObjectAwareModel
from mobilesamv2 import sam_model_registry, SamPredictor
from typing import Any, Dict, Generator,List
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ObjectAwareModel_path", type=str, default='./PromptGuidedDecoder/ObjectAwareModel.pt', help="ObjectAwareModel path")
    parser.add_argument("--Prompt_guided_Mask_Decoder_path", type=str, default='./PromptGuidedDecoder/Prompt_guided_Mask_Decoder.pt', help="Prompt_guided_Mask_Decoder path")
    parser.add_argument("--encoder_path", type=str, default="./", help="select your own path")
    parser.add_argument("--img_path", type=str, default="./test_images/", help="path to image file")
    parser.add_argument("--imgsz", type=int, default=1024, help="image size")
    parser.add_argument("--iou",type=float,default=0.9,help="yolo iou")
    parser.add_argument("--conf", type=float, default=0.3, help="yolo object confidence threshold")
    parser.add_argument("--retina",type=bool,default=True, help="draw segmentation masks",)
    parser.add_argument("--output_dir", type=str, default="./", help="image save path")
    parser.add_argument("--encoder_type", choices=['tiny_vit','sam_vit_h','mobile_sam','efficientvit_l2','efficientvit_l1','efficientvit_l0'], help="choose the model type")
    return parser.parse_args()

def create_model(Prompt_guided_path, obj_model_path):
    ObjAwareModel = ObjectAwareModel(obj_model_path)
    PromptGuidedDecoder=sam_model_registry['PromptGuidedDecoder'](Prompt_guided_path)
    mobilesamv2 = sam_model_registry['vit_h']()
    mobilesamv2.prompt_encoder=PromptGuidedDecoder['PromtEncoder']
    mobilesamv2.mask_decoder=PromptGuidedDecoder['MaskDecoder']
    return mobilesamv2,ObjAwareModel
    
def show_anns(anns):
    if len(anns) == 0:
        return
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((anns.shape[1], anns.shape[2], 4))
    img[:,:,3] = 0
    for ann in range(anns.shape[0]):
        m = anns[ann].bool()
        m=m.cpu().numpy()
        color_mask = np.concatenate([np.random.random(3), [1]])
        img[m] = color_mask
    ax.imshow(img)

def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]

encoder_path={'efficientvit_l2':'../weight/l2.pt',
            'tiny_vit':'../weight/mobile_sam.pt',
            'sam_vit_h':'../weight/sam_vit_h.pt',
            'mobile_sam':'../weight/mobile_sam.pt',
            }


MODEL_CACHE = "model_cache"


class ModelOutput(BaseModel):
    json_out: Optional[Path]
    img_out: Path


class Predictor(BasePredictor):
    
    def setup(self, sam):
        """Load the model into memory to make running multiple predictions efficient"""
        sam_checkpoint = "cls_predictor.predict(image = args.img_path + image_name)"
        model_type = "default"
        self.sam = sam #sam_model_registry[model_type](checkpoint=sam_checkpoint).to("cuda")
        self.generator = SamAutomaticMaskGenerator(self.sam, output_mode="coco_rle")

        # semantic segmentation
        rank = 0
        # the following models are pre-downloaded and cached to MODEL_CACHE to speed up inference 
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14",
            cache_dir=MODEL_CACHE,
        )
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14",
            cache_dir=MODEL_CACHE,
        ).to(rank)

        self.oneformer_ade20k_processor = OneFormerProcessor.from_pretrained(
            "shi-labs/oneformer_ade20k_swin_large",
            cache_dir=MODEL_CACHE,
        )
        self.oneformer_ade20k_model = OneFormerForUniversalSegmentation.from_pretrained(
            "shi-labs/oneformer_ade20k_swin_large",
            cache_dir=MODEL_CACHE,
        ).to(rank)

        self.oneformer_coco_processor = OneFormerProcessor.from_pretrained(
            "shi-labs/oneformer_coco_swin_large",
            cache_dir=MODEL_CACHE,
        )
        self.oneformer_coco_model = OneFormerForUniversalSegmentation.from_pretrained(
            "shi-labs/oneformer_coco_swin_large",
            cache_dir=MODEL_CACHE,
        ).to(rank)

        self.blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-large",
            cache_dir=MODEL_CACHE,
        )
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large",
            cache_dir=MODEL_CACHE,
        ).to(rank)

        self.clipseg_processor = AutoProcessor.from_pretrained(
            "CIDAS/clipseg-rd64-refined",
            cache_dir=MODEL_CACHE,
        )
        self.clipseg_model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined",
            cache_dir=MODEL_CACHE,
        ).to(rank)
        self.clipseg_processor.image_processor.do_resize = False

    def predict(
        self,
        image: Path = Input(description="Input image"),
        base_image_name: Path = Input(description="ex. sample.png => sample"),
        output_path: Path = Input(description="Output image"),
        output_json: bool = Input(default=True, description="return raw json output"),
    ) -> ModelOutput:
        """Run a single prediction on the model"""
        seg_json = os.path.join(output_path, "seg_json", base_image_name + ".json")
        json_out = os.path.join(output_path, "seg_out", base_image_name + "_out.json") #"tmp/seg_out.json"
        seg_out = os.path.join(output_path, "seg_images", base_image_name + "_out.png")#"tmp/seg_out.png"

        if os.path.exists(seg_out):
            print("exist", image)
            return
        else:
            print("not exist", image)

        img = cv2.imread(str(image))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        masks = self.generator.generate(img)

        with open(seg_json, "w") as f:
            json.dump(masks, f)



        semantic_annotation_pipeline(
            seg_json,
            str(image),
            json_out,
            seg_out,
            clip_processor=self.clip_processor,
            clip_model=self.clip_model,
            oneformer_ade20k_processor=self.oneformer_ade20k_processor,
            oneformer_ade20k_model=self.oneformer_ade20k_model,
            oneformer_coco_processor=self.oneformer_coco_processor,
            oneformer_coco_model=self.oneformer_coco_model,
            blip_processor=self.blip_processor,
            blip_model=self.blip_model,
            clipseg_processor=self.clipseg_processor,
            clipseg_model=self.clipseg_model,
        )

        return ModelOutput(
            json_out=Path(json_out) if output_json else None, img_out=Path(seg_out)
        )


def semantic_annotation_pipeline(
    seg_json,
    image,
    json_out,
    seg_out,
    rank=0,
    scale_small=1.2,
    scale_large=1.6,
    clip_processor=None,
    clip_model=None,
    oneformer_ade20k_processor=None,
    oneformer_ade20k_model=None,
    oneformer_coco_processor=None,
    oneformer_coco_model=None,
    blip_processor=None,
    blip_model=None,
    clipseg_processor=None,
    clipseg_model=None,
):
    anns = mmcv.load(seg_json)
    img = mmcv.imread(image)
    bitmasks, class_names = [], []
    class_ids_from_oneformer_coco = oneformer_coco_segmentation(
        Image.fromarray(img), oneformer_coco_processor, oneformer_coco_model, 0
    )
    class_ids_from_oneformer_ade20k = oneformer_ade20k_segmentation(
        Image.fromarray(img), oneformer_ade20k_processor, oneformer_ade20k_model, 0
    )

    for ann in anns:
        valid_mask = torch.tensor(maskUtils.decode(ann["segmentation"])).bool()
        # get the class ids of the valid pixels
        coco_propose_classes_ids = class_ids_from_oneformer_coco[valid_mask]
        ade20k_propose_classes_ids = class_ids_from_oneformer_ade20k[valid_mask]
        top_k_coco_propose_classes_ids = (
            torch.bincount(coco_propose_classes_ids.flatten()).topk(1).indices
        )
        top_k_ade20k_propose_classes_ids = (
            torch.bincount(ade20k_propose_classes_ids.flatten()).topk(1).indices
        )
        local_class_names = set()
        local_class_names = set.union(
            local_class_names,
            set(
                [
                    CONFIG_ADE20K_ID2LABEL["id2label"][str(class_id.item())]
                    for class_id in top_k_ade20k_propose_classes_ids
                ]
            ),
        )
        local_class_names = set.union(
            local_class_names,
            set(
                (
                    [
                        CONFIG_COCO_ID2LABEL["refined_id2label"][str(class_id.item())]
                        for class_id in top_k_coco_propose_classes_ids
                    ]
                )
            ),
        )
        patch_small = mmcv.imcrop(
            img,
            np.array(
                [
                    ann["bbox"][0],
                    ann["bbox"][1],
                    ann["bbox"][0] + ann["bbox"][2],
                    ann["bbox"][1] + ann["bbox"][3],
                ]
            ),
            scale=scale_small,
        )
        patch_large = mmcv.imcrop(
            img,
            np.array(
                [
                    ann["bbox"][0],
                    ann["bbox"][1],
                    ann["bbox"][0] + ann["bbox"][2],
                    ann["bbox"][1] + ann["bbox"][3],
                ]
            ),
            scale=scale_large,
        )
        patch_huge = mmcv.imcrop(
            img,
            np.array(
                [
                    ann["bbox"][0],
                    ann["bbox"][1],
                    ann["bbox"][0] + ann["bbox"][2],
                    ann["bbox"][1] + ann["bbox"][3],
                ]
            ),
            scale=scale_large,
        )
        valid_mask_huge_crop = mmcv.imcrop(
            valid_mask.numpy(),
            np.array(
                [
                    ann["bbox"][0],
                    ann["bbox"][1],
                    ann["bbox"][0] + ann["bbox"][2],
                    ann["bbox"][1] + ann["bbox"][3],
                ]
            ),
            scale=scale_large,
        )
        op_class_list = open_vocabulary_classification_blip(
            patch_large, blip_processor, blip_model, rank
        )
        local_class_list = list(
            set.union(local_class_names, set(op_class_list))
        )  # , set(refined_imagenet_class_names)
        mask_categories = clip_classification(
            patch_small,
            local_class_list,
            3 if len(local_class_list) > 3 else len(local_class_list),
            clip_processor,
            clip_model,
            rank,
        )
        class_ids_patch_huge = clipseg_segmentation(
            patch_huge, mask_categories, clipseg_processor, clipseg_model, rank
        ).argmax(0)
        top_1_patch_huge = (
            torch.bincount(
                class_ids_patch_huge[torch.tensor(valid_mask_huge_crop)].flatten()
            )
            .topk(1)
            .indices
        )
        top_1_mask_category = mask_categories[top_1_patch_huge.item()]

        ann["class_name"] = str(top_1_mask_category)
        ann["class_proposals"] = mask_categories
        class_names.append(ann["class_name"])
        bitmasks.append(maskUtils.decode(ann["segmentation"]))

    mmcv.dump(anns, json_out)
    imshow_det_bboxes(
        img,
        bboxes=None,
        labels=np.arange(len(bitmasks)),
        segms=np.stack(bitmasks),
        class_names=class_names,
        font_size=25,
        show=False,
        out_file=seg_out,
    )

def main(args):
    # import pdb;pdb.set_trace()
    output_dir=args.output_dir  
    os.makedirs(os.path.join(output_dir, "seg_images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "seg_json"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "seg_out"), exist_ok=True)
    mobilesamv2, ObjAwareModel=create_model(args.Prompt_guided_Mask_Decoder_path, args.ObjectAwareModel_path)
    image_encoder=sam_model_registry[args.encoder_type](encoder_path[args.encoder_type])
    mobilesamv2.image_encoder=image_encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mobilesamv2.to(device=device)
    mobilesamv2.eval()
    cls_predictor = Predictor()
    cls_predictor.setup(mobilesamv2)
    image_files= os.listdir(args.img_path)
    for image_name in image_files:
        cls_predictor.predict(image = os.path.join(args.img_path, image_name), base_image_name = image_name.split(".")[-2], output_path = output_dir)
        #plt.savefig("{}".format(os.path.join(output_dir, image_name)), bbox_inches='tight', pad_inches = 0.0)

if __name__ == "__main__":
    args = parse_args()
    main(args)
