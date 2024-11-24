import torch
import os
import numpy as np
import cv2
import argparse
import albumentations as A
import segmentation_models_pytorch as smp
from albumentations.pytorch.transforms import ToTensorV2

def main():
    # Tạo đối tượng ArgumentParser
    parser = argparse.ArgumentParser(description="Process an image file.")

    # Thêm tham số image_path
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image file')

    # Parse các tham số
    args = parser.parse_args()

    # Lấy giá trị của image_path
    image_path = args.image_path

    def infer(image):

        model = smp.UnetPlusPlus(
            encoder_name="resnet34",        
            encoder_weights="imagenet",     
            in_channels=3,                  
            classes=3)

        checkpoint = torch.load('/kaggle/working/model.pth')
        model.load_state_dict(checkpoint['model'])

        val_transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        color_dict= {0: (0, 0, 0),
                    1: (255, 0, 0),
                    2: (0, 255, 0)}
        def mask_to_rgb(mask, color_dict):
            output = np.zeros((mask.shape[0], mask.shape[1], 3))

            for k in color_dict.keys():
                output[mask==k] = color_dict[k]

            return np.uint8(output)

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.eval()
        dir_path = os.getcwd()

        img_path = os.path.join(dir_path, image)
        ori_img = cv2.imread(img_path)
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        ori_w = ori_img.shape[0]
        ori_h = ori_img.shape[1]
        img = cv2.resize(ori_img, (256, 256))
        transformed = val_transform(image=img)
        input_img = transformed["image"]
        input_img = input_img.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output_mask = model(input_img).squeeze(0).cpu().numpy().transpose(1,2,0)
        mask = cv2.resize(output_mask, (ori_h, ori_w))
        mask = np.argmax(mask, axis=2)
        mask_rgb = mask_to_rgb(mask, color_dict)
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite("{}output".format(image), mask_rgb)

    infer(image_path)

if __name__ == "__main__":
    main()
