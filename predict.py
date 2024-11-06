"""

SSH로 실행할 코드,

모델을 예측해서 저장할 것.
코드 분리 없이 작성.

record.py가 실행된 후
데이터가 저장되어있는 상태에서 진행
"""

import os
import argparse
import torch
import net
import numpy as np

parser = argparse.ArgumentParser()

# data options
parser.add_argument("--subject", type=int, default=31, help="Subject number")
parser.add_argument("--action", type=int, default=10, help="Action number")
parser.add_argument("--file_idx", type=int, default=1, help="Starting file index")
parser.add_argument("--repeat", type=int, default=10, help="the total number of files")
parser.add_argument("--save_path", type=str, default="./data/", help="Path to save data")
parser.add_argument("--save_gen_path", type=str, default="./data_gen/", help="Path to save data")

# Model options
parser.add_argument("--model_path", type=str, default="./net/", help="Path to the model weight")
parser.add_argument("--model_epoch", type=int, default=15)
parser.add_argument("--train_sub", nargs="+", default=[11, 21, 12])
parser.add_argument("--feature_dim", type=int, default=16, help="")
parser.add_argument("--model_name", type=str, default="Model_07", help="")

# Parse arguments
args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda"
pool = torch.nn.MaxPool3d((1, 2, 2), (1, 2, 2)).to(device)


print(f"Train subjects: {args.train_sub}, {args.model_epoch} epoch(s), Test subject: {args.subject}")


classifier = net.resnet3d_18(in_channels=args.feature_dim, num_classes=11, layers=[2, 2, 2, 2]).to(device)
generator = net.gen_Model_07(in_channels=3, out1_channels=3, out2_channels=args.feature_dim).to(device)

pth_path = "saved_model"
pth_cls_name = f'{args.model_name}_{"_".join(map(str, args.train_sub))}_{args.model_epoch:03}_cls.pth'
pth_mtl_name = f'{args.model_name}_{"_".join(map(str, args.train_sub))}_{args.model_epoch:03}_mlt.pth'

# load
cls_pth = torch.load(os.path.join(pth_path, pth_cls_name))
gen_pth = torch.load(os.path.join(pth_path, pth_mtl_name))
generator.load_state_dict(gen_pth)
classifier.load_state_dict(cls_pth)

generator.eval()
classifier.eval()
for file_idx in range(args.file_idx, args.file_idx + args.repeat):
    frame_name = f"SubClsIdx_{args.subject:03}_{args.action:03}_{file_idx:03}_frm.npy"  # for loading and saving
    event_name = f"SubClsIdx_{args.subject:03}_{args.action:03}_{file_idx:03}_evt.npy"  # for saving
    label_name = f"SubClsIdx_{args.subject:03}_{args.action:03}_{file_idx:03}_cls.npy"  # for saving
    frame_path = os.path.join(args.save_path, f'{args.subject:03}',  frame_name)
    frame = np.load(frame_path)
    frame = torch.tensor(frame).unsqueeze(0)

    with torch.no_grad():
        frame = frame.permute(0, 4, 1, 2, 3).float().contiguous()
        frame = frame.to(device)

        # pooling
        frame = pool(frame)

        # Forward
        event_generated, cls_feature = generator(frame)
        label_pred = classifier(cls_feature)



    save_path = args.save_gen_path
    new_dir_path = os.path.join(save_path, f'{args.subject:03}')
    os.makedirs(new_dir_path, exist_ok=True)
    event_save_path = os.path.join(save_path, f'{args.subject:03}', event_name)
    label_save_path = os.path.join(save_path, f'{args.subject:03}', label_name)


    # change label
    label_pred = label_pred.argmax()

    # reshape, permute and rescale
    event_generated = event_generated.squeeze(0)
    label_pred = label_pred.squeeze(0)
    event_generated = event_generated.permute(1, 2, 3, 0) * 255

    # save
    np.save(event_save_path, np.array(event_generated.to("cpu")))
    np.save(label_save_path, np.array(label_pred.to("cpu")))
    print(event_save_path)
    print(label_save_path)


