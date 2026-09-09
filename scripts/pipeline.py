"""
联合推理 Pipeline：RDNet（去反射）→ 增强（提亮）

用法：
  python3 scripts/pipeline.py /path/to/images [output_dir] [--enhance clahe|gamma|both]

流程：
  1. RDNet 分离透射层 T 和反射层 R
  2. 对 T 做低光增强（CLAHE / Gamma / 两者）
  3. 保存所有中间结果和最终结果
"""
import os
import sys
import argparse
import warnings

# 将 RDNet 目录加入 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RDNet'))

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from data.dataset_sir import RealDataset, DataLoader
from engine import Engine
from options.net_options.train_options import TrainOptions


def enhance_clahe(img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """CLAHE 对比度增强，在 LAB 空间只对 L 通道操作"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


def enhance_gamma(img: np.ndarray, gamma: float = 0.6) -> np.ndarray:
    """Gamma 校正提亮，gamma < 1 提亮暗区"""
    return ((img.astype(np.float32) / 255.0) ** gamma * 255).astype(np.uint8)


def enhance_image(img: np.ndarray, method: str = 'both') -> dict:
    """
    对单张图片做增强，返回多种增强结果的 dict
    """
    results = {'original': img}
    if method in ('clahe', 'both'):
        results['clahe'] = enhance_clahe(img)
    if method in ('gamma', 'both'):
        results['gamma'] = enhance_gamma(img, gamma=0.6)
        results['gamma_0.4'] = enhance_gamma(img, gamma=0.4)
    return results


def run_rdnet_inference(input_dir, output_dir, model_path='pretrained/rdnet.pth', gpu_id='0'):
    """RDNet 反射去除，返回透射层保存目录"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 处理路径
    if not os.path.isabs(model_path):
        model_path = os.path.join(project_root, model_path)
    if not os.path.isabs(input_dir):
        input_dir = os.path.join(project_root, input_dir)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型权重未找到: {model_path}")

    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]

    try:
        opt = TrainOptions().parse()
        opt.isTrain = False
        opt.icnn_path = model_path
        opt.resume = True
        opt.gpu_ids = [int(gpu_id)] if gpu_id != '-1' else []
        opt.no_log = True
        opt.display_id = 0
        opt.verbose = False
        opt.if_align = True
        opt.nThreads = 1

        cudnn.benchmark = True

        dataset = RealDataset(input_dir)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                                num_workers=opt.nThreads, pin_memory=True)
        dummy = RealDataset(input_dir, size=1)
        engine = Engine(opt, dummy, dummy, dummy, dummy)

        print(f"[RDNet] 处理 {len(dataset)} 张图片...")
        print(f"  输入: {input_dir}")
        print(f"  输出: {output_dir}")

        for i, data in enumerate(dataloader):
            print(f"  [{i+1}/{len(dataset)}] {data['fn'][0]}")
            with torch.cuda.amp.autocast():
                engine.model.test(data, savedir=output_dir)
    finally:
        sys.argv = original_argv

    return output_dir


def run_pipeline(input_dir, output_dir='./output', enhance_method='both',
                 model_path='pretrained/rdnet.pth', gpu_id='0',
                 skip_rdnet=False):
    """
    完整 Pipeline：RDNet 去反射 → 低光增强
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: RDNet 反射去除
    if not skip_rdnet:
        rdnet_output = os.path.join(output_dir, 'rdnet_output')
        run_rdnet_inference(input_dir, rdnet_output, model_path, gpu_id)
    else:
        rdnet_output = input_dir

    # Step 2: 对每张透射层结果做增强
    print(f"\n[Enhance] 增强方法: {enhance_method}")
    enhance_dir = os.path.join(output_dir, 'enhanced')
    os.makedirs(enhance_dir, exist_ok=True)

    # 遍历 RDNet 输出的子目录
    for subdir in sorted(os.listdir(rdnet_output)):
        subpath = os.path.join(rdnet_output, subdir)
        if not os.path.isdir(subpath):
            continue

        # 查找透射层图片
        l_img = None
        for fname in os.listdir(subpath):
            if fname.endswith('_l.png'):
                l_img = os.path.join(subpath, fname)
                break

        if l_img is None:
            print(f"  [跳过] {subdir} — 找不到透射层图片")
            continue

        print(f"  [增强] {subdir}")
        img = cv2.imread(l_img)

        # 保存透射层原图到增强目录
        out_subdir = os.path.join(enhance_dir, subdir)
        os.makedirs(out_subdir, exist_ok=True)
        cv2.imwrite(os.path.join(out_subdir, 'T_raw.png'), img)

        # 应用增强
        enhanced = enhance_image(img, method=enhance_method)
        for key, img_e in enhanced.items():
            if key == 'original':
                continue
            save_path = os.path.join(out_subdir, f'T_{key}.png')
            cv2.imwrite(save_path, img_e)
            print(f"    保存: {save_path}")

    print(f"\n[完成] 结果保存在: {output_dir}")
    print(f"  RDNet 输出: {os.path.join(output_dir, 'rdnet_output')}")
    print(f"  增强结果:   {enhance_dir}")
    print(f"  - T_raw.png:   透射层原图")
    print(f"  - T_clahe.png:  CLAHE 增强")
    print(f"  - T_gamma.png:  Gamma 校正 (0.6)")
    print(f"  - T_gamma_0.4.png: Gamma 校正 (0.4, 更强提亮)")


def main():
    parser = argparse.ArgumentParser(description='TintedGlass Enhancement Pipeline')
    parser.add_argument('input_dir', help='输入图片目录')
    parser.add_argument('output_dir', nargs='?', default='./output', help='输出目录')
    parser.add_argument('--enhance', default='both', choices=['clahe', 'gamma', 'both', 'none'],
                        help='增强方法 (默认: both)')
    parser.add_argument('--model', default='pretrained/rdnet.pth', help='RDNet 权重路径')
    parser.add_argument('--gpu', default='0', help='GPU ID')
    parser.add_argument('--skip-rdnet', action='store_true',
                        help='跳过 RDNet，直接对已有透射图做增强')
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("TintedGlass Enhancement Pipeline")
    print("  RDNet (反射去除) → 低光增强 (CLAHE/Gamma)")
    print("=" * 60)

    run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        enhance_method=args.enhance,
        model_path=args.model,
        gpu_id=args.gpu,
        skip_rdnet=args.skip_rdnet,
    )


if __name__ == '__main__':
    main()
