import sleap
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.style.use("seaborn-deep")
sleap.versions()

unet_bu = sleap.load_metrics("unet_bu_models/230213_190838.multi_instance", split="val")
unet_bu = {"name": "U-Net Bottom Up", "model": unet_bu}
unet_centered = sleap.load_metrics("unet_td_models/230209_221559.centered_instance", split="val")
unet_centered = {"name": "U-Net Centered Instance Top Down", "model": unet_centered}
unet_centroid = sleap.load_metrics("unet_td_models/230209_221559.centroid", split="val")
unet_centroid = {"name": "U-Net Centroid Top Down", "model": unet_centroid}
leap_centered = sleap.load_metrics("leap_td_models/LEAP_half.centered_instance", split="val")
leap_centered = {"name": "Leap Centered Instance Top Down", "model": leap_centered}
leap_centroid = sleap.load_metrics("leap_td_models/230209_232159.centroid", split="val")
leap_centroid = {"name": "Leap Centroid Top Down", "model": leap_centroid}
leap_multi = sleap.load_metrics("leap_bu_models/230212_181606.multi_instance", split="val")
leap_multi = {"name": "Leap Bottom Up", "model": leap_multi}
resnet_centered = sleap.load_metrics("resnet_td_models/230210_121938.centered_instance", split="val")
resnet_centered = {"name": "ResNet Centered Instance Top Down", "model": resnet_centered}
resnet_centroid = sleap.load_metrics("resnet_td_models/230210_121938.centroid", split="val")
resnet_centroid = {"name": "ResNet Centroid Top Down", "model": resnet_centroid}


models = [
    resnet_centroid, 
    resnet_centered, 
    leap_multi, 
    leap_centroid, 
    leap_centered, 
    unet_centroid, 
    unet_centered, 
    unet_bu, 
    ]
for model in models:
    print(f'Evaluation metrics for {model["name"]}') 
    print("Error distance (50%):", model["model"]["dist.p50"])
    print("Error distance (90%):", model["model"]["dist.p90"])
    print("Error distance (95%):", model["model"]["dist.p95"])
    print("mAR:", model["model"]["oks_voc.mAR"])
    print("mAP:", model["model"]["oks_voc.mAP"])
    print("mPCK:",model["model"]["pck.mPCK"])
    plt.figure(figsize=(6, 3), dpi=150, facecolor="w")
    sns.histplot(model["model"]["dist.dists"].flatten(), binrange=(0, 20), kde=True, kde_kws={"clip": (0, 20)}, stat="probability")
    plt.xlabel("Localization error (px)")
    plt.title(model["name"])
    plt.savefig(f'{model["name"]}_localization_error.png', bbox_inches='tight')
    plt.clf()
    plt.figure(figsize=(6, 3), dpi=150, facecolor="w")
    sns.histplot(model["model"]["oks_voc.match_scores"].flatten(), binrange=(0, 1), kde=True, kde_kws={"clip": (0, 1)}, stat="probability")
    plt.xlabel("Object Keypoint Similarity")
    plt.ylim(0, 0.14)
    plt.title(model["name"])
    plt.savefig(f'{model["name"]}_OKS.png', bbox_inches='tight')
    plt.clf()
    plt.figure(figsize=(4, 4), dpi=150, facecolor="w")
    for precision, thresh in zip(model["model"]["oks_voc.precisions"][::2], model["model"]["oks_voc.match_score_thresholds"][::2]):
        plt.plot(model["model"]["oks_voc.recall_thresholds"], precision, "-", label=f"OKS @ {thresh:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.title(model["name"])
    plt.savefig(f'{model["name"]}_precision_recall.png', bbox_inches='tight')
    plt.clf()