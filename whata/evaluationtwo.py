import sleap
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.style.use("seaborn-deep")
sleap.versions()

unet_bu = sleap.load_metrics("unet_bu_models/230213_190838.multi_instance", split="val")
unet_bu = {"name": "U-Net BU", "model": unet_bu}
unet_centered = sleap.load_metrics("unet_td_models/230209_221559.centered_instance", split="val")
unet_centered = {"name": "U-Net TD", "model": unet_centered}
unet_td_reduced = sleap.load_metrics("unet_td_models/230205_165841.centered_instance", split="val")
unet_td_reduced = {"name": "U-Net TD reduced filters", "model": unet_td_reduced}
leap_centered = sleap.load_metrics("leap_td_models/230209_232159.centered_instance", split="val")
leap_centered = {"name": "Leap TD", "model": leap_centered}
leap_centered_quarter = sleap.load_metrics("leap_td_models\Quarter_filters\Centered_instance", split="val")
leap_centered_quarter = {"name": "Leap TD reduced filters", "model": leap_centered_quarter}
leap_multi = sleap.load_metrics("leap_bu_models/230212_181606.multi_instance", split="val")
leap_multi = {"name": "Leap BU", "model": leap_multi}
resnet_centered = sleap.load_metrics("resnet_td_models/230210_121938.centered_instance", split="val")
resnet_centered = {"name": "ResNet TD", "model": resnet_centered}
resnet_multi = sleap.load_metrics("restnet_bu\Initial\multi_instance", split="val")
resnet_multi = {"name": "ResNet BU", "model": resnet_multi}



models = [
    resnet_centered, 
    leap_multi, 
    leap_centered, 
    unet_centered, 
    unet_bu, 
    resnet_multi,
    leap_centered_quarter,
    unet_td_reduced
]

# Create a summary dataframe to hold all the model metrics
# Create summary dataframes to hold the model metrics
summary_TD = pd.DataFrame(columns=["Model", "Error distance (50%)", "Error distance (90%)", "Error distance (95%)", "mAR", "mAP", "mPCK"])
summary_BU = pd.DataFrame(columns=["Model", "Error distance (50%)", "Error distance (90%)", "Error distance (95%)", "mAR", "mAP", "mPCK"])
summary_reduced = pd.DataFrame(columns=["Model", "Error distance (50%)", "Error distance (90%)", "Error distance (95%)", "mAR", "mAP", "mPCK"])
# Separate models into two lists
models_TD = [model for model in models if "TD" in model["name"] and "reduced filters" not in model["name"] ]
models_BU = [model for model in models if "BU" in model["name"] and "reduced filters" not in model["name"] ]
models_reduced = [model for model in models if "reduced filters" in model["name"] ]

# Define a function to generate plots and update summary dataframe
def process_models(models, summary_df, plot_title):
    plt.figure(figsize=(12, 6))
    plt.title(f"{plot_title} - Localization error and OKS")
    plt.figure(figsize=(12, 6))
    plt.title(f"{plot_title} - Precision-Recall curves")

    for model in models:
        summary_df = summary_df.append({
            "Model": model["name"],
            "Error distance (50%)": model["model"]["dist.p50"],
            "Error distance (90%)": model["model"]["dist.p90"],
            "Error distance (95%)": model["model"]["dist.p95"],
            "mAR": model["model"]["oks_voc.mAR"],
            "mAP": model["model"]["oks_voc.mAP"],
            "mPCK": model["model"]["pck.mPCK"]
        }, ignore_index=True)

        plt.figure(1)
        sns.histplot(model["model"]["dist.dists"].flatten(), binrange=(0, 20), kde=True, kde_kws={"clip": (0, 20)}, stat="probability", label=model["name"])

        plt.figure(2)
        for precision, thresh in zip(model["model"]["oks_voc.precisions"][::2], model["model"]["oks_voc.match_score_thresholds"][::2]):
            plt.plot(model["model"]["oks_voc.recall_thresholds"], precision, "-", label=f"{model['name']} - OKS @ {thresh:.2f}")

    plt.figure(1)
    plt.xlabel("Localization error (px)")
    plt.xlim([0, 7.5])
    plt.ylim([0, 0.14])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5) 
    plt.savefig(f'{plot_title}_localization_error.png', bbox_inches='tight')  

    plt.figure(2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5) 
    plt.savefig(f'{plot_title}_precision_recall.png', bbox_inches='tight') 

    plt.show()


    return summary_df

# Generate plots and summary dataframes for each group of models
summary_TD = process_models(models_TD, summary_TD, "Top Down Models")
summary_BU = process_models(models_BU, summary_BU, "Bottom Up Models")
summary_reduced = process_models(models_reduced, summary_reduced, "Models with reduced filters")
# Print the summary dataframes
print("Top Down Models:\n", summary_TD)
print("Bottom Up Models:\n", summary_BU)
print("Models with reduced filters: \n", summary_reduced)

print("Top Down Models:\n", summary_TD.to_latex(index=False))
print("Bottom Up Modells:\n", summary_BU.to_latex(index=False))
print("Reduced Filters Models:\n", summary_reduced.to_latex(index=False))