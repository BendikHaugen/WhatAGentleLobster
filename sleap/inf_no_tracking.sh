#!/bin/sh
# 
# $1 - batch size
# 
# Unet TD 
sleap-track "video/channel1.mp4" -m models_unet_td/230209_221559.centered_instance -m models_unet_td/230209_221559.centroid --frames 0-200 --batch_size $1 -o unet_td_channel1_no_track.predictions.slp   
sleap-track "video/channel2.mp4" -m models_unet_td/230209_221559.centered_instance -m models_unet_td/230209_221559.centroid --frames 0-200 --batch_size $1 -o unet_td_channel2_no_track.predictions.slp  
sleap-track "video/channel3.mp4" -m models_unet_td/230209_221559.centered_instance -m models_unet_td/230209_221559.centroid --frames 0-200 --batch_size $1 -o unet_td_channel3_no_track.predictions.slp  
sleap-track "video/channel4.mp4" -m models_unet_td/230209_221559.centered_instance -m models_unet_td/230209_221559.centroid --frames 0-200 --batch_size $1 -o unet_td_channel4_no_track.predictions.slp

# Unet BU
sleap-track "video/channel1.mp4" -m models_unet_bu/230213_190838.multi_instance --frames 0-200 --batch_size $1 -o unet_bu_channel1_no_track.predictions.slp  
sleap-track "video/channel2.mp4" -m models_unet_bu/230213_190838.multi_instance --frames 0-200 --batch_size $1 -o unet_bu_channel2_no_track.predictions.slp 
sleap-track "video/channel3.mp4" -m models_unet_bu/230213_190838.multi_instance --frames 0-200 --batch_size $1 -o unet_bu_channel3_no_track.predictions.slp 
sleap-track "video/channel4.mp4" -m models_unet_bu/230213_190838.multi_instance --frames 0-200 --batch_size $1 -o unet_bu_channel4_no_track.predictions.slp

# ResNet TD
sleap-track "video/channel1.mp4" -m models_resnet_td/230210_121938.centroid -m models_resnet_td/230210_121938.centered_instance --frames 0-200 --batch_size $1 -o resnet_td_channel1_no_track.predictions.slp
sleap-track "video/channel2.mp4" -m models_resnet_td/230210_121938.centroid -m models_resnet_td/230210_121938.centered_instance --frames 0-200 --batch_size $1 -o resnet_td_channel2_no_track.predictions.slp
sleap-track "video/channel3.mp4" -m models_resnet_td/230210_121938.centroid -m models_resnet_td/230210_121938.centered_instance --frames 0-200 --batch_size $1 -o resnet_td_channel3_no_track.predictions.slp
sleap-track "video/channel4.mp4" -m models_resnet_td/230210_121938.centroid -m models_resnet_td/230210_121938.centered_instance --frames 0-200 --batch_size $1 -o resnet_td_channel4_no_track.predictions.slp

# ResNet BU 
sleap-track "video/channel1.mp4" -m models_resnet_bu/230214_074630.multi_instance --frames 0-200 --batch_size $1 -o resnet_bu_channel1_no_track.predictions.slp
sleap-track "video/channel2.mp4" -m models_resnet_bu/230214_074630.multi_instance --frames 0-200 --batch_size $1 -o resnet_bu_channel2_no_track.predictions.slp
sleap-track "video/channel3.mp4" -m models_resnet_bu/230214_074630.multi_instance --frames 0-200 --batch_size $1 -o resnet_bu_channel3_no_track.predictions.slp
sleap-track "video/channel4.mp4" -m models_resnet_bu/230214_074630.multi_instance --frames 0-200 --batch_size $1 -o resnet_bu_channel4_no_track.predictions.slp

# LEAP TD
sleap-track "video/channel1.mp4" -m models_leap_td/230209_232159.centroid -m models_leap_td/LEAP_half.centered_instance --frames 0-200 --batch_size $1 -o leap_td_channel1_no_track.predictions.slp
sleap-track "video/channel2.mp4" -m models_leap_td/230209_232159.centroid -m models_leap_td/LEAP_half.centered_instance --frames 0-200 --batch_size $1 -o leap_td_channel2_no_track.predictions.slp
sleap-track "video/channel3.mp4" -m models_leap_td/230209_232159.centroid -m models_leap_td/LEAP_half.centered_instance --frames 0-200 --batch_size $1 -o leap_td_channel3_no_track.predictions.slp
sleap-track "video/channel4.mp4" -m models_leap_td/230209_232159.centroid -m models_leap_td/LEAP_half.centered_instance --frames 0-200 --batch_size $1 -o leap_td_channel4_no_track.predictions.slp

# LEAP BU
sleap-track "video/channel1.mp4" -m models_leap_bu/230212_181606.multi_instance --frames 0-200 --batch_size $1 -o leap_bu_channel1_no_track.predictions.slp 
sleap-track "video/channel2.mp4" -m models_leap_bu/230212_181606.multi_instance --frames 0-200 --batch_size $1 -o leap_bu_channel2_no_track.predictions.slp
sleap-track "video/channel3.mp4" -m models_leap_bu/230212_181606.multi_instance --frames 0-200 --batch_size $1 -o leap_bu_channel3_no_track.predictions.slp
sleap-track "video/channel4.mp4" -m models_leap_bu/230212_181606.multi_instance --frames 0-200 --batch_size $1 -o leap_bu_channel4_no_track.predictions.slp
