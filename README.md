



Code borrow form https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark





## Requirements

1. Install `pytorch` and `torchvision` (we use `pytorch==1.12.0` and `torchvision==0.13.0`).
2. `pip install -r requirements.txt`

## Run
### Download Processed Data
Please download and organize the [processed datasets](https://drive.google.com/drive/folders/1R0R8SlVbLI1iUFQCzh_mH90H_4CW2iwt?usp=sharing) in this structure:
```
Benchmark
├── Data
    ├── NTU-Fi_HAR
    │   ├── test_amp
    │   ├── train_amp
    ├── UT_HAR
    │   ├── data
    │   ├── label
    
```

*Example: `python train_baselin_model.py --model panet --dataset UT_HAR_data`*






## Dataset
#### UT-HAR
[*A Survey on Behavior Recognition Using WiFi Channel State Information*](https://ieeexplore.ieee.org/document/8067693) [[Github]](https://github.com/ermongroup/Wifi_Activity_Recognition)  
- **CSI size** : 1 x 250 x 90
- **number of classes** : 7
- **classes** : lie down, fall, walk, pickup, run, sit down, stand up
- **train number** : 3977
- **test number** : 996  

#### NTU-HAR
[*Efficientfi: Towards Large-Scale Lightweight Wifi Sensing via CSI Compression*](https://ieeexplore.ieee.org/document/9667414)  
- **CSI size** : 3 x 114 x 500
- **number of classes** : 6
- **classes** : box, circle, clean, fall, run, walk
- **train number** : 936
- **test number** : 264  



#### Notice
Please download and unzip all the datasets with Linux system in order to avoid decoding errors.

## Datasets Reference
```
@article{yousefi2017survey,
  title={A survey on behavior recognition using WiFi channel state information},
  author={Yousefi, Siamak and Narui, Hirokazu and Dayal, Sankalp and Ermon, Stefano and Valaee, Shahrokh},
  journal={IEEE Communications Magazine},
  volume={55},
  number={10},
  pages={98--104},
  year={2017},
  publisher={IEEE}
}

@article{yang2022autofi,
  title={AutoFi: Towards Automatic WiFi Human Sensing via Geometric Self-Supervised Learning},
  author={Yang, Jianfei and Chen, Xinyan and Zou, Han and Wang, Dazhuo and Xie, Lihua},
  journal={arXiv preprint arXiv:2205.01629},
  year={2022}
}

@article{yang2022efficientfi,
  title={Efficientfi: Towards large-scale lightweight wifi sensing via csi compression},
  author={Yang, Jianfei and Chen, Xinyan and Zou, Han and Wang, Dazhuo and Xu, Qianwen and Xie, Lihua},
  journal={IEEE Internet of Things Journal},
  year={2022},
  publisher={IEEE}
}

@article{wang2022caution,
  title={CAUTION: A Robust WiFi-based Human Authentication System via Few-shot Open-set Gait Recognition},
  author={Wang, Dazhuo and Yang, Jianfei and Cui, Wei and Xie, Lihua and Sun, Sumei},
  journal={IEEE Internet of Things Journal},
  year={2022},
  publisher={IEEE}
}

@article{zhang2021widar3,
  title={Widar3. 0: Zero-effort cross-domain gesture recognition with wi-fi},
  author={Zhang, Yi and Zheng, Yue and Qian, Kun and Zhang, Guidong and Liu, Yunhao and Wu, Chenshu and Yang, Zheng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021},
  publisher={IEEE}
}  
```
