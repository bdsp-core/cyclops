# Automated detection of interictal epileptiform discharges with few EEG channels

This repository contains materials related to our publication, *"Automated detection of interictal epileptiform discharges with few EEG channels"*. It includes the code used to generate the findings of our paper. The dataset can be found on [dropbox](https://www.dropbox.com/scl/fo/7m9gu06yahaepzsu9xryp/AFwqrAGYfEL5GkYWk0YvFp4?rlkey=0tmwsmojrmpkpwn7qpeykn5on&st=a23mlzup&dl=0).

## Citation  
If you find this work useful, please cite it as:  
TO DO

## Abstract
Interictal epileptiform discharges (IEDs) are crucial for epilepsy diagnosis and management. New EEG devices with fewer electrodes are more accessible but their ability to detect IEDs is uncertain. The aim of this study is to develop and validate a machine learning model capable of detecting IEDs in reduced-channel EEG data, enabling broader epilepsy diagnosis. <br>
Using EEG samples from 3,378 patients and an external validation set of 51 patients, we trained Cyclops, a deep neural network designed to function across various channel configurations. <br>
Performance was evaluated using AUROC and other clinically relevant metrics, including IED source location sensitivity. Cyclops demonstrated strong performance even with minimal channels. AUROC for one channel: 0.876 [95% CI: 0.854-0.897]; best configuration based on a clinically available product: 0.950 [95% CI: 0.936-0.962]; for the detection of focal IEDs with two local channels, AUROC values ranged from 0.701 [95% CI: 0.656-0.745] to 0.930 [95% CI: 0.902-0.955] with a median AUROC of 0.809. On the external validation set, performance ranged from 0.692 [95% CI: 0.593-0.782] to 0.949 [95% CI: 0.922-0.972] with a median AUROC of 0.846. Thus, Cyclops supports effective IED detection with reduced EEG setups, enhancing accessibility and expanding epilepsy diagnosis to broader patient populations.

