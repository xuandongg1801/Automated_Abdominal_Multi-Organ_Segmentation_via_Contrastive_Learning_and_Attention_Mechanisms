# Automated Abdominal Multi-Organ Segmentation via Contrastive Learning and Attention Mechanisms

Đồ án cuối kỳ Deep Learning: phân vùng đa cơ quan vùng bụng trên ảnh CT bằng ResNet-UNet, TransUNet và supervised pixel-level contrastive learning.

## 1. Tổng quan dự án

Dự án tập trung vào bài toán **Medical Image Segmentation** trên bộ dữ liệu **Synapse Multi-organ CT**. Repo hiện tại triển khai hai nhóm mô hình:

- **Model 1 - CNN-based:** ResNet50-UNet.
- **Model 2 - Attention-based:** TransUNet, kết hợp ResNet-50 encoder với ViT-B/16 Transformer blocks.

Hướng tiếp cận chính:

- Tiền xử lý CT slice: áp dụng HU window `[-125, 275]`, chuẩn hóa giá trị ảnh, resize về `224 x 224`, chuyển ảnh 1 kênh thành 3 kênh để phù hợp với backbone pretrained.
- Data augmentation trong training loader: intensity scaling, intensity shifting và Gaussian noise nhẹ.
- Loss: Dice + Cross Entropy, có tùy chọn thêm **supervised pixel-level contrastive loss** trên feature map.
- Training: transfer learning từ ResNet-50 ImageNet weights và ViT pretrained weights cho TransUNet.
- Evaluation: Dice, IoU, pixel accuracy trong training/evaluation helper; notebook đánh giá bổ sung F1, ROC-AUC, Average Precision, PR-AUC, HD95 và ASD.
- Demo: ứng dụng Streamlit cho phép upload CT slice hoặc `.npy/.npz`, chọn model checkpoint, chạy inference và hiển thị mask/overlay.

## 2. Thành viên

| STT | Họ và Tên | Lớp | MSSV | GitHub |
| :--: | :-- | :--: | :--: | :-- |
| 1 | Trần Viết Gia Huy | CS0001 | 31231027056 | [@Tommyhuy1705](https://github.com/Tommyhuy1705) |
| 2 | Nguyễn Minh Nhựt | CS0001 | 31231022656 | [@Sura3607](https://github.com/Sura3607) |
| 3 | Nguyễn Trọng Hưởng | CS0001 | 31231023691 | [@trongjhuongwr](https://github.com/trongjhuongwr) |
| 4 | Tô Xuân Đông | CS0001 | 31231025345 | [@xuandongg1801](https://github.com/xuandongg1801) |

## 3. Cấu trúc repo

```text
.
|-- app/
|   |-- streamlit_app.py      # Streamlit web demo
|   |-- inference.py          # Load checkpoint, preprocess, predict, visualize
|   `-- model_registry.py     # Kaggle Model handles và metadata model
|-- notebooks/
|   |-- 01_train_resnet_unet_kaggle.ipynb
|   |-- 02_train_transunet_kaggle.ipynb
|   `-- 03_evaluate_models_kaggle.ipynb
|-- src/
|   |-- utils.py              # Dataset loader, metrics, train/eval loop, contrastive loss
|   `-- models/
|       |-- resnet_unet.py
|       |-- transunet.py
|       |-- losses.py
|       `-- layers.py
|-- requirements.txt
`-- README.md
```

## 4. Dataset

Dataset sử dụng trong notebook là **Synapse Multi-organ CT** với 9 lớp:

- `0`: Background
- `1`: Spleen
- `2`: Right kidney
- `3`: Left kidney
- `4`: Gallbladder
- `5`: Liver
- `6`: Stomach
- `7`: Aorta
- `8`: Pancreas

Training notebooks tự động tìm layout dữ liệu Synapse quen thuộc trên Kaggle, đặc biệt:

- `train_npz/`
- `test_vol_h5/`

Trong các lần chạy đã lưu output notebook, data split gồm:

- Train: 1866 slices
- Validation: 345 slices
- Test: 1568 slices

## 5. Mô hình

### ResNet-UNet

`src/models/resnet_unet.py` triển khai ResNet-50 encoder và U-Net decoder cho semantic segmentation. Encoder có thể dùng ImageNet pretrained weights thông qua `torchvision`.

### TransUNet

`src/models/transunet.py` triển khai kiến trúc hybrid:

- ResNet-50 encoder để trích xuất feature map cục bộ.
- Patch embedding từ CNN feature map thành token sequence.
- ViT-B/16 Transformer blocks từ `timm`.
- U-Net style decoder để sinh segmentation logits.

## 6. Training

Training được thực hiện trong các notebook Kaggle:

- `notebooks/01_train_resnet_unet_kaggle.ipynb`
- `notebooks/02_train_transunet_kaggle.ipynb`

Mỗi notebook train nhiều biến thể bằng cách sweep `contrastive_weight`:

- `0.00`: baseline, chỉ dùng Dice + Cross Entropy.
- `0.01`: Dice + Cross Entropy + contrastive loss.
- `0.03`: Dice + Cross Entropy + contrastive loss.
- `0.05`: Dice + Cross Entropy + contrastive loss.

Cấu hình chính trong notebook:

- Optimizer: `AdamW`
- ResNet-UNet batch size mặc định: `12`
- TransUNet batch size mặc định: `4`
- Early stopping patience: `20`
- Contrastive temperature: `0.1`
- Contrastive max samples: `2048`

Checkpoint tốt nhất được lưu theo validation Dice và có thể upload lên Kaggle Model bằng `kagglehub`.

## 7. Kết quả đánh giá

Notebook `notebooks/03_evaluate_models_kaggle.ipynb` load các checkpoint từ Kaggle Model và đánh giá trên Synapse test set.

Bảng tổng hợp từ output notebook:

| Model | Contrastive weight | Mean Dice | Mean IoU | Pixel Acc | HD95 |
| :-- | --: | --: | --: | --: | --: |
| ResNet-UNet cw0 | 0.00 | 0.8209 | 0.7123 | 0.9925 | 7.6537 |
| ResNet-UNet cw001 | 0.01 | 0.8125 | 0.7045 | 0.9927 | 7.0932 |
| ResNet-UNet cw003 | 0.03 | 0.8012 | 0.6890 | 0.9919 | 7.9375 |
| ResNet-UNet cw005 | 0.05 | 0.8053 | 0.6942 | 0.9925 | 7.1660 |
| TransUNet cw0 | 0.00 | 0.8197 | 0.7132 | 0.9926 | 8.0682 |
| TransUNet cw001 | 0.01 | **0.8248** | **0.7210** | **0.9928** | 7.7150 |
| TransUNet cw003 | 0.03 | 0.8168 | 0.7085 | 0.9926 | 7.2277 |
| TransUNet cw005 | 0.05 | 0.8108 | 0.7039 | 0.9926 | 8.0434 |

Theo kết quả này, biến thể tốt nhất theo Mean Dice/IoU là **TransUNet cw001**.

## 8. Streamlit web demo

Ứng dụng demo nằm tại:

```text
app/streamlit_app.py
```

Tính năng chính:

- Upload CT slice dạng `.png`, `.jpg`, `.tif`, `.npy` hoặc `.npz`.
- Chọn architecture: ResNet-UNet hoặc TransUNet.
- Chọn training variant theo `contrastive_weight`.
- Chạy inference bằng checkpoint từ Kaggle Model hoặc checkpoint local.
- Hiển thị input, ground truth nếu có, predicted mask, overlay và legend 9 lớp.
- Tính Mean Dice và Mean IoU trên slice upload nếu file có label/mask.
- Download predicted mask, colored mask và overlay.

Chạy local:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Nếu muốn dùng checkpoint local thay vì tải từ Kaggle Model, set biến môi trường tương ứng, ví dụ:

```bash
TRANSUNET_CW001_MODEL_DIR=/path/to/checkpoint_folder
RESNET_UNET_MODEL_DIR=/path/to/checkpoint_folder
```

## 9. Kaggle Model variants

`app/model_registry.py` đang đăng ký các biến thể:

- `resnet-unet`
- `resnet-unet-cw001`
- `resnet-unet-cw003`
- `resnet-unet-cw005`
- `transunet`
- `transunet-cw001`
- `transunet-cw003`
- `transunet-cw005`

Mặc định app dùng **TransUNet contrastive 0.01** vì đây là biến thể có Mean Dice/IoU tốt nhất trong notebook evaluation.

## 10. Ghi chú về phạm vi hiện tại

Repo hiện tại không có module riêng cho Albumentations, ElasticTransform, GridDistortion hay Cosine LR scheduler. README này mô tả theo đúng triển khai hiện có trong code và notebook:

- Augmentation hiện có là intensity augmentation trong `src/utils.py`.
- Training loop hiện tại dùng `AdamW`, chưa cấu hình scheduler riêng.
- Web demo hiện tại là Streamlit, không phải Gradio.
