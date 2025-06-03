import h5py
import pandas as pd
import numpy as np

def get_labels(path):
    file_path = path  # chỉnh đúng đường dẫn

    row_1_values_all = []  # mảng chứa tất cả giá trị hàng 1 các dataset

    with h5py.File(file_path, "r") as f:
        def process_dataset(name, obj):
            if isinstance(obj, h5py.Dataset):
                # Đọc toàn bộ dữ liệu dataset
                data = obj[()]
                df = pd.DataFrame(data)

                # Kiểm tra dataset có ít nhất 2 hàng không để lấy hàng 1
                if df.shape[0] > 1:
                    # Lấy giá trị hàng 1 (index=1) dạng numpy array
                    row_1 = df.iloc[1].values

                    # Thêm từng phần tử trong hàng 1 vào mảng chung
                    row_1_values_all.extend(row_1.tolist())
                else:
                    print(f"⚠ Dataset {name} không có đủ 2 hàng để lấy hàng 1.")

        f.visititems(process_dataset)

    # Chuyển list thành numpy array (tùy chọn)
    labels = np.array(row_1_values_all)

    print("Tổng giá trị hàng 1 của tất cả dataset gộp lại:")
    print(labels.shape)
    return labels

def get_data(file_path):
    dataset_arrays = []  # List chứa từng dataset dạng numpy array (30, 48)

    with h5py.File(file_path, "r") as f:
        def process_dataset(name, obj):
            if isinstance(obj, h5py.Dataset):
                data = obj[()]
                # Kiểm tra kích thước để đảm bảo (30,48)
                if data.shape == (30, 48):
                    dataset_arrays.append(data)
                else:
                    print(f"⚠ Dataset '{name}' có kích thước {data.shape}, bỏ qua.")

        f.visititems(process_dataset)

    # Chuyển list thành numpy array (shape: [số dataset, 30, 48])
    Gait = np.array(dataset_arrays)

    print(f"Số dataset đã lấy: {Gait.shape[0]}")
    print(f"Kích thước mảng cuối cùng: {Gait.shape}")
    return Gait