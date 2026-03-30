import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from PIL import Image

# ========================================== #
# 自訂 Test 資料集 (放在最外面)
# ========================================== #
class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.image_files =[f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.test_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_name

# ========================================== #
# 主程式區塊
# ========================================== #
def main():
    BATCH_SIZE = 128
    EPOCHS = 50
    LEARNING_RATE = 1e-4  # [修改] 微調預訓練模型時，建議把學習率調小一點 (1e-3 改為 1e-4)，避免破壞原本學好的特徵
    NUM_CLASSES = 100
    DATA_DIR = './data' # 確認這裡的路徑正確

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"目前使用的運算設備: {device}")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=val_test_transform)
    test_dataset = TestDataset(os.path.join(DATA_DIR, 'test'), transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # ========================================== #
    # [修改] 模型初始化：使用預訓練權重並修改最後一層
    # ========================================== #
    print("載入 ResNet-34 預訓練模型...")
    # 1. 載入帶有 ImageNet 預訓練權重的 ResNet-34 (DEFAULT 代表使用當前最好的預訓練權重版本)
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    
    # 2. 獲取原先全連接層的輸入特徵維度 (ResNet-34 最後一層輸入為 512 維)
    num_ftrs = model.fc.in_features
    
    # 3. 替換掉最後一層，改為輸出我們的 100 個類別
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    # 將模型搬移到 GPU
    model = model.to(device)
    # ========================================== #

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # [新增] 紀錄最佳驗證準確率的變數
    best_val_acc = 0.0

    print("開始訓練...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")

        # ========================================== #
        #[新增] 自動儲存表現最好的模型權重
        # ========================================== #
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model_resnet34.pth')
            print(f"  >>> 發現更高的驗證準確率 ({best_val_acc:.2f}%)，已儲存模型！")

    # ========================================== #
    # [修改] 測試前，先載回我們剛剛存的最佳模型
    # ========================================== #
    print("\n訓練結束！載入最佳模型進行測試集預測...")
    model.load_state_dict(torch.load('best_model_resnet34.pth'))
    model.eval()
    predictions =[]

    with torch.no_grad():
        for inputs, img_names in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted_classes = torch.max(outputs, 1)
            
            for img_name, pred_class in zip(img_names, predicted_classes):
                file_id = os.path.splitext(img_name)[0]
                predictions.append([file_id, pred_class.item()])

    csv_filename = "prediction.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_id', 'label']) 
        writer.writerows(predictions)

    print(f"預測完成！最佳驗證準確率為: {best_val_acc:.2f}%，結果已儲存至 {csv_filename}。")

# ==== 程式真正的執行進入點 ====
if __name__ == '__main__':
    # 加上 freeze_support() 可以避免某些 Windows 環境下的異常
    import multiprocessing
    multiprocessing.freeze_support()
    
    main()