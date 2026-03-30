import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from PIL import Image


# 自訂 Test 資料集

class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.image_files = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
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


# 主程式區塊

def main():
    # 設定參數
    BATCH_SIZE = 64 
    EPOCHS = 25
    LEARNING_RATE = 1e-4  
    NUM_CLASSES = 100
    DATA_DIR = './data' 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"目前使用的運算設備: {device}")

    # 資料增強與預處理
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

    # 載入資料集
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=val_test_transform)
    test_dataset = TestDataset(os.path.join(DATA_DIR, 'test'), transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print("載入 ResNet-101 預訓練模型...")
    model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 初始化 AMP 的 GradScaler (新版 API)
    scaler = torch.amp.GradScaler('cuda')

    # 學習率排程
    WARMUP_EPOCHS = 5      
    HOLD_EPOCHS = 10       
    
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / float(WARMUP_EPOCHS)
        elif epoch < (WARMUP_EPOCHS + HOLD_EPOCHS):
            return 1.0
        else:
            return 0.1

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

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
            
            # 使用 BF16 混合精度 (RTX 40 系列優化)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        
        # 驗證階段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] LR: {current_lr:.6f} | "
              f"Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")

        scheduler.step()

        # 儲存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model_resnet101.pth')
            print(f"  >>> 發現更高的驗證準確率 ({best_val_acc:.2f}%)，已儲存模型！")


    # 測試集預測

    print("\n訓練結束！載入最佳模型進行測試集預測...")
    model.load_state_dict(torch.load('best_model_resnet101.pth'))
    model.eval()
    predictions =[]

    # 取得 PyTorch index 對應真實資料夾名稱的列表
    # 例如 class_names[2] 會正確對應回 '10'
    class_names = train_dataset.classes 

    with torch.no_grad():
        for inputs, img_names in test_loader:
            inputs = inputs.to(device)
            
            # 預測時同樣使用 BF16 保持一致性與速度
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(inputs)
                
            _, predicted_indices = torch.max(outputs, 1)
            
            for img_name, pred_idx in zip(img_names, predicted_indices):
                # 1. 移除附檔名 (如 .jpg)
                file_id = os.path.splitext(img_name)[0]
                
                # 2. [關鍵修正] 將模型的預測 index 轉換回真實的資料夾名稱
                real_label = class_names[pred_idx.item()]
                
                predictions.append([file_id, real_label])


    # 儲存 CSV

    csv_filename = "prediction.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # [關鍵修正] 將標題改為題目要求的格式
        writer.writerow(['image_name', 'pred_label']) 
        
        writer.writerows(predictions)

    print(f"預測完成！最佳驗證準確率為: {best_val_acc:.2f}%，結果已儲存至 {csv_filename}。")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()