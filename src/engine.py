import os
import time
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, corrects = 0.0, 0
    total_samples = len(dataloader.dataset)
    total_batches = len(dataloader)

    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        corrects += (outputs.argmax(1) == labels).sum().item()

        if (i + 1) % 200 == 0 or (i + 1) == total_batches:
            print(f"Batch [{i+1}/{total_batches}] | Anlık Loss: {loss.item():.4f}")

    return running_loss / total_samples, corrects / total_samples

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss, corrects = 0.0, 0
    total_samples = len(dataloader.dataset)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            corrects += (outputs.argmax(1) == labels).sum().item()

    return running_loss / total_samples, corrects / total_samples

def train_model(model, train_loader, val_loader, criterion, optimizer, device, save_path, scheduler=None, epochs=5):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"Eğitim Başlatılıyor ({device})...")
    start_time = time.time()

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1} -> Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f">> En iyi model kaydedildi: Acc = {best_acc:.4f}")

    elapsed = time.time() - start_time
    print(f"\nTamamlandı. Süre: {elapsed // 60:.0f}dk {elapsed % 60:.0f}sn | En İyi Val Acc: {best_acc:.4f}")
    return history

def evaluate_and_plot(model, val_loader, class_names, device, weight_path=None, cmap='Blues'):
    if weight_path:
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n--- Sınıflandırma Raporu ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=class_names, yticklabels=class_names)
    plt.title('Karışıklık Matrisi')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek Sınıf')
    plt.tight_layout()
    plt.show()