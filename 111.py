# 测试
import torch
import matplotlib.pyplot as plt

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
print('Accuracy of the network on the 400 test images: %d %%' % (100 * correct / total))

#训练集acc/loss
acc = history.history['accuracy']
loss = history.history['loss']
#测试集acc/loss
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

#acc曲线
plt.subplot(1,2,1)
plt.plot(acc, label='Training Acc')
plt.plot(val_acc, label='Validation Acc')
plt.title('Training and Validation ACC')
plt.legend()

#loss曲线
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

