import matplotlib.pyplot as plt
import re
def pare_file(path):
    train_info={'loss':[],'acc':[],'cls_acc':[],'miu':[],'fwiou':[]}
    val_info={'loss':[],'acc':[],'cls_acc':[],'miu':[],'fwiou':[]}
    with open(path) as f:
        text=f.read()
    train_p = 'Train Loss:(\d+\.\d+),Train Acc:(\d+\.\d+),Train cls acc:(\d+\.\d+),Train Mean IU:(\d+\.\d+),Train fwavacc:(\d+\.\d+)'
    val_p = 'Val Loss:(\d+\.\d+),Val Acc:(\d+\.\d+),Val cls acc:(\d+\.\d+),Val mean iu:(\d+\.\d+),Val fwavacc:(\d+\.\d+)'
    train_result = re.findall(train_p, text)
    val_result = re.findall(val_p, text)

    for item in train_result:
        train_info['loss'].append(float(item[0]))
        train_info['acc'].append(float(item[1]))
        train_info['cls_acc'].append(float(item[2]))
        train_info['miu'].append(float(item[3]))
        train_info['fwiou'].append(float(item[4]))

    for item in val_result:
        val_info['loss'].append(float(item[0]))
        val_info['acc'].append(float(item[1]))
        val_info['cls_acc'].append(float(item[2]))
        val_info['miu'].append(float(item[3]))
        val_info['fwiou'].append(float(item[4]))

    return train_info,val_info


if __name__ == '__main__':
    file_path1='/raid/lcq/workspace/crack/FCN-VGG/log/log_crack_1556168187.txt'
    file_path2='/raid/lcq/workspace/crack/resnet_Atrous/log/log_crack_1556168373.txt'

    file1_train_info,_=pare_file(file_path1)
    file2_train_info, _ = pare_file(file_path2)


    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax.plot(file1_train_info['loss'],color="blue",linewidth=1, linestyle="-",label="FCN32")
    ax.plot(file2_train_info['loss'],color="red",linewidth=1, linestyle="-",label="Ours")
    ax.grid()
    #ax.set_title('(a) ')
    ax.set_xlabel('epoch\n(a)')
    ax.set_ylabel('Train Loss')
    ax.legend()


    ax = fig.add_subplot(222)
    ax.plot(file1_train_info['acc'], color="blue", linewidth=1, linestyle="-", label="FCN32")
    ax.plot(file2_train_info['acc'], color="red", linewidth=1, linestyle="-", label="Ours")
    ax.grid()
    ax.set_xlabel('epoch\n(b)')
    ax.set_ylabel('Train PA')
    ax.legend()


    ax = fig.add_subplot(223)
    ax.plot(file1_train_info['miu'], color="blue", linewidth=1, linestyle="-", label="FCN32")
    ax.plot(file2_train_info['miu'], color="red", linewidth=1, linestyle="-", label="Ours")
    ax.grid()
    ax.set_xlabel('epoch\n(c)')
    ax.set_ylabel('Train MIoU')
    ax.legend()

    ax = fig.add_subplot(224)
    ax.plot(file1_train_info['fwiou'], color="blue", linewidth=1, linestyle="-", label="FCN32")
    ax.plot(file2_train_info['fwiou'], color="red", linewidth=1, linestyle="-", label="Ours")
    ax.grid()
    ax.set_xlabel('epoch\n(d)')
    ax.set_ylabel('Train FWIoU')
    ax.legend()
    plt.tight_layout()
    # plt.savefig('./result.png', dpi=300)
    plt.show()