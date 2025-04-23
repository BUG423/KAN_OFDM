import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from kan import *
def conv3x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """ 1D convolution with kernel size 3 """
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes, out_planes, stride=1):
    """ 1D convolution with kernel size 1 """
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# Define constants
K = 64
CP = K // 4
P = 64  # number of pilot carriers per OFDM block
allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])
pilotCarriers = allCarriers[::K // P]  # Pilots is every (K/P)th carrier.
dataCarriers = np.delete(allCarriers, pilotCarriers)
mu = 2
payloadBits_per_OFDM = len(dataCarriers) * mu
payloadBits_per_OFDM = K * mu
SNRdb = 20  # signal to noise-ratio in dB at the receiver

# Mapping table
mapping_table = {
    (0, 0): -1 - 1j,
    (0, 1): -1 + 1j,
    (1, 0): 1 - 1j,
    (1, 1): 1 + 1j,
}
demapping_table = {v: k for k, v in mapping_table.items()}

# Functions
def Modulation(bits):
    bit_r = bits.reshape((int(len(bits) / mu), mu))
    return (2 * bit_r[:, 0] - 1) + 1j * (2 * bit_r[:, 1] - 1)  # This is just for QAM modulation

def OFDM_symbol(Data, pilot_flag):
    symbol = np.zeros(K, dtype=complex)  # the overall K subcarriers
    symbol[pilotCarriers] = pilotValue  # allocate the pilot subcarriers
    symbol[dataCarriers] = Data  # allocate the pilot subcarriers
    return symbol

def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)

def addCP(OFDM_time):
    cp = OFDM_time[-CP:]  # take the last CP samples ...
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning

def channel(signal, channelResponse, SNRdb):
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdb / 10)
    noise = np.sqrt(sigma2 / 2) * (np.random.randn(*convolved.shape) + 1j * np.random.randn(*convolved.shape))
    return convolved + noise

def removeCP(signal):
    return signal[CP:(CP + K)]

def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)

def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest

def get_payload(equalized):
    return equalized[dataCarriers]

def PS(bits):
    return bits.reshape((-1,))

def ofdm_simulate(codeword, channelResponse, SNRdb):
    OFDM_data = np.zeros(K, dtype=complex)
    OFDM_data[allCarriers] = pilotValue
    OFDM_time = IDFT(OFDM_data)
    OFDM_withCP = addCP(OFDM_time)
    OFDM_TX = OFDM_withCP
    OFDM_RX = channel(OFDM_TX, channelResponse, SNRdb)
    OFDM_RX_noCP = removeCP(OFDM_RX)

    symbol = np.zeros(K, dtype=complex)
    codeword_qam = Modulation(codeword)
    symbol[np.arange(K)] = codeword_qam
    OFDM_data_codeword = symbol
    OFDM_time_codeword = np.fft.ifft(OFDM_data_codeword)
    OFDM_withCP_cordword = addCP(OFDM_time_codeword)
    OFDM_RX_codeword = channel(OFDM_withCP_cordword, channelResponse, SNRdb)
    OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword)
    return np.concatenate((np.concatenate((np.real(OFDM_RX_noCP), np.imag(OFDM_RX_noCP))), np.concatenate((np.real(OFDM_RX_noCP_codeword), np.imag(OFDM_RX_noCP_codeword))))), abs(channelResponse)

Pilot_file_name = 'Pilot_' + str(P)
if os.path.isfile(Pilot_file_name):
    print('Load Training Pilots txt')
    bits = np.loadtxt(Pilot_file_name, delimiter=',')
else:
    bits = np.random.binomial(n=1, p=0.5, size=(K * mu,))
    np.savetxt(Pilot_file_name, bits, delimiter=',')

pilotValue = Modulation(bits)

class BasicBlock(nn.Module):
    """ Supports: groups=1, dilation=1 """
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x1(in_planes, planes, stride)
        # self.conv1 = KANConv1DLayer(in_planes, planes,stride=stride,kernel_size=3)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes * self.expansion)
        # self.conv2 = KANConv1DLayer(planes, planes * self.expansion, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(planes * self.expansion)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        identity = x
        print(f"identity size: {identity.size()}")  # 打印输入张量的尺寸
        out = self.conv1(x)
        print(f"conv1 size: {out.size()}")  # 打印输入张量的尺寸
        out = self.bn1(out)
        print(f"bn1 size: {out.size()}")  # 打印输入张量的尺寸
        out = self.relu(out)
        print(f"relu1 size: {out.size()}")  # 打印输入张量的尺寸
        out = self.conv2(out)
        print(f"conv2 size: {out.size()}")  # 打印输入张量的尺寸
        out = self.bn2(out)
        print(f"bn2 size: {out.size()}")  # 打印输入张量的尺寸
        if self.downsample is not None:
            identity = self.downsample(x)
            print(f"size: {identity.size()}")  # 打印输入张量的尺寸
        out += identity
        out = self.relu(out)
        return out

class FcBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FcBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.prep_channel = 512
        self.fc_dim = 256

        # prep layer2
        self.prep1 = nn.Conv1d(
            self.in_channel, self.prep_channel, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(self.prep_channel)
        # fc layers
        self.fc1 = nn.Linear(self.prep_channel, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        print("Input shape:", x.shape)
        x = self.prep1(x)
        print("After prep1 shape:", x.shape)
        x = self.bn1(x)
        print("After bn1 shape:", x.shape)
        x = x.view(x.size(0), -1)  # 将张量展平成 (batch_size, 512)
        print("After view shape:", x.shape)
        x = self.fc1(x)
        print("After fc1 shape:", x.shape)
        x = self.relu(x)
        print("After relu (fc1) shape:", x.shape)
        x = self.dropout(x)
        print("After dropout (fc1) shape:", x.shape)
        x = self.fc2(x)
        print("After fc2 shape:", x.shape)
        x = self.relu(x)
        print("After relu (fc2) shape:", x.shape)
        x = self.dropout(x)
        print("After dropout (fc2) shape:", x.shape)
        x = self.fc3(x)
        print("After fc3 shape:", x.shape)
        return x

class Kan13Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Kan13Block, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.prep_channel = 512
        self.fc_dim = 256

        # prep layer2
        self.prep1 = nn.Conv1d(
            self.in_channel, self.prep_channel, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(self.prep_channel)
        
        # fc layers
        self.fc1 = nn.Linear(self.prep_channel, 256)
        self.kan1 = KAN([256,18,128])
        
        self.fc2 = nn.Linear(128, 64)
        
        self.fc3 = nn.Linear(64, 32)
        self.kan3 = KAN([32,20,16])
        
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.prep1(x)
        x = self.bn1(x)
        x = self.fc1(x.view(x.size(0), -1))
        x = self.kan1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.kan3(x)
        return x
    
class ResNet(nn.Module):
    """
    ResNet 1D
    in_dim: input channel (for IMU data, in_dim=6)
    out_dim: output dimension (3)
    len(group_sizes) = 4
    """
    def __init__(
        self,
        block_type=BasicBlock,
        in_dim=256,
        out_dim=16,
        group_sizes=[2, 2, 2, 2],
        zero_init_residual=True,
        kan = 3
    ):
        super(ResNet, self).__init__()
        self.base_plane = 64
        self.inplanes = self.base_plane
        
        self.include_top = False
        self.include_top_kan = True

        # Input module
        self.input_block = nn.Sequential(
            nn.Conv1d(
                in_dim, self.base_plane, kernel_size=7, stride=2, padding=3, bias=False
            ),
            nn.BatchNorm1d(self.base_plane),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Residual groups
        self.residual_groups = nn.Sequential(
            self._make_residual_group1d(block_type, 64, group_sizes[0], stride=1),
            self._make_residual_group1d(block_type, 128, group_sizes[1], stride=2),
            self._make_residual_group1d(block_type, 256, group_sizes[2], stride=2),
            self._make_residual_group1d(block_type, 512, group_sizes[3], stride=2),
        )
        self.output_block1 = Kan13Block(512 * block_type.expansion, out_dim)
           
        self._initialize(zero_init_residual)

    def _make_residual_group1d(self, block, planes, group_size, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride=stride, downsample=downsample)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, group_size):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _initialize(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        print("Input shape:", x.shape)  # 打印输入数据的形状
        x = self.input_block(x)
        print("After input block shape:", x.shape)  # 打印输入块后的形状
        x = self.residual_groups(x)
        print("After residual groups shape:", x.shape)  # 打印残差组后的形状
        out = self.output_block1(x)  # mean
        print("Output shape:", out.shape)  # 打印输出的形状
        return out

def training():
    training_epochs = 100
    batch_size = 512
    display_step = 5
    test_step = 1000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True, eps=1e-12
    )
    criterion = nn.MSELoss()
    # optimizer = optim.RMSprop(model.parameters(), lr=0.001)

    H_folder_train = '../H_dataset/'
    H_folder_test = '../H_dataset/'
    train_idx_low = 1
    train_idx_high = 301
    test_idx_low = 301
    test_idx_high = 401

    channel_response_set_train = []
    for train_idx in range(train_idx_low, train_idx_high):
        print("Processing the ", train_idx, "th document")
        H_file = H_folder_train + str(train_idx) + '.txt'
        with open(H_file) as f:
            for line in f:
                numbers_str = line.split()
                numbers_float = [float(x) for x in numbers_str]
                h_response = np.asarray(numbers_float[0:int(len(numbers_float) / 2)]) + 1j * np.asarray(numbers_float[int(len(numbers_float) / 2):len(numbers_float)])
                channel_response_set_train.append(h_response)

    channel_response_set_test = []
    for test_idx in range(test_idx_low, test_idx_high):
        print("Processing the ", test_idx, "th document")
        H_file = H_folder_test + str(test_idx) + '.txt'
        with open(H_file) as f:
            for line in f:
                numbers_str = line.split()
                numbers_float = [float(x) for x in numbers_str]
                h_response = np.asarray(numbers_float[0:int(len(numbers_float) / 2)]) + 1j * np.asarray(numbers_float[int(len(numbers_float) / 2):len(numbers_float)])
                channel_response_set_test.append(h_response)

    print('Length of training channel response', len(channel_response_set_train), 'Length of testing channel response', len(channel_response_set_test))

    train_losses = []
    train_error_rates = []
    test_error_rates = []
    epoch_times = []
    data_load_times = []

    start_time = time.time()

    for epoch in range(training_epochs):
        epoch_start_time = time.time()
        print("Epoch:", epoch)
        avg_cost = 0.
        total_batch = 50

        data_load_start_time = time.time()
        input_samples = []
        input_labels = []
        for index_k in range(0, total_batch * batch_size):
            bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
            channel_response = channel_response_set_train[np.random.randint(0, len(channel_response_set_train))]
            signal_output, para = ofdm_simulate(bits, channel_response, SNRdb)
            input_labels.append(bits[16:32])
            input_samples.append(signal_output)
        # 将 input_samples 和 input_labels 转换为 NumPy 数组
        input_samples = np.asarray(input_samples)
        input_labels = np.asarray(input_labels)
        # 打印转换后的形状
        print("Input samples shape:", input_samples.shape)
        print("Input labels shape:", input_labels.shape)
        data_load_end_time = time.time()
        data_load_times.append(data_load_end_time - data_load_start_time)

        for index_m in range(total_batch):
            batch_x = torch.tensor(np.asarray(input_samples[index_m*batch_size:(index_m+1)*batch_size]), dtype=torch.float32).to(device)
            batch_y = torch.tensor(np.asarray(input_labels[index_m*batch_size:(index_m+1)*batch_size]), dtype=torch.float32).to(device)
            # 打印每个批次数据的形状
            print("Batch X shape:", batch_x.shape)
            print("Batch Y shape:", batch_y.shape)
            # 改变形状为三维张量 (batch_size, 1, feature_size)
            batch_x = batch_x.unsqueeze(2)
            # 打印每个批次数据的形状
            print("Batch X shape:", batch_x.shape)
            print("Batch Y shape:", batch_y.shape)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            avg_cost += loss.item() / total_batch

        train_losses.append(avg_cost)

        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

        if epoch % test_step == 0:
            print('Big Test Set')

            input_samples_test = []
            input_labels_test = []
            test_number = 100

            for test_idx in range(0, test_number):
                bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
                channel_response = channel_response_set_test[np.random.randint(0, len(channel_response_set_test))]
                signal_output, para = ofdm_simulate(bits, channel_response, SNRdb)
                input_labels_test.append(bits[16:32])
                input_samples_test.append(signal_output)

            batch_x = torch.tensor(np.asarray(input_samples_test), dtype=torch.float32).to(device)
            batch_y = torch.tensor(np.asarray(input_labels_test), dtype=torch.float32).to(device)
            # 打印每个批次数据的形状
            print("Batch X shape:", batch_x.shape)
            print("Batch Y shape:", batch_y.shape)
            # 改变形状为三维张量 (batch_size, 1, feature_size)
            batch_x = batch_x.unsqueeze(2)
            # 打印每个批次数据的形状
            print("Batch X shape:", batch_x.shape)
            print("Batch Y shape:", batch_y.shape)

            with torch.no_grad():
                outputs = model(batch_x)
                mean_error = torch.mean(torch.abs(outputs - batch_y))
                mean_error_rate = 1 - torch.mean(torch.mean(((outputs - 0.5).sign().float() == (batch_y - 0.5).sign().float()).float(), 1))
                test_error_rates.append(mean_error_rate.item())
                print("OFDM Detection QAM output number is", 16, ", SNR =", SNRdb, ", Num Pilot =", P, ", Prediction and the mean error on test set are:", mean_error.item(), mean_error_rate.item())

            batch_x = torch.tensor(np.asarray(input_samples), dtype=torch.float32).to(device)
            batch_y = torch.tensor(np.asarray(input_labels), dtype=torch.float32).to(device)
            # 打印每个批次数据的形状
            print("Batch X shape:", batch_x.shape)
            print("Batch Y shape:", batch_y.shape)
            # 改变形状为三维张量 (batch_size, 1, feature_size)
            batch_x = batch_x.unsqueeze(2)
            # 打印每个批次数据的形状
            print("Batch X shape:", batch_x.shape)
            print("Batch Y shape:", batch_y.shape)

            with torch.no_grad():
                outputs = model(batch_x)
                mean_error = torch.mean(torch.abs(outputs - batch_y))
                mean_error_rate = 1 - torch.mean(torch.mean(((outputs - 0.5).sign().float() == (batch_y - 0.5).sign().float()).float(), 1))
                train_error_rates.append(mean_error_rate.item())
                print("Prediction and the mean error on train set are:", mean_error.item(), mean_error_rate.item())

        epoch_end_time = time.time()
        epoch_times.append(epoch_end_time - epoch_start_time)

    total_time = time.time() - start_time
    avg_epoch_time = np.mean(epoch_times)

    # 定义保存路径
    save_dir = '/code/dnn/OFDM_DNN-master/Kan13Block/'
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # Save loss, error rate and time data
    np.save(save_dir+'/train_losses'+"_SNRdb_"+str(SNRdb)+"_Num_Pilot_"+str(P)+'.npy', np.array(train_losses))
    np.save(save_dir+'/train_error_rates'+"_SNRdb_"+str(SNRdb)+"_Num_Pilot_"+str(P)+'.npy', np.array(train_error_rates))
    np.save(save_dir+'/test_error_rates'+"_SNRdb_"+str(SNRdb)+"_Num_Pilot_"+str(P)+'.npy', np.array(test_error_rates))
    np.save(save_dir+'/epoch_times'+"_SNRdb_"+str(SNRdb)+"_Num_Pilot_"+str(P)+'.npy', np.array(epoch_times))
    np.save(save_dir+'/data_load_times'+"_SNRdb_"+str(SNRdb)+"_Num_Pilot_"+str(P)+'.npy', np.array(data_load_times))

training()
