import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from kan import *

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

class KA(nn.Module):
    def __init__(self):
        super(KA, self).__init__()
        self.encoder_h1 = nn.Linear(256, 500)
        self.encoder_h2 = nn.Linear(500, 250)
        self.encoder_h3 = nn.Linear(250, 120)
        self.encoder_h4 = nn.Linear(120, 64)
        self.kan1 = KAN([64,4, 16])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.encoder_h1(x))
        #print("After encoder_h1:", x.size())  # 打印 encoder_h1 后的形状
        x = self.relu(self.encoder_h2(x))
        #print("After encoder_h2:", x.size())  # 打印 encoder_h2 后的形状
        x = self.relu(self.encoder_h3(x))
        #print("After encoder_h3:", x.size())  # 打印 encoder_h3 后的形状
        x = self.relu(self.encoder_h4(x))
        x = self.sigmoid(self.kan1(x))
        #print("After encoder_h4:", x.size())  # 打印 encoder_h4 后的形状
        return x

       
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder_h1 = nn.Linear(256, 500)
        self.encoder_h2 = nn.Linear(500, 250)
        self.encoder_h3 = nn.Linear(250, 120)
        self.encoder_h4 = nn.Linear(120, 16)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.encoder_h1(x))
        #print("After encoder_h1:", x.size())  # 打印 encoder_h1 后的形状
        x = self.relu(self.encoder_h2(x))
        #print("After encoder_h2:", x.size())  # 打印 encoder_h2 后的形状
        x = self.relu(self.encoder_h3(x))
        #print("After encoder_h3:", x.size())  # 打印 encoder_h3 后的形状
        x = self.sigmoid(self.encoder_h4(x))
        #print("After encoder_h4:", x.size())  # 打印 encoder_h4 后的形状
        return x

    
def training():
    training_epochs = 100
    batch_size = 512
    display_step = 5
    test_step = 1000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model =KA().to(device)

    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs")
    #     model = nn.DataParallel(model)

    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)

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
        data_load_end_time = time.time()
        data_load_times.append(data_load_end_time - data_load_start_time)

        for index_m in range(total_batch):
            batch_x = torch.tensor(np.asarray(input_samples[index_m*batch_size:(index_m+1)*batch_size]), dtype=torch.float32).to(device)
            batch_y = torch.tensor(np.asarray(input_labels[index_m*batch_size:(index_m+1)*batch_size]), dtype=torch.float32).to(device)

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

            with torch.no_grad():
                outputs = model(batch_x)
                mean_error = torch.mean(torch.abs(outputs - batch_y))
                mean_error_rate = 1 - torch.mean(torch.mean(((outputs - 0.5).sign().float() == (batch_y - 0.5).sign().float()).float(), 1))
                test_error_rates.append(mean_error_rate.item())
                print("OFDM Detection QAM output number is", 16, ", SNR =", SNRdb, ", Num Pilot =", P, ", Prediction and the mean error on test set are:", mean_error.item(), mean_error_rate.item())

            batch_x = torch.tensor(np.asarray(input_samples), dtype=torch.float32).to(device)
            batch_y = torch.tensor(np.asarray(input_labels), dtype=torch.float32).to(device)

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

    print("Optimization finished")
    print("Total training time: {:.2f} seconds".format(total_time))
    print("Average time per epoch: {:.2f} seconds".format(avg_epoch_time))
    print("Average data loading time per epoch: {:.2f} seconds".format(np.mean(data_load_times)))

    # 定义保存路径
    save_dir = '/code/dnn/OFDM_DNN-master/KA14/'

    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # Save loss, error rate and time data
    np.save(save_dir+'/train_losses'+"_SNRdb_"+str(SNRdb)+"_Num_Pilot_"+str(P)+'.npy', np.array(train_losses))
    np.save(save_dir+'/train_error_rates'+"_SNRdb_"+str(SNRdb)+"_Num_Pilot_"+str(P)+'.npy', np.array(train_error_rates))
    np.save(save_dir+'/test_error_rates'+"_SNRdb_"+str(SNRdb)+"_Num_Pilot_"+str(P)+'.npy', np.array(test_error_rates))
    np.save(save_dir+'/epoch_times'+"_SNRdb_"+str(SNRdb)+"_Num_Pilot_"+str(P)+'.npy', np.array(epoch_times))
    np.save(save_dir+'/data_load_times'+"_SNRdb_"+str(SNRdb)+"_Num_Pilot_"+str(P)+'.npy', np.array(data_load_times))

training()
