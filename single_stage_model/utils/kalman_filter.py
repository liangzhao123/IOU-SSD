import numpy as np
import matplotlib.pyplot as plt

Q = 0.00001
R = 0.1
P_k_k1 = 1
Kg = 0
P_k1_k1 = 1
x_k_k1 = 0
ADC_OLD_Value = 0

kalman_adc_old = 0

class Kalman(object):
    def __init__(self):
        self.Q = 0.00001
        self.R = 0.1
        self.P_k_k1 = 1
        self.Kg = 0
        self.P_k1_k1 = 1
        self.x_k_k1 = 0
        self.ADC_OLD_Value = 0
        self.kalman_adc_old = 0
    def run(self,ADC_Value):

        Z_k = ADC_Value

        if (abs(self.kalman_adc_old - ADC_Value) >= 30):
            x_k1_k1 = ADC_Value * 0.382 + self.kalman_adc_old * 0.618
        else:
            x_k1_k1 = self.kalman_adc_old

        self.x_k_k1 = x_k1_k1
        self.P_k1_k1 = self.P_k1_k1 + self.Q

        self.Kg = self.P_k_k1 / (self.P_k_k1 + self.R)

        kalman_adc = self.x_k_k1 + self.Kg * (Z_k - self.kalman_adc_old)
        self.P_k1_k1 = (1 - self.Kg) * self.P_k_k1
        self.P_k_k1 = self.P_k1_k1

        self.ADC_OLD_Value = ADC_Value
        self.kalman_adc_old = kalman_adc
        return kalman_adc

def kalman(ADC_Value):
    global kalman_adc_old
    global P_k1_k1
    Z_k = ADC_Value

    if (abs(kalman_adc_old - ADC_Value) >= 30):
        x_k1_k1 = ADC_Value * 0.382 + kalman_adc_old * 0.618
    else:
        x_k1_k1 = kalman_adc_old

    x_k_k1 = x_k1_k1
    P_k_k1 = P_k1_k1 + Q

    Kg = P_k_k1 / (P_k_k1 + R)

    kalman_adc = x_k_k1 + Kg * (Z_k - kalman_adc_old)
    P_k1_k1 = (1 - Kg) * P_k_k1
    P_k_k1 = P_k1_k1

    ADC_OLD_Value = ADC_Value
    kalman_adc_old = kalman_adc
    return kalman_adc


a = [100] * 200
array = np.array(a)

s = np.random.normal(0, 10, 200)

test_array = array + s
plt.plot(test_array)
adc = []
kalman = Kalman()
for i in range(200):
    adc.append(kalman.run(test_array[i]))

plt.plot(adc)
plt.plot(array)
plt.show()


# a = [100] * 200
# array = np.array(a)
#
# s = np.random.normal(0, 25, 200)
#
# test_array = array + s
# plt.plot(test_array)
# adc = []
# for i in range(200):
#     adc.append(kalman(test_array[i]))
#
# plt.plot(adc)
# plt.plot(array)
# plt.show()