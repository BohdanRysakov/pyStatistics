import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def average (array):
    # середнє значення
    sum = 0
    for i in range(len(array)):
        sum += array[i]
    return  sum / len(array)


def median(data):
    #Медіана
    sorted_data = sorted(data)
    n = len(data)
    if n % 2 == 0:
        mid = n // 2
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2
    else:
        return sorted_data[n // 2]

def mode(arr):
    #мода
    freq_dict = {}
    for elem in arr:
        if elem in freq_dict:
            freq_dict[elem] += 1
        else:
            freq_dict[elem] = 1

    mode = None
    max_freq = 0
    for key, value in freq_dict.items():
        if value > max_freq:
            max_freq = value
            mode = key

    return mode



def get_variance(data):
    #Дисперсія
    n = len(data)
    mean = sum(data) / n
    cal_variance = sum((x - mean) ** 2 for x in data) / (n - 1)
    return cal_variance

# ======Середньоквадратичне відхилення========
def calculate_sd(data):
    #Середнеквадратичне відхилення
    n = len(data)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / (n - 1)
    sd = math.sqrt(variance)
    return sd

# Варіант 106
#генеруємо масив
#За допомогою
#numpy.random.normal для Python
mean = 5  # середнє значення
variance = 1.2  # дисперсія
size = 124  # кількість чисел
# генеруємо 124 числа
array = np.random.normal(loc=mean, scale=np.sqrt(variance), size=size)
print("масив:")
print(str(array))


print('\n\n')
print('----------------------------------------------------------')
print("Середнє значення: " + str(round(average(array), 3)))
print("Медіана: " + str(round(median(array), 3)))
print("Мода: " + str(round(mode(array), 3)))
print("Дисперсія: " + str(round(get_variance(array), 3)))
print("Середньоквадратичне відхилення: " + str(round(calculate_sd(array), 3)))

#Гістограма частот
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Гістограма частот")
ax.hist(array, edgecolor='black')
fig.show()


#Полігон частот
bins = np.linspace(np.floor(array.min()), np.ceil(array.max()), num=11)
freq, _ = np.histogram(array, bins=bins)


plt.plot(bins[:-1], freq, linestyle='-', marker='o', color='b')
plt.xlabel('Інтервали')
plt.ylabel('Частота')
plt.title('Полігон частот для випадкової вибірки з дисперсією 1.2')
plt.show()


#Діаграма розмаху
plt.title("Діаграма розмаху")
plt.boxplot(array)
plt.show()


#Парето
sorted_array = sorted(array, reverse=True)

cumulative_percentage = 100 * np.cumsum(sorted_array) / sum(sorted_array)


fig, ax1 = plt.subplots()


ax1.bar(range(len(sorted_array)), sorted_array, color='tab:blue')
ax1.set_xlabel('Значення')
ax1.set_ylabel('Кількість', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')


ax2 = ax1.twinx()
ax2.plot(range(len(sorted_array)), cumulative_percentage, color='tab:red', marker='o')
ax2.set_ylabel('Відсоток', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')


ax1.set_xticks(range(len(sorted_array)))
ax1.set_xticklabels(sorted_array)

plt.title('Діаграма Парето')
plt.show()

#Кругова

intArray =array.astype(int)
uniqueArray= np.unique(intArray)
countArray= [np.count_nonzero(intArray == x) for x in uniqueArray]

fig4, ax4 = plt.subplots()
colors = sns.color_palette('pastel')[ 0:5 ]
plt.pie(countArray, labels=uniqueArray,colors = colors,autopct='%.0f%%',textprops={'fontsize': 10})
plt.title('Кругова діаграма')
plt.show()