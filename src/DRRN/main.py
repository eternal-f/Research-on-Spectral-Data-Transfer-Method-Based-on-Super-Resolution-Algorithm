import matplotlib.pyplot as plt

psnr = [65, 75.89, 76.16, 80.06, 79.49, 63.25]
rom = [286, 4757, 5517, 1033, 1574, 0]
name = ['卷积', '残差', '注意力', '循环', '引导注意力', '双三次插值']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams.update({'font.size': 15})
plt.scatter(rom, psnr)
for i in range(len(psnr)):
    plt.annotate(name[i], xy=(rom[i], psnr[i]), xytext=(rom[i]+0.1, psnr[i]+0.1))
plt.xlabel('模型大小')
plt.ylabel('PSNR')
plt.show()
